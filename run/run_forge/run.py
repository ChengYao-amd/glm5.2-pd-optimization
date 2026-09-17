#!/usr/bin/env python3
"""Launch/resume one prepared Forge campaign with a cumulative wall-time budget."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

from lib.common import (activate_forge, bundle_manifest, device_locks, git, load_gateway, lock,
    read_json, redact, runtime_env, runtime_identity, verify_compat, verify_protected, write_json)


def command(bundle, m, hours, deadline, resume=False):
    repo, r = bundle / 'repo', m['recipe']
    cmd = [sys.executable, '-m', 'kernel_agents.cli', 'forge-loop', '--workspace', str(repo),
           '--max-hours', str(hours), '--deadline-unix', str(deadline),
           '--agent-backend', 'codex', '--model', m['model'], '--agent-reasoning-effort', m['effort'],
           '--agent-fallback-provider', 'none', '--agent-options-json',
           json.dumps({'home': str(bundle / 'state/codex')}),
           '--profile-timeout-sec', str(m.get('profile_timeout_sec', 7200)),
           '--profiling', '--no-experience-kb',
           '--experiments-dir', str(bundle / 'artifacts/experiments'),
           '--result-json', str(bundle / 'artifacts/result.json')]
    if resume:
        return cmd + ['--resume']
    return cmd + ['--kernel', str(repo / r['kernel']), '--driver', str(repo / 'measurement/driver.py'),
        '--git-branch', 'forge-' + m.get('task_id', r['operator']),
        '--program-md-file', str(repo / 'measurement/program.md'),
        '--invocation-spec-file', str(repo / 'measurement/invocation_spec.json'),
        '--task-type', 'repository', '--framework', r['framework'], '--operator-name', r['operator'],
        '--source-files', ','.join(str(repo / p) for p in r['sources']),
        '--target-functions', ','.join(r['targets']), '--fellow', r['fellow'],
        '--gpu-target', 'gfx950', '--gpu-type', 'mi355x', '--nproc-per-node', str(r['nproc']),
        '--bench-repeat', str(m['bench_repeat']), '--snr-threshold', str(m['snr_threshold']),
        '--no-prepare-task']


def remaining_budget(ledger, hours):
    budget = ledger.get('budget_hours', hours)
    if abs(budget - hours) > 1e-9:
        raise ValueError('Existing cumulative budget differs; resume with the original --max-hours')
    used = sum(s.get('elapsed_seconds', 0) for s in ledger.get('sessions', []))
    return max(0, budget * 3600 - used)


def startup_owner_alive(session):
    """Conservative check for a live owner of a checkpoint-less startup."""
    if session.get('host') != socket.gethostname():
        return False
    for key in ('pid', 'supervisor_pid'):
        pid = session.get(key)
        if not pid:
            continue
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            pass
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--max-hours', type=float, default=12, help='Total cumulative campaign budget, including resumed sessions')
    p.add_argument('--allocation-deadline-unix', type=float, default=0,
                   help='Allocation end time; Forge ends 5 minutes early, watchdog stops any remaining children')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--retry-start', action='store_true', help='Retry a failed startup that never created a native checkpoint')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--credentials-file', default='/llm_gateway/credentials.json')
    p.add_argument('--gateway-config', default='/llm_gateway/config.toml')
    args = p.parse_args()
    if args.max_hours < 1:
        p.error('--max-hours must be at least 1')
    bundle, m = bundle_manifest(args.bundle)
    activate_forge(m['kernelforge'])
    verify_compat(m)
    verify_protected(bundle, m)
    if m['status'] != 'prepared':
        raise ValueError('Bundle must pass preparation before optimization')
    ledger_path = bundle / 'run.json'
    ledger = read_json(ledger_path) if ledger_path.exists() else {'budget_hours': args.max_hours, 'sessions': []}
    seconds = remaining_budget(ledger, args.max_hours)
    if seconds < 60:
        raise ValueError('Cumulative campaign budget exhausted')
    now = time.time()
    deadline = now + seconds
    if args.allocation_deadline_unix:
        deadline = min(deadline, args.allocation_deadline_unix - 300)
    if deadline - now < 300:
        raise ValueError('Insufficient allocation time to start and finalize a session')
    # Upstream uses max-hours for its iteration-loop clock and the absolute
    # deadline mainly for Analysis. Clamp both clocks to this allocation.
    cmd = command(bundle, m, max(1, (deadline - now) / 3600), deadline, args.resume)
    if args.dry_run:
        print(json.dumps({'argv': cmd, 'remaining_budget_seconds': seconds,
                          'effective_session_seconds': deadline - now, 'devices': m['devices']}, indent=2))
        return
    with ExitStack() as stack:
        stack.enter_context(lock(bundle / 'operation.lock'))
        stack.enter_context(device_locks(bundle, m))
        native_state = bundle / 'repo/forge_experiments/run_state.json'
        if args.resume and not native_state.exists():
            raise ValueError('No native resume checkpoint exists')
        if args.retry_start and (args.resume or native_state.exists() or
                startup_owner_alive(ledger['sessions'][-1] if ledger['sessions'] else {}) or
                git(bundle / 'repo', 'rev-parse', 'HEAD') != m['pristine_commit']):
            raise ValueError('Startup retry requires a stopped pristine task with no native checkpoint')
        if not args.resume and (native_state.exists() or ledger['sessions'] and not args.retry_start):
            raise ValueError('Existing session; use --resume (or inspect failed startup)')
        if not args.resume and git(bundle / 'repo', 'status', '--porcelain', '--untracked-files=no'):
            raise ValueError('Pristine workspace is dirty')
        env, secrets = load_gateway(runtime_env(bundle, m), args.credentials_file, args.gateway_config)
        current = runtime_identity(env)
        if current != m['runtime_identity']:
            raise ValueError(f'Runtime identity changed: expected {m["runtime_identity"]}, got {current}')
        # This validates the actual Forge provider/gateway without making an LLM request.
        from lib.common import environment
        with environment(env):
            from forge_llm.agent_backends.codex import CodexBackend
            CodexBackend().preflight()
        log_dir = bundle / 'artifacts/logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        index = len(ledger['sessions']) + 1
        log_path = log_dir / f'forge-{index:03d}.log'
        session = {'started_unix': time.time(), 'host': socket.gethostname(), 'resume': args.resume,
                   'supervisor_pid': os.getpid(),
                   'deadline_unix': deadline, 'allocation_deadline_unix': args.allocation_deadline_unix,
                   'elapsed_seconds': 0, 'status': 'starting', 'log': str(log_path), 'argv': cmd}
        ledger['sessions'].append(session)
        ledger['status'] = 'starting'
        write_json(ledger_path, ledger)
        try:
            proc = subprocess.Popen(cmd, cwd=bundle / 'repo', env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        except Exception as exc:
            session.update(status='startup_failed', error=redact(str(exc), secrets),
                           elapsed_seconds=time.time() - session['started_unix'], ended_unix=time.time())
            ledger['status'] = 'startup_failed'
            write_json(ledger_path, ledger)
            raise
        session.update(status='running', pid=proc.pid, supervisor_pid=os.getpid())
        ledger['status'] = 'running'
        write_json(ledger_path, ledger)
        terminate_at = [None]
        def stop(signum, frame):
            if terminate_at[0] is None:
                terminate_at[0] = time.monotonic()
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        def reader():
            with log_path.open('a', buffering=1) as log:
                for line in proc.stdout:
                    line = redact(line, secrets)
                    log.write(line)
                    print(line, end='', flush=True)
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        while proc.poll() is None:
            try:
                verify_protected(bundle, m)
            except ValueError as exc:
                session['integrity_error'] = str(exc)
                stop(signal.SIGTERM, None)
            # Native Forge receives the earlier deadline. This also bounds a hung finalizer.
            if time.time() >= deadline + 60:
                stop(signal.SIGTERM, None)
            if terminate_at[0] is not None and time.monotonic() - terminate_at[0] > 45:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            session['elapsed_seconds'] = time.time() - session['started_unix']
            session['heartbeat_unix'] = time.time()
            write_json(ledger_path, ledger)
            time.sleep(5)
        thread.join(timeout=10)
        session.update(elapsed_seconds=time.time() - session['started_unix'],
                       ended_unix=time.time(), exit_code=proc.returncode)
        session['status'] = 'finished' if proc.returncode == 0 else 'interrupted_or_failed'
        ledger['status'] = session['status']
        ledger['remaining_budget_seconds'] = remaining_budget(ledger, args.max_hours)
        write_json(ledger_path, ledger)
        print(f'Session ended; remaining budget {ledger["remaining_budget_seconds"] / 3600:.3f}h; checkpoint {native_state}')
        raise SystemExit(proc.returncode)


if __name__ == '__main__':
    main()
