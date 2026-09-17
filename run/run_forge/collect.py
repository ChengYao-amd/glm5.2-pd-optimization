#!/usr/bin/env python3
"""Summarize a campaign; independently revalidate before exporting source changes."""
import argparse
import json
from pathlib import Path
import statistics
import time

from lib.common import (bundle_manifest, device_locks, git, lock, read_json, runtime_env,
                        runtime_identity, verify_protected, write_json)
from lib.forge_api import measure


def medians(measurement):
    runs = measurement['measurements']
    return {key: statistics.median(run['case_times'][key] for run in runs)
            for key in runs[0]['case_times']}


def kept_revision(state, pristine):
    """The working HEAD may be an unvalidated candidate after interruption."""
    best = state.get('best', {}).get('commit_hash')
    if best:
        return best
    if state.get('cumulative', {}).get('kept', 0):
        raise ValueError('Checkpoint records KEEP but has no best commit; repair checkpoint identity first')
    return pristine


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--report-only', action='store_true', help='No GPU activity or patch publication')
    args = p.parse_args()
    bundle, m = bundle_manifest(args.bundle)
    summary = {'task_id': m['task_id'], 'scope': m['recipe']['scope'], 'e2e_verified': False,
               'pristine_commit': m['pristine_commit'], 'current_head_commit': git(bundle / 'repo', 'rev-parse', 'HEAD')}
    for key, relative in [('run', 'run.json'), ('forge_result', 'artifacts/result.json'),
                           ('native_state', 'repo/forge_experiments/run_state.json')]:
        path = bundle / relative
        if path.exists():
            summary[key] = read_json(path)
    summary['status'] = summary.get('run', {}).get('status', m['status'])
    selected = kept_revision(summary.get('native_state', {}), m['pristine_commit'])
    selected = git(bundle / 'repo', 'rev-parse', '--verify', selected + '^{commit}')
    summary['best_commit'] = selected
    export = bundle / 'export'
    export.mkdir(exist_ok=True)
    if not args.report_only:
        with lock(bundle / 'operation.lock'), device_locks(bundle, m):
            # Do not publish a patch that silently depends on a local untracked helper.
            untracked = git(bundle / 'repo', 'ls-files', '--others', '--exclude-standard').splitlines()
            if untracked:
                raise ValueError(f'Untracked files are not part of a reproducible revision: {untracked[:12]}')
            paths = git(bundle / 'repo', 'diff', '--name-only', m['pristine_commit'], selected).splitlines()
            forbidden = [path for path in paths if not any(path == allowed or
                (allowed.endswith('/') or allowed.endswith('_')) and path.startswith(allowed)
                for allowed in m['recipe']['editable'])]
            if forbidden:
                raise ValueError(f'Changes outside task edit scope: {forbidden}')
            # Test the exact kept commit in its own checkout with cold private caches,
            # leaving an interrupted/dirty original workspace available for resume.
            recheck = bundle / 'rechecks' / f'{selected[:12]}-{time.time_ns()}'
            recheck.mkdir(parents=True)
            git(bundle / 'repo', 'worktree', 'add', '--detach', str(recheck / 'repo'), selected)
            verify_protected(recheck, m)
            fresh = dict(m, task_id=m['task_id'] + '-recheck-' + recheck.name)
            env = runtime_env(recheck, fresh)
            if runtime_identity(env) != m['runtime_identity']:
                raise ValueError('Recheck runtime does not match the prepared baseline')
            candidate = measure(recheck, m, env, label='independent-recheck')
            verify_protected(recheck, m)
            write_json(bundle / 'artifacts/independent-recheck.json', candidate)
            summary['recheck_workspace'] = str(recheck)
            before = medians(read_json(bundle / 'artifacts/baseline.json'))
            after = medians(candidate)
            score = statistics.mean(before[c] / after[c] for c in before)
            summary.update(baseline_case_ms=before, candidate_case_ms=after, mean_case_speedup=score,
                           status='microbench_improved' if score > 1.005 and paths else 'no_improvement')
            counts = m['recipe'].get('workload_counts', {})
            if set(counts) == set(before):
                summary['workload_counts_per_graph_rank'] = counts
                summary['estimated_component_ms_per_graph_rank'] = {
                    'baseline': sum(before[c] * counts[c] for c in counts),
                    'candidate': sum(after[c] * counts[c] for c in counts),
                    'scope': 'Microbenchmark times weighted by source/sequence-derived counts; not e2e latency.'}
            if summary['status'] == 'microbench_improved':
                if m['task_id'] == 'tp4_bf16_allreduce_001':
                    patch = git(bundle / 'repo', 'diff', '--binary', '--relative=aiter',
                                m['pristine_commit'], selected, '--', 'aiter')
                    (export / 'aiter.patch').write_text(patch + '\n')
                else:
                    patch = git(bundle / 'repo', 'diff', '--binary', m['pristine_commit'], selected)
                    (export / 'standalone-kernel.patch').write_text(patch + '\n')
                    summary['integration_required'] = 'Integrate the standalone component into the serving callsite before e2e claims.'
    write_json(export / 'summary.json', summary)
    (export / 'summary.md').write_text(f"# {m['task_id']}\n\nStatus: {summary['status']}\n\n"
        f"Scope: {m['recipe']['scope']}\n\nE2E verified: false.\n\n"
        f"Pristine: `{m['pristine_commit']}`\n\nKept revision: `{summary['best_commit']}`\n\n"
        'See summary.json, artifacts/baseline.json, and artifacts/independent-recheck.json for measured evidence.\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in {'run', 'native_state', 'forge_result'}}, indent=2))


if __name__ == '__main__':
    main()
