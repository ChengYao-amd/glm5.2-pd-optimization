"""Run three prompt-driven analysis stages in one local Codex SDK thread."""

import argparse
import asyncio
from contextlib import suppress
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import time

from tools.validate_handoff import validate

HERE = Path(__file__).resolve().parent
STAGES = ['01_hotspots', '02_traceback', '03_handoff']


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


async def analyze(args):
    from openai_codex import AsyncCodex, CodexConfig, Sandbox, ApprovalMode
    from openai_codex.errors import InvalidRequestError

    output = args.output_dir.resolve()
    trace = args.trace_dir.resolve()
    if not trace.exists():
        raise ValueError(f'Trace path does not exist: {trace}')
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / 'run.json'
    if state_path.exists() and not args.resume:
        raise ValueError('Output already has run.json; use --resume or a new output directory')
    credentials = {}
    if args.credentials_file.is_file():
        credentials = json.loads(args.credentials_file.read_text())
        os.environ.update(credentials)

    def redact(text):
        for value in credentials.values():
            if value:
                text = text.replace(value, '[REDACTED]')
        return text

    prompt_snapshot = output / 'prompts.json'
    prompts = json.loads(prompt_snapshot.read_text()) if args.resume else {
        name: (HERE / 'prompts' / f'{name}.md').read_text()
        for name in ['system', *STAGES]
    }
    schema_path = output / 'contract.json' if args.resume else HERE / 'handoff.schema.json'
    schema = json.loads(schema_path.read_text())
    hashes = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in prompts.items()}
    state = json.loads(state_path.read_text()) if args.resume else {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'trace_path': str(trace), 'model': args.model, 'effort': args.effort,
        'sdk_version': version('openai-codex'), 'prompt_hashes': hashes,
        'schema_version': 1, 'next_stage': 0, 'thread_id': None, 'stages': [],
    }
    if state['trace_path'] != str(trace) or state['model'] != args.model or state['effort'] != args.effort:
        raise ValueError('Resume must use the same trace, model and effort')
    if args.resume and state['prompt_hashes'] != hashes:
        raise ValueError('Saved prompts do not match this run')
    if args.resume and state['status'] == 'complete':
        errors, tasks = validate(output, args.kernelforge, schema=schema)
        if errors:
            raise ValueError('\n'.join(errors))
        print(f'Already complete: {len(tasks)} validated tasks in {output / "tasks"}')
        return 0
    for name in ['evidence', 'scratch', 'drafts']:
        (output / name).mkdir(exist_ok=True)
    write_json(output / 'contract.json', schema)
    write_json(prompt_snapshot, prompts)
    deadline = time.monotonic() + args.max_minutes * 60
    reserve = min(300, args.max_minutes * 60 * .25)
    context = (
        f'Trace input: {trace}\nOutput directory: {output}\n'
        'Write preliminary analysis.md and candidates.yaml early, then refine them in place. '
        'If a previous stage was interrupted, reconstruct these files from existing evidence first.\n'
        f'KernelForge checkout: {args.kernelforge.resolve()}\n'
        f'Reader: {HERE / "tools/trace_io.py"}\n'
        f'Contract: {output / "contract.json"}\n'
        f'Validator: python {HERE / "tools/validate_handoff.py"} {output} '
        f'--kernelforge {args.kernelforge.resolve()} --task-dir drafts\n'
        f'Reprofiling on this dedicated instance authorized: {args.allow_reprofile}\n'
        f'At most {args.max_reprofiles} additional capture/profiling attempts; record each in analysis.md.\n'
    )
    if args.context_file:
        context += '\nExperiment context:\n' + args.context_file.read_text()
    config = CodexConfig(cwd=str(output), config_overrides=(
        'features.multi_agent=false', 'features.memories=false',
    ))
    options = dict(model=args.model, cwd=str(output), sandbox=Sandbox.full_access,
                   approval_mode=ApprovalMode.deny_all,
                   developer_instructions=prompts['system'])
    state['status'] = 'running'
    write_json(state_path, state)
    events = (output / 'events.jsonl').open('a', buffering=1)

    async def run_turn(thread, prompt, label, seconds):
        turn = await thread.turn(prompt, effort=args.effort)
        state['active_turn_id'] = turn.id
        write_json(state_path, state)
        response = []

        async def consume():
            async for event in turn.stream():
                payload = event.payload.model_dump(mode='json', by_alias=True)
                events.write(redact(json.dumps({'stage': label, 'method': event.method,
                                                'payload': payload})) + '\n')
                if event.method == 'item/completed':
                    item = payload['item']
                    kind = item.get('type')
                    if kind == 'agentMessage':
                        response.append(item.get('text', ''))
                    if kind in ('commandExecution', 'fileChange', 'agentMessage'):
                        detail = item.get('command', item.get('text', ''))
                        print(redact(f'[{label}] {kind}: {str(detail)[:400]}'), flush=True)
                elif event.method == 'thread/tokenUsage/updated':
                    state['usage'] = payload.get('tokenUsage')
                elif event.method == 'turn/completed':
                    if payload['turn']['status'] != 'completed':
                        raise RuntimeError(str(payload['turn']))

        started = time.monotonic()
        consumer = asyncio.create_task(consume())

        async def finish_stream():
            # Keep the SDK queue reader alive until it receives turn/completed.
            # Cancelling stream() first can strand its underlying blocking reader.
            with suppress(InvalidRequestError):  # Turn may have just completed.
                await turn.interrupt()
            try:
                await consumer
            except RuntimeError:
                pass  # Expected interrupted turn status.

        try:
            reminder_after = max(1, seconds - min(90, seconds * .25))
            done, _ = await asyncio.wait({consumer}, timeout=reminder_after)
            if not done:
                with suppress(InvalidRequestError):
                    await turn.steer('Stage deadline is approaching. Stop new investigations, '
                                     'write/update analysis.md and candidates.yaml now, and finish this stage. '
                                     'For handoff packaging, finish drafts and record unresolved evidence explicitly.')
                done, _ = await asyncio.wait({consumer}, timeout=max(1, seconds - (time.monotonic() - started)))
            if not done:
                await finish_stream()
                raise asyncio.TimeoutError(f'{label} exceeded its time budget')
            await consumer
        except asyncio.CancelledError:
            await finish_stream()
            raise
        finally:
            (output / f'{label}.md').write_text(redact('\n\n'.join(response)))
            state['stages'].append({'stage': label, 'elapsed_s': round(time.monotonic() - started, 2)})
            write_json(state_path, state)

    try:
        async with AsyncCodex(config) as client:
            if state['thread_id']:
                thread = await client.thread_resume(state['thread_id'], **options)
            else:
                thread = await client.thread_start(**options)
                state['thread_id'] = thread.id
                write_json(state_path, state)
            for i in range(state['next_stage'], len(STAGES)):
                name = STAGES[i]
                remaining = deadline - time.monotonic()
                seconds = remaining if i == 2 else remaining - reserve
                if i == 0:
                    seconds = min(seconds, args.max_minutes * 60 * .3)
                print(f'[{name}] thread={thread.id}, budget={seconds:.0f}s', flush=True)
                try:
                    await run_turn(thread, context + f'\nThis stage has about {seconds:.0f} seconds.\n'
                                   + prompts[name], name, max(1, seconds))
                except asyncio.TimeoutError:
                    if i == 2:
                        raise
                    print(f'[{name}] time limit; preserving evidence and moving on', flush=True)
                    state['exploration_incomplete'] = True
                state['next_stage'] = i + 1
                write_json(state_path, state)
                if args.stop_after == name:
                    state['status'] = 'paused'
                    write_json(state_path, state)
                    return 0
            for attempt in range(3):
                errors, tasks = validate(output, args.kernelforge, 'drafts', schema)
                write_json(output / 'validation.json', {'valid': not errors, 'errors': errors, 'tasks': tasks})
                if not errors:
                    break
                remaining = deadline - time.monotonic()
                if attempt == 2 or remaining < 15:
                    raise ValueError('Handoff validation failed:\n' + '\n'.join(errors))
                await run_turn(thread, context + '\nFix draft handoffs without changing the contract:\n'
                               + '\n'.join(errors), f'repair_{attempt + 1}', remaining)
            (output / 'tasks').mkdir(exist_ok=True)
            for path in (output / 'drafts').glob('*.yaml'):
                target = output / 'tasks' / path.name
                temporary = target.with_suffix('.tmp')
                temporary.write_bytes(path.read_bytes())
                temporary.replace(target)
            state['status'] = 'partial' if state.get('exploration_incomplete') else 'complete'
            state['tasks'] = tasks
            print(f'{state["status"]}: {len(tasks)} validated handoffs in {output / "tasks"}', flush=True)
            return 0
    except (asyncio.CancelledError, asyncio.TimeoutError):
        state['status'] = 'interrupted'
        raise
    except Exception as exc:
        state['status'] = 'failed'
        state['error'] = redact(str(exc))
        raise
    finally:
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        write_json(state_path, state)
        events.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--kernelforge', type=Path, default=Path('/KernelForge'))
    parser.add_argument('--model', default='gpt-5.6-sol')
    parser.add_argument('--effort', default='max', choices=['low', 'medium', 'high', 'xhigh', 'max'])
    parser.add_argument('--max-minutes', type=float, default=60)
    parser.add_argument('--allow-reprofile', action='store_true')
    parser.add_argument('--max-reprofiles', type=int, default=2)
    parser.add_argument('--context-file', type=Path)
    parser.add_argument('--credentials-file', type=Path, default=Path('/llm_gateway/credentials.json'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', choices=STAGES)
    args = parser.parse_args()
    if args.max_minutes <= 0:
        parser.error('--max-minutes must be positive')

    async def entry():
        task = asyncio.current_task()
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
        return await analyze(args)

    try:
        raise SystemExit(asyncio.run(entry()))
    except (asyncio.CancelledError, KeyboardInterrupt):
        raise SystemExit(130)


if __name__ == '__main__':
    main()
