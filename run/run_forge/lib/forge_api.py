"""Small adapter to the pinned KernelForge preparation and measurement APIs."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import time

from .common import activate_forge, case_ids, environment, read_json, write_json


def preflight(bundle, manifest, env, *, repair=False, max_minutes=45):
    activate_forge(manifest['kernelforge'])
    from kernel_agents.loop.task_preparer import preflight_task, prepare_task_sync
    from kernel_agents.config import Config
    repo = bundle / 'repo'
    driver = repo / 'measurement/driver.py'
    ids = case_ids(read_json(repo / 'measurement/cases.json'))
    deadline = time.time() + max_minutes * 60
    kwargs = dict(driver=str(driver), snr_threshold=manifest['snr_threshold'],
                  require_graph=True, require_profile=True, expected_case_ids=ids, deadline_unix=deadline)
    with environment(env):
        result = preflight_task(**kwargs)
        write_json(bundle / 'artifacts/preflight.json', asdict(result))
        if not result.ok and repair:
            config = Config(workspace=str(repo), project_root=manifest['kernelforge'],
                gpu_target='gfx950', gpu_type='mi355x', agent_backend='codex',
                agent_model=manifest['model'], agent_reasoning_effort=manifest['effort'],
                agent_fallback_provider='', experiments_dir=bundle / 'artifacts/preparation')
            prep = prepare_task_sync(config=config, workspace_dir=str(repo),
                kernel=str(repo / manifest['recipe']['kernel']), driver=str(driver),
                program_md=(repo / 'measurement/program.md').read_text(),
                source_files=[str(repo / p) for p in manifest['recipe']['sources']],
                target_functions=manifest['recipe']['targets'], fellow=manifest['recipe']['fellow'],
                snr_threshold=manifest['snr_threshold'], preflight=result,
                invocation_spec_file=str(repo / 'measurement/invocation_spec.json'),
                expected_case_ids=ids, nproc_per_node=manifest['recipe']['nproc'],
                read_only_files=[str(repo / p) for p in manifest['protected_files'] if not p.endswith('/driver.py')],
                deadline_unix=deadline)
            write_json(bundle / 'artifacts/preparation-result.json', asdict(prep))
            result = preflight_task(**kwargs)
            write_json(bundle / 'artifacts/preflight.json', asdict(result))
    if not result.ok or not result.details.get('correctness', {}).get('passed', False):
        raise RuntimeError(f'Driver preflight failed: {result.summary()}')
    return asdict(result)


def measure(bundle, manifest, env, *, label='baseline', repetitions=3):
    activate_forge(manifest['kernelforge'])
    from kernel_agents.mcp_server.tools.bench import bench_wallclock
    from kernel_agents.mcp_server.tools.test import test_correctness
    driver = str(bundle / 'repo/measurement/driver.py')
    async def run():
        correctness = await test_correctness(driver, snr_threshold=manifest['snr_threshold'], timeout_sec=600)
        measurements = [await bench_wallclock(driver, warmup_iters=10, bench_iters=30,
                        timeout_sec=600, repeat=manifest['bench_repeat']) for _ in range(repetitions)]
        return {'correctness': correctness, 'measurements': measurements}
    with environment(env):
        result = asyncio.run(run())
    write_json(bundle / f'artifacts/{label}.json', result)
    # Upper APIs return structured failures; never declare success from mere parsing.
    if not result['correctness'].get('pass', result['correctness'].get('passed', False)):
        raise RuntimeError(f'Correctness failed; see artifacts/{label}.json')
    expected = set(case_ids(read_json(bundle / 'repo/measurement/cases.json')))
    for measurement in result['measurements']:
        reported = set(measurement.get('case_times', {}))
        if not measurement.get('success', False) or reported != expected:
            raise RuntimeError(f'Invalid benchmark case set/result; see artifacts/{label}.json')
    return result
