#!/usr/bin/env python3
"""Materialize an isolated task, validate its driver, and measure its baseline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import socket
import sys
import yaml

from lib.common import (HERE, RECIPES, UPSTREAM_SHA, activate_forge, bundle_manifest,
    case_ids, device_locks, digest, git, load_gateway, lock, protected_hashes, read_json,
    runtime_env, runtime_identity, snapshot_git, validate_analysis, verify_protected, write_json)
from lib.forge_api import measure, preflight


def materialize(args):
    analysis, bundle = args.analysis_dir.resolve(), args.output_dir.resolve()
    recipe = args.recipe
    errors, tasks = validate_analysis(analysis, args.kernelforge)
    if errors:
        raise ValueError('\n'.join(errors))
    parent = next(t for t in tasks if t['id'] == recipe['parent'])
    if parent['readiness'] != 'ready_for_handoff':
        raise ValueError(f'Parent is not ready: {parent}')
    if bundle.exists():
        raise ValueError(f'Output already exists: {bundle}; use --verify or a fresh directory')
    bundle.mkdir(parents=True)
    input_dir = bundle / 'input'
    input_dir.mkdir()
    if args.recipe_file:
        shutil.copy2(args.recipe_file, input_dir / 'recipe.yaml')
    for name in ('candidates.yaml', 'contract.json', 'validation.json', 'analysis.md'):
        if (analysis / name).exists():
            shutil.copy2(analysis / name, input_dir / name)
    original = analysis / 'tasks' / f"{recipe['parent']}.yaml"
    shutil.copy2(original, input_dir / 'original_task.yaml')
    task = yaml.safe_load(original.read_text())
    for evidence in task['x-handoff']['evidence']:
        src = (analysis / evidence['path']).resolve()
        if not src.is_relative_to(analysis):
            raise ValueError(f'External evidence path is unsupported: {src}')
        dst = input_dir / evidence['path']
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    resolved = dict(task)
    resolved.update(task_id=args.task_id, description=recipe['scope'])
    resolved['x-forge'] = dict(parent_task_id=recipe['parent'], scope=recipe['scope'],
                               errata=recipe['errata'])
    if args.task_id == 'tp4_bf16_allreduce_001':
        for inp in resolved['x-handoff']['inputs']:
            if inp['name'] == 'rank_buffer':
                inp['spec']['aliasing'] = 'raw input unchanged; output distinct; SGLang caller writes back separately'
            if inp['name'] == 'collective_parameters':
                inp['spec'].pop('accumulate_template', None)
                inp['spec']['is_broadcast_reg_outptr'] = False
        resolved['constraints'] = [c.replace('in-place output behavior', 'out-of-place operator output and caller-side writeback behavior')
                                   for c in resolved['constraints']]
    (input_dir / 'resolved_task.yaml').write_text(yaml.safe_dump(resolved, sort_keys=False))
    write_json(input_dir / 'errata.json', {'original_sha256': digest(original), 'corrections': recipe['errata']})
    repo = bundle / 'repo'
    repo.mkdir()
    sources = {}
    if recipe['snapshot_aiter']:
        print('Snapshotting AITER tracked sources and submodules...', flush=True)
        sources['aiter'] = snapshot_git(args.aiter_source, repo / 'aiter')
        write_json(input_dir / 'aiter-source-manifest.json', sources['aiter'])
    # All task templates use direct local imports from measurement/.
    template = args.template_dir or HERE / 'templates' / recipe['template']
    shutil.copytree(template, repo / 'measurement',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    if recipe['template'] == 'moe':
        if not args.fixture_dir or not (args.fixture_dir / 'dump.done').exists():
            raise ValueError('MoE requires a completed --fixture-dir capture')
        fixture_dir = args.fixture_dir.resolve()
        shutil.copy2(repo / 'aiter/aiter/configs/model_configs/glm5_fp4_tuned_fmoe.csv',
                     repo / 'measurement/candidate_config.csv')
        fixture_files = {p.name: {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns,
                                 'sha256': digest(p)} for p in sorted(fixture_dir.glob('layer_*.pt'))}
        cfg = {'schema_version': 1, 'task_id': args.task_id, 'fixture_dir': str(fixture_dir),
               'fixture_files': fixture_files,
               'scored_cases': [{'id': f'moe_layer{n:02d}_m192', 'layer_id': n} for n in [3, 39, 77]],
               'correctness_variants': ['real', 'scaled', 'zeros', 'hot_experts', 'tail48'],
               'numerics': {'min_snr_db': 60, 'rtol': 0.05, 'atol': 0.01, 'calibrated': False},
               'scope': recipe['scope']}
        write_json(repo / 'measurement/cases.json', cfg)
    cases = read_json(repo / 'measurement/cases.json')
    ids = case_ids(cases)
    external_fixtures = {}
    if cases.get('fixture_dir') and cases.get('numerics', {}).get('calibrated'):
        fixture_root = Path(cases['fixture_dir']).resolve()
        for name in [*cases['fixture_files'], cases.get('goldens_file', 'goldens.pt')]:
            path = fixture_root / name
            external_fixtures[str(path)] = dict(size=path.stat().st_size,
                mtime_ns=path.stat().st_mtime_ns, sha256=digest(path))
    spec = {'task_id': args.task_id, 'invocation': {'launcher_locator': recipe['kernel'],
            'arguments': [], 'scope': recipe['scope']},
            'tests': {'driver_contract': {'case_selectors': [{'CASE_ID': cid} for cid in ids]},
                      'related_files': ['measurement/driver.py', 'measurement/reference.py', 'measurement/cases.json']}}
    write_json(repo / 'measurement/invocation_spec.json', spec)
    program = f'''# GLM-5.2 optimization task: {args.task_id}

## Exact scope
{recipe['scope']}
Parent analysis task: {recipe['parent']}. This scope takes precedence over the broader parent family.
Synthetic values are identified in cases.json; do not claim captured activations or e2e gains.

## Workload and implementation
MI355X / gfx950. {recipe.get('workload', 'TP4/DP4/EP1 serving, local M=48, global M=192, H=6144.')}.
Only {recipe['nproc']} driver rank(s). Entry: {recipe['kernel']}.
Read measurement/cases.json, invocation_spec.json and the frozen source evidence.
Frozen input evidence directory: {input_dir}
Input/semantic corrections: {recipe['errata']}

## Measurement contract
driver.py (no flags) owns COMPLETE correctness; --bench-mode owns the COMPLETE case suite.
Use graph replay; --profile-run [--profile-case ID] contains only the target path.
Scored case IDs: {ids}. Preserve IDs, distributions, timing boundaries, side effects, and case weights.
SNR gate: {args.snr_threshold} dB plus driver-owned exact/allclose checks. Every failure must exit nonzero.
Do not hoist required transforms, fills, copies, or resets out of the benchmark.
The external runtime sets private caches. Verify compilation and imports use this workspace.

## Modification rules
Editable paths/prefixes: {recipe['editable']}. Add new implementation helpers with candidate_ prefix
under measurement/ if that prefix is allowed. Do not edit any driver/reference/harness/cases/spec/program,
source evidence, original baseline, runtime infrastructure, or files outside this task workspace.
Do not modify the installed /aiter, /sglang or /KernelForge. For standalone components the copied AITER
tree is fixed baseline support; implement optimizations in measurement/kernel.py or candidate_ helpers.
Use measurements to guide optimization; baseline is the original implementation, never trace averages.

## Original analysis (broader scope; use exact scope above)
```yaml
{yaml.safe_dump(resolved, sort_keys=False)}
```
'''
    (repo / 'measurement/program.md').write_text(program)
    (repo / '.gitignore').write_text('__pycache__/\n*.pyc\nforge_experiments/\nagent_logs/\nagent_status.json\n'
        '.cache/\nbuild/\n*.so\n*.o\n*.hsaco\n')
    git(repo, 'init', '-q')
    git(repo, 'checkout', '-qb', 'forge-' + args.task_id)
    git(repo, 'config', 'user.name', 'KernelForge experiment')
    git(repo, 'config', 'user.email', 'kernelforge@local')
    git(repo, 'add', '.')
    git(repo, 'commit', '-qm', f'Pristine task snapshot: {args.task_id}')
    m = {'schema_version': 1, 'task_id': args.task_id, 'parent_task_id': recipe['parent'],
         'bundle_path': str(bundle), 'node': args.node, 'analysis_dir': str(analysis),
         'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'materialized',
         'kernelforge': str(args.kernelforge.resolve()), 'kernelforge_commit': UPSTREAM_SHA,
         'recipe': recipe, 'model': args.model, 'effort': args.effort, 'devices': args.devices,
         'snr_threshold': args.snr_threshold, 'bench_repeat': args.bench_repeat,
         'source_commits': {k: v['commit'] for k, v in sources.items()},
         'original_task_sha256': digest(original), 'pristine_commit': git(repo, 'rev-parse', 'HEAD'),
         'protected_files': protected_hashes(repo),
         'external_fixture_files': external_fixtures,
         'kernelforge_compat': read_json(args.kernelforge / '.forge_compat.json')
                               if (args.kernelforge / '.forge_compat.json').exists() else None,
         'image_id': args.image_id or ((bundle.parent / 'image-id.txt').read_text().strip()
                                      if (bundle.parent / 'image-id.txt').exists() else '')}
    write_json(bundle / 'manifest.json', m)
    return bundle, m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis-dir', type=Path)
    p.add_argument('--task-id')
    p.add_argument('--recipe-file', type=Path, help='Experiment-specific recipe, frozen in the bundle')
    p.add_argument('--template-dir', type=Path, help='Experiment-specific complete measurement template')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--config', type=Path)
    p.add_argument('--kernelforge', type=Path, default=Path('/KernelForge'))
    p.add_argument('--aiter-source', type=Path, default=Path('/aiter'))
    p.add_argument('--fixture-dir', type=Path)
    p.add_argument('--node', default=socket.gethostname())
    p.add_argument('--image-id', default='')
    p.add_argument('--devices', default=None)
    p.add_argument('--model', default='gpt-5.6-sol')
    p.add_argument('--effort', default='max')
    p.add_argument('--snr-threshold', type=float, default=60)
    p.add_argument('--bench-repeat', type=int, default=3)
    p.add_argument('--max-minutes', type=float, default=45)
    p.add_argument('--materialize-only', action='store_true')
    p.add_argument('--verify', action='store_true', help='Verify an existing materialized bundle')
    p.add_argument('--repair-driver', action='store_true', help='Permit pinned Forge preparer to repair driver')
    args = p.parse_args()
    if args.config:
        p.set_defaults(**yaml.safe_load(args.config.read_text()))
        args = p.parse_args()
    args.kernelforge, args.aiter_source = Path(args.kernelforge), Path(args.aiter_source)
    activate_forge(args.kernelforge)
    if not args.verify and (not args.analysis_dir or not args.task_id):
        p.error('--analysis-dir and --task-id are required to materialize')
    if not args.verify:
        args.recipe = load_recipe(args.task_id, args.recipe_file)
        if args.template_dir and not args.recipe_file:
            p.error('--template-dir requires an explicit --recipe-file')
    if not args.devices:
        nproc = args.recipe['nproc'] if not args.verify else 1
        args.devices = ','.join(str(i) for i in range(nproc))
    if args.snr_threshold <= 0 or args.max_minutes <= 0 or args.bench_repeat < 1:
        p.error('Threshold, budget and repeats must be positive')
    bundle, m = bundle_manifest(args.output_dir) if args.verify else materialize(args)
    if args.materialize_only:
        print(f'Materialized: {bundle}')
        return
    with lock(bundle / 'operation.lock'), device_locks(bundle, m):
        verify_protected(bundle, m)
        if m['status'] == 'prepared':
            raise ValueError('Already prepared; use check.py --driver-bundle for revalidation')
        env = runtime_env(bundle, m)
        if args.repair_driver:
            env, _ = load_gateway(env)
        m['runtime_identity'] = runtime_identity(env)
        if m['runtime_identity']['arch'] != 'gfx950' or m['runtime_identity']['gpu_count'] != m['recipe']['nproc']:
            raise ValueError(f'Incorrect target runtime: {m["runtime_identity"]}')
        try:
            preflight(bundle, m, env, repair=args.repair_driver, max_minutes=args.max_minutes)
            measure(bundle, m, env)
            if args.repair_driver:
                # Repair may change driver only; all other measurement inputs stay immutable.
                old_driver = m['protected_files'].pop('measurement/driver.py', None)
                verify_protected(bundle, m)
                m['protected_files']['measurement/driver.py'] = digest(bundle / 'repo/measurement/driver.py')
            else:
                verify_protected(bundle, m)
            git(bundle / 'repo', 'add', 'measurement')
            if git(bundle / 'repo', 'diff', '--cached', '--name-only'):
                git(bundle / 'repo', 'commit', '-qm', 'Freeze verified measurement driver')
            m['pristine_commit'] = git(bundle / 'repo', 'rev-parse', 'HEAD')
            m['status'] = 'prepared'
        except Exception:
            m['status'] = 'preparation_failed'
            write_json(bundle / 'manifest.json', m)
            raise
        write_json(bundle / 'manifest.json', m)
    print(f'Prepared and baselined: {bundle}')


def load_recipe(task_id, path=None):
    recipe = yaml.safe_load(path.read_text()) if path else dict(RECIPES[task_id])
    required = {'template', 'parent', 'kernel', 'nproc', 'fellow', 'framework', 'operator',
                'snapshot_aiter', 'targets', 'sources', 'editable', 'scope', 'errata'}
    if not isinstance(recipe, dict) or required - recipe.keys():
        raise ValueError('Recipe is missing required measurement/optimization boundaries')
    if type(recipe['nproc']) is not int or not 1 <= recipe['nproc'] <= 8:
        raise ValueError('Recipe nproc must be an integer in [1, 8]')
    for field in ('targets', 'sources', 'editable'):
        if not isinstance(recipe[field], list) or not recipe[field]:
            raise ValueError(f'Recipe {field} must be a nonempty list')
    for relative in [recipe['kernel'], *recipe['sources'], *recipe['editable']]:
        p = Path(relative)
        if p.is_absolute() or '..' in p.parts or not p.parts or str(p) == '.':
            raise ValueError(f'Recipe path must stay inside its repository: {relative}')
    if path and not recipe.get('workload'):
        raise ValueError('Custom recipes must describe their actual workload')
    return recipe


if __name__ == '__main__':
    main()
