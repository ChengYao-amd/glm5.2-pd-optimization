"""Check the analysis handoff contract and local evidence, without judging science."""

import argparse
import ast
from collections import Counter
import json
import math
from pathlib import Path

import jsonschema
import yaml

HERE = Path(__file__).resolve().parents[1]


def backend_names(kernelforge):
    path = Path(kernelforge) / 'src/kernel_agents/fellows/constants.py'
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', '') == 'FELLOW_AGENT_MODULES' for t in node.targets):
            return set(ast.literal_eval(node.value))
    raise ValueError(f'No backend registry in {path}')


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def validate(output, kernelforge, task_dir='tasks', schema=None):
    output = Path(output)
    contract = output / 'contract.json'
    schema = schema or json.loads((contract if contract.exists() else HERE / 'handoff.schema.json').read_text())
    checker = jsonschema.Draft202012Validator(schema)
    backends = backend_names(kernelforge)
    errors, summaries = [], []
    try:
        index = yaml.safe_load((output / 'candidates.yaml').read_text())
    except (OSError, yaml.YAMLError) as exc:
        return [f'candidates.yaml: {exc}'], []
    if not isinstance(index, dict) or not isinstance(index.get('candidates'), list):
        return ['candidates.yaml must contain a candidates list'], []
    rows = index['candidates']
    if any(not isinstance(r, dict) or not isinstance(r.get('id'), str) for r in rows):
        return ['Each candidate must have a string id'], []
    ids = [r['id'] for r in rows]
    errors.extend(f'Duplicate candidate id: {k}' for k, n in Counter(ids).items() if n > 1)
    expected = set()
    for row in rows:
        cid = row['id']
        if row.get('merged_into'):
            target = row['merged_into']
            seen = {cid}
            while target in ids and target not in seen:
                seen.add(target)
                parent = next(r for r in rows if r['id'] == target)
                if not parent.get('merged_into'):
                    break
                target = parent['merged_into']
            else:
                errors.append(f'{cid}: invalid/cyclic merged_into reference')
            continue
        if row.get('task') != f'tasks/{cid}.yaml':
            errors.append(f'{cid}: task must be tasks/{cid}.yaml')
            continue
        expected.add(f'{cid}.yaml')
        path = output / task_dir / f'{cid}.yaml'
        try:
            task = yaml.safe_load(path.read_text())
            structural = sorted(checker.iter_errors(task), key=lambda e: str(e.path))
            errors.extend(f'{cid}:{"/".join(map(str, e.path))}: {e.message}' for e in structural)
            if structural:
                continue
            h = task['x-handoff']
            if task['task_id'] != cid:
                errors.append(f'{cid}: task_id does not match candidate id')
            if set(task['backends']) - backends:
                errors.append(f'{cid}: unsupported backend {set(task["backends"]) - backends}')
            evidence = h['evidence']
            evidence_ids = {e['id'] for e in evidence}
            if len(evidence_ids) != len(evidence):
                errors.append(f'{cid}: duplicate evidence id')
            for e in evidence:
                if not (output / e['path']).is_file():
                    errors.append(f'{cid}: evidence file missing: {e["path"]}')
            for obj in walk(h):
                if set(obj.get('evidence_refs', [])) - evidence_ids:
                    errors.append(f'{cid}: unknown evidence reference')
            for m in h['measurements']:
                if not math.isfinite(m['value']) or not m['evidence_refs']:
                    errors.append(f'{cid}: measurement must be finite and reference evidence')
            for k in task['kernels_to_review']:
                if k['source_path'] and not (output / k['source_path']).is_file():
                    errors.append(f'{cid}: kernel source missing: {k["source_path"]}')
            if h['readiness'] == 'ready_for_handoff':
                if not all([task['backends'], task['shapes']['primary'], task['kernels_to_review'], h['source'], h['inputs'], evidence]):
                    errors.append(f'{cid}: ready task requires backend, shape, source, inputs and evidence')
                if any(not k['source_path'] for k in task['kernels_to_review']):
                    errors.append(f'{cid}: ready task has an unresolved kernel source')
                if any(i['knowledge'] == 'unknown' or not i['spec'] or not i['evidence_refs'] for i in h['inputs']):
                    errors.append(f'{cid}: ready task has an unresolved input')
            if h['readiness'] == 'needs_evidence' and not h['missing_information']:
                errors.append(f'{cid}: needs_evidence requires explicit missing_information')
            summaries.append({'id': cid, 'readiness': h['readiness'], 'priority': task['priority']})
        except (OSError, yaml.YAMLError) as exc:
            errors.append(f'{cid}: {exc}')
    actual = {p.name for p in (output / task_dir).glob('*.yaml')}
    errors.extend(f'Unindexed task: {name}' for name in sorted(actual - expected))
    if not rows and not index.get('no_candidates_reason'):
        errors.append('Empty candidate list requires no_candidates_reason')
    return errors, summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--kernelforge', type=Path, default=Path('/KernelForge'))
    p.add_argument('--task-dir', default='tasks', choices=['tasks', 'drafts'])
    args = p.parse_args()
    errors, tasks = validate(args.output, args.kernelforge, args.task_dir)
    print(json.dumps({'valid': not errors, 'errors': errors, 'tasks': tasks}, indent=2))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
