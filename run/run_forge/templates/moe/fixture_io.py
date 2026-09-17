"""Frozen fixture loading, cache invalidation, and correctness variants."""
import hashlib
import json
import os
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def configure_runtime():
    root = Path(os.environ.get('FORGE_EXPECTED_AITER_ROOT', '/aiter')).resolve()
    watched = [HERE / 'kernel.py', HERE / 'candidate_config.csv', root / 'aiter/fused_moe.py']
    watched.extend(sorted((root / 'aiter/ops/flydsl').rglob('*.py')))
    watched.extend(sorted((root / 'aiter/configs/model_configs').glob('*.csv')))
    h = hashlib.sha256()
    for p in watched:
        h.update(str(p.relative_to(root) if p.is_relative_to(root) else p.name).encode())
        h.update(p.read_bytes())
    fingerprint = h.hexdigest()
    cache = Path(os.environ.get('TMPDIR', '/tmp')) / 'moe-source-cache' / fingerprint[:20]
    cache.mkdir(parents=True, exist_ok=True)
    # FlyDSL's disk key does not cover every helper/closure; source-versioned
    # directories prevent accidentally timing an old binary after a source edit.
    os.environ['FLYDSL_RUNTIME_CACHE_DIR'] = str(cache / 'flydsl')
    os.environ['TRITON_CACHE_DIR'] = str(cache / 'triton')
    os.environ['AITER_USE_FLYDSL_MOE_SORTING'] = '1'
    os.environ['AITER_CONFIG_FMOE'] = str(HERE / 'candidate_config.csv')
    os.environ['AITER_REBUILD'] = '0'
    os.environ.setdefault('AITER_JIT_DIR', str(Path(os.environ.get('TMPDIR', '/tmp')) / 'aiter'))
    import aiter
    actual = Path(aiter.__file__).resolve().parents[1]
    if actual != root:
        raise RuntimeError(f'AITER import mismatch: {actual} != {root}')
    return fingerprint


def contract():
    return json.loads((HERE / 'cases.json').read_text())


def unpack(t, device='cuda'):
    if t is None:
        return None
    x = t['bytes'].to(device).view(getattr(torch, t['dtype'])).reshape(t['shape'])
    if list(x.stride()) != t['stride']:
        raise ValueError('Captured fixture layout was not reconstructed exactly')
    if t.get('is_shuffled'):
        x.is_shuffled = True
    return x


def load_layer(layer, device='cuda', include_output=True):
    cfg = contract()
    path = Path(cfg['fixture_dir']) / f'layer_{layer:02d}.pt'
    identity = cfg['fixture_files'][path.name]
    if path.stat().st_size != identity['size'] or path.stat().st_mtime_ns != identity['mtime_ns']:
        raise RuntimeError(f'Captured fixture identity changed: {path.name}')
    data = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    tensors = {k: unpack(v, device) for k, v in data['tensors'].items() if k != 'output'}
    expected = unpack(data['tensors']['output'], device) if include_output else None
    tensors.update(data['scalars'])
    return tensors, expected


def variant(inputs, mode):
    result = dict(inputs)
    if mode == 'real':
        return result
    if mode == 'scaled':
        result['hidden_states'] = inputs['hidden_states'] * 0.25
    elif mode == 'zeros':
        result['hidden_states'] = torch.zeros_like(inputs['hidden_states'])
    elif mode == 'hot_experts':
        rows = inputs['topk_ids'].shape[0]
        ids = torch.arange(9, dtype=torch.int32, device='cuda').expand(rows, -1).clone()
        ids[:, -1] = 256
        result['topk_ids'] = ids
    elif mode == 'tail48':
        for name in ('hidden_states', 'topk_ids', 'topk_weight'):
            result[name] = inputs[name][:48].contiguous()
    else:
        raise ValueError(f'Unknown correctness variant: {mode}')
    return result
