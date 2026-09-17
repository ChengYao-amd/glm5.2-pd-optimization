"""Bounded capture at SGLang's routed AITER MoE boundary.

Copies are inserted into the real graph; a later idle-server trigger snapshots
the last M=192 replay. No CUDA tensors are read by Python during graph capture.
"""
import json
import os
from pathlib import Path
import threading
import time

import torch

_layers = {}
_thread = None
SELECTED = {3, 39, 77}
ROOT = Path(os.environ.get('FORGE_MOE_CAPTURE_DIR', '/tmp/forge-moe-capture'))


def pack_tensor(t):
    if t is None:
        return None
    return {'bytes': t.detach().contiguous().view(torch.uint8).cpu(),
            'dtype': str(t.dtype).removeprefix('torch.'), 'shape': list(t.shape),
            'stride': list(t.stride()), 'is_shuffled': bool(getattr(t, 'is_shuffled', False))}


def _save():
    ROOT.mkdir(parents=True, exist_ok=True)
    while not (ROOT / 'dump.trigger').exists():
        time.sleep(1)
    try:
        torch.cuda.set_device(0)
        torch.cuda.synchronize()
        summary = {'created_unix': time.time(), 'scope': 'TP rank 0; last M=192 graph replay per layer',
                   'selected_layers': sorted(SELECTED), 'layers': {}}
        for layer_id, state in sorted(_layers.items()):
            ids = state['topk_ids'].cpu()
            counts = torch.bincount(ids.flatten().to(torch.int64), minlength=257)
            summary['layers'][str(layer_id)] = {
                'shape': list(ids.shape), 'expert_token_histogram': counts.tolist(),
                'active_experts': int((counts > 0).sum()),
                'padded_blocks_32': int(((counts + 31) // 32).sum()),
                'observed_invocations': int(state['counter'].item()),
                'graph_copy_inserted': state['graph_copy_inserted'],
            }
            if layer_id not in SELECTED:
                continue
            tensors = {name: pack_tensor(value) for name, value in state['static'].items()}
            tensors.update({name: pack_tensor(state[name]) for name in ('hidden_states', 'topk_ids', 'topk_weight', 'output')})
            payload = {'layer_id': layer_id, 'tensors': tensors, 'scalars': state['scalars'],
                       'metadata': summary['layers'][str(layer_id)]}
            temporary = ROOT / f'layer_{layer_id:02d}.pt.tmp'
            torch.save(payload, temporary)
            temporary.replace(ROOT / f'layer_{layer_id:02d}.pt')
        (ROOT / 'routing_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        (ROOT / 'dump.done').write_text('complete\n')
        print('FORGE_MOE_CAPTURE_SAVED', len(_layers), sorted(SELECTED), flush=True)
    except Exception as exc:
        (ROOT / 'dump.error').write_text(repr(exc))
        print('FORGE_MOE_CAPTURE_ERROR', type(exc).__name__, flush=True)


def capture_call(original, *, layer_id, **kwargs):
    global _thread
    output = original(**kwargs)
    x, ids = kwargs['hidden_states'], kwargs['topk_ids']
    if tuple(x.shape) != (192, 6144) or tuple(ids.shape) != (192, 9):
        return output
    if layer_id is None or not 3 <= layer_id <= 77 or kwargs['w1'].shape[0] != 257:
        return output
    from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_rank
    if get_tensor_model_parallel_rank() != 0:
        return output
    layer_id = int(layer_id)
    if layer_id not in _layers:
        state = {'topk_ids': torch.empty_like(ids),
                 'topk_weight': torch.empty_like(kwargs['topk_weight']),
                 'counter': torch.zeros((), device=x.device, dtype=torch.int64),
                 'graph_copy_inserted': False}
        if layer_id in SELECTED:
            state['hidden_states'] = torch.empty_like(x)
            state['output'] = torch.empty_like(output)
            state['static'] = {k: v for k, v in kwargs.items()
                               if (isinstance(v, torch.Tensor) or v is None)
                               and k not in ('hidden_states', 'topk_ids', 'topk_weight')}
            state['scalars'] = {k: (v.value if hasattr(v, 'value') else str(v).removeprefix('torch.')
                                   if isinstance(v, torch.dtype) else v)
                                for k, v in kwargs.items() if v is not None and not isinstance(v, torch.Tensor)}
        _layers[layer_id] = state
        print('FORGE_MOE_CAPTURE_REGISTER', layer_id, 'selected', layer_id in SELECTED, flush=True)
    state = _layers[layer_id]
    state['topk_ids'].copy_(ids)
    state['topk_weight'].copy_(kwargs['topk_weight'])
    state['counter'].add_(1)
    if layer_id in SELECTED:
        state['hidden_states'].copy_(x)
        state['output'].copy_(output)
    if torch.cuda.is_current_stream_capturing():
        state['graph_copy_inserted'] = True
    if _thread is None:
        _thread = threading.Thread(target=_save, daemon=True)
        _thread.start()
    return output
