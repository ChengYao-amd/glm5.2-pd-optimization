"""Immutable original-implementation goldens and numerical acceptance policy."""
import math
from pathlib import Path

import torch


def load_goldens(config):
    return torch.load(Path(config['fixture_dir']) / 'goldens.pt', map_location='cpu', weights_only=True)


def compare(actual, expected, policy):
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    ref, got = expected.float(), actual.float()
    noise = (ref - got).square().mean().item()
    signal = ref.square().mean().item()
    snr = 200.0 if noise == 0 else -200.0 if signal == 0 else 10 * math.log10(signal / noise)
    max_diff = (ref - got).abs().max().item()
    allclose = torch.allclose(got, ref, rtol=policy['rtol'], atol=policy['atol'])
    if not finite:
        snr = -200.0
    return {'snr_db': snr, 'max_diff': max_diff,
            'ok': finite and allclose and snr >= policy['min_snr_db']}
