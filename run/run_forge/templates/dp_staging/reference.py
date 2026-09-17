"""Protected, independent bit-preserving staging oracle; no candidate imports."""

from __future__ import annotations

import torch


def bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    return (
        actual.shape == expected.shape
        and actual.dtype == expected.dtype
        and bool(torch.equal(actual.view(torch.int16), expected.view(torch.int16)))
    )


def gather_reference(local_input, global_shape, start: int, valid_rows: int):
    result = torch.zeros(global_shape, dtype=local_input.dtype, device=local_input.device)
    result[start : start + valid_rows].copy_(local_input[:valid_rows])
    return result


def scatter_reference(global_input, local_shape, start: int, valid_rows: int):
    result = torch.zeros(local_shape, dtype=global_input.dtype, device=global_input.device)
    result[:valid_rows].copy_(global_input[start : start + valid_rows])
    return result


def make_input(rows: int, hidden: int, variant: int, seed: int):
    """CPU fixtures cover finite values and every BF16 bit pattern, including NaNs."""
    if variant == 0:
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(rows, hidden, generator=generator).to(torch.bfloat16)
    bits = (torch.arange(rows * hidden, dtype=torch.int32) * 17 + seed) % 65536
    return bits.to(torch.int16).view(torch.bfloat16).reshape(rows, hidden)
