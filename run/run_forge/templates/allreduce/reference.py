# SPDX-License-Identifier: MIT
"""Frozen, independent inputs and FP32 SUM oracle for GLM TP4 all-reduce.

The values are synthetic; no model activation distribution is claimed.
No AITER code participates in the oracle. RCCL only gathers the BF16 inputs;
addition is performed explicitly in FP32 in rank order, then rounded to BF16.
"""
from __future__ import annotations

import math

import torch
import torch.distributed as dist


def make_input(case_id: str, rank: int, device, seed: int, mode: str = "nominal"):
    shape = (192, 6144)
    gen = torch.Generator(device=device).manual_seed(seed + 1009 * rank)
    if case_id == "dense_out_192x6144":
        x = torch.rand(shape, generator=gen, device=device) * 4 - 2
        x += (rank - 1.5) / 8
    else:
        x = torch.randn(shape, generator=gen, device=device)
        if case_id == "moe_out_192x6144":
            x *= (rank + 1) / 4

    if mode == "zeros":
        x.zero_()
    elif mode == "cancellation":
        # Rank pairs cancel, with a small exactly representable remainder.
        # At these magnitudes every FP32 addition is exact; summing in BF16
        # loses the low terms and fails this independent reference.
        idx = torch.arange(x.numel(), device=device).reshape(shape)
        common = ((idx + seed) % 127 - 63).float()
        if rank == 0:
            x = common * 128
        elif rank == 1:
            x = -common * 128
        elif rank == 2:
            x = ((idx + seed) % 17 - 8).float() / 64
        else:
            x = ((idx + seed) % 13 - 6).float() / 128
    elif mode == "large_small":
        # Adjacent large/small columns exercise both ranges without FP32
        # overflow or a mathematically ambiguous catastrophic cancellation.
        scales = torch.where(
            torch.arange(shape[1], device=device) % 2 == 0, 65536.0, 1.0 / 65536
        )
        x *= scales
    elif mode != "nominal":
        raise ValueError(f"unsupported numerical mode: {mode}")

    x = x.to(torch.bfloat16)
    if case_id == "gather_192x6144":
        x[:rank * 48].zero_()
        x[(rank + 1) * 48:].zero_()
    elif case_id not in ("moe_out_192x6144", "dense_out_192x6144"):
        raise ValueError(f"unknown case_id: {case_id}")
    return x


def reference_sum(x, group):
    rank_inputs = [torch.empty_like(x) for _ in range(4)]
    dist.all_gather(rank_inputs, x, group=group)
    acc = rank_inputs[0].float()
    for value in rank_inputs[1:]:
        acc = acc + value.float()
    return acc.to(torch.bfloat16)


def compare(reference, output, original_input, current_input, case_id, gates):
    valid_tensor = (
        isinstance(output, torch.Tensor)
        and output.shape == reference.shape
        and output.dtype == torch.bfloat16
        and output.device == reference.device
        and output.is_contiguous()
    )
    if not valid_tensor:
        return {"snr_db": -200.0, "max_diff": float("inf"), "allclose": False,
                "finite": False, "input_preserved": False, "out_of_place": False,
                "exact_required": False}
    ref64 = reference.double()
    diff = output.double() - ref64
    finite = bool(torch.isfinite(output).all().item())
    noise = diff.square().sum().item()
    power = ref64.square().sum().item()
    snr = 200.0 if noise == 0 else (
        10 * math.log10(power / noise) if power > 0 and math.isfinite(noise) else -200.0
    )
    # Gather is a placement operation: only one rank contributes each element,
    # and every all-zero output is also exactly specified.
    exact_required = case_id.startswith("gather_") or not bool(reference.any().item())
    return {
        "snr_db": min(200.0, snr),
        "max_diff": diff.abs().max().item() if finite else float("inf"),
        "allclose": bool(torch.allclose(output, reference, rtol=gates["rtol"], atol=gates["atol"])),
        "finite": finite,
        "input_preserved": bool(torch.equal(original_input, current_input)),
        "out_of_place": output.untyped_storage().data_ptr() != current_input.untyped_storage().data_ptr(),
        "exact_required": not exact_required or bool(torch.equal(output, reference)),
    }


def passed(result, gates):
    return result["snr_db"] >= gates["snr_threshold_db"] and all(
        result[key] for key in ("allclose", "finite", "input_preserved", "out_of_place", "exact_required")
    )
