"""Frozen independent GLM oracle and deterministic synthetic input generation."""

from __future__ import annotations

import math

import torch


def make_inputs(case, contract, device="cuda"):
    # Generate on CPU so --profile-run adds no random/scale GPU kernels.
    rng = torch.Generator(device="cpu").manual_seed(contract["seed"] + case["rows"])
    shape = (case["rows"], contract["hidden"])
    x = torch.randn(shape, generator=rng)
    residual = torch.randn(shape, generator=rng)
    weight = 1.0 + 0.25 * torch.randn((shape[1],), generator=rng)
    mode = case["mode"]
    if mode == "zeros":
        x.zero_()
        residual.zero_()
    elif mode == "large":
        x.mul_(1.0e4)
        residual.mul_(1.0e4)
    elif mode == "small":
        x.mul_(1.0e-4)
        residual.mul_(1.0e-4)
    elif mode == "cancellation":
        x = (x * 256.0).to(torch.bfloat16).float()
        residual = -x + residual * 0.5
        residual[:, ::4] = -x[:, ::4]
    elif mode == "rounding_sensitive":
        x = x.to(torch.bfloat16).float()
        residual = residual * x.abs().clamp_min(0.125) * 0.00390625
    elif mode == "signed_weight":
        weight = torch.randn((shape[1],), generator=rng)
        weight[::4] = 0.0
        weight[1::4] = -1.0
    elif mode != "random":
        raise ValueError(f"unknown fixed input mode: {mode}")
    return tuple(t.to(dtype=torch.bfloat16).to(device=device) for t in (x, residual, weight))


def reference(x, residual, weight, epsilon):
    summed = x.float() + residual.float()
    inv_rms = torch.rsqrt(summed.square().mean(dim=-1, keepdim=True) + epsilon)
    normalized = (summed * inv_rms * weight.float()).to(torch.bfloat16)
    return normalized, summed.to(torch.bfloat16)


def snr_db(expected, actual):
    if expected.numel() == 0:
        return 100.0
    reference_f = expected.double()
    actual_f = actual.double()
    if not torch.isfinite(actual_f).all().item():
        return -100.0
    error_power = (reference_f - actual_f).square().mean().item()
    if error_power == 0:
        return 100.0
    signal_power = reference_f.square().mean().item()
    if signal_power == 0:
        return -100.0
    return min(100.0, 10.0 * math.log10(signal_power / error_power))


def compare(out, residual_out, expected, contract):
    ref_out, ref_residual = expected
    numerical = contract["numerics"]
    snr = min(snr_db(ref_out, out), snr_db(ref_residual, residual_out))
    normalized_ok = torch.allclose(
        out, ref_out,
        rtol=numerical["normalized_rtol"], atol=numerical["normalized_atol"],
    )
    residual_ok = torch.equal(residual_out, ref_residual)
    ok = normalized_ok and residual_ok and snr >= numerical["min_snr_db"]
    max_diff = (out.float() - ref_out.float()).abs().max().item() if out.numel() else 0.0
    return {
        "ok": bool(ok), "snr_db": snr, "normalized_allclose": normalized_ok,
        "residual_exact": residual_ok, "normalized_max_abs": max_diff,
    }
