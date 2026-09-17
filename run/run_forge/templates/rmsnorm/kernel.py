"""Editable candidate; initial implementation is the observed AITER baseline.

Runtime dependencies: PyTorch/ROCm and AITER at the recorded source revision.
Forge may replace this implementation with a self-contained Triton/HIP kernel.
The callable writes two distinct preallocated outputs and preserves all inputs.
"""

from aiter.ops.rmsnorm import add_rmsnorm as _aiter_add_rmsnorm


def fused_add_rmsnorm(x, residual, weight, out, residual_out, eps=1e-5):
    """GLM residual-add RMSNorm, BF16 tensors and FP32 intermediate arithmetic."""
    # Empty tensors are correctness-only; AITER's raw launch has no zero-grid guard.
    if x.shape[0] == 0:
        return
    _aiter_add_rmsnorm(out, x, residual, residual_out, weight, eps, False)
