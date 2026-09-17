# Residual-add RMSNorm component template

This one-GPU task isolates the BF16 residual-add RMSNorm observed in task
`residual_norm_quant_chain_001` from the 025 analysis. The only scored case is
`residual_rmsnorm_48x6144`. This is **not** the full norm/staging family: it does
not cover DP gather/scatter, collectives, the no-residual branch, or the full
78-layer boundary sequence. The parent's 5.0131% time share cannot be attributed
to this component. No production frequency is invented for validation tails.

`kernel.py` is the editable candidate and initially calls AITER's original
`aiter.ops.rmsnorm.add_rmsnorm(out, x, residual, residual_out, weight, eps, False)`.
At source revision `2c71811b32c8ce2e1266aedaec199df7d90f597d`, SGLang's
`rmsnorm2d_fwd_with_add` dispatches this same raw entrypoint for BF16 2-D inputs
with hidden size at most 8192. The corresponding source is
`csrc/kernels/rmsnorm_quant_kernels.cu`, kernel
`add_rmsnorm_quant_kernel<bf16,bf16,256,24,true,false,true,1>`.
Its stale Python file header mentioning an Opus-only backend does not describe
the actual `_use_hip_common` dispatch.

The initial candidate requires PyTorch built for ROCm, AITER and its build
dependencies. Use the recorded AITER source tree and an isolated local JIT cache
for preparation; record its source hash and loaded module identity. The
candidate may be replaced by self-contained Triton/HIP code while preserving
the public callable. The frozen measurement does not otherwise import AITER or
SGLang. Keep `driver.py`, `reference.py`, `graph_harness.py`, and `cases.json`
immutable throughout a campaign. Do not use the reference from candidate code.

The reference uses `s = x.float() + residual.float()` and computes
`BF16(s * rsqrt(mean(s*s) + 1e-5) * weight.float())`. The second independent output
is `BF16(s)`. Normalization must use the unrounded FP32 sum. There is no Gemma
`1 + weight` offset and no quantization/scale output. Inputs and outputs are
contiguous and non-aliasing; all input tensors must remain unchanged.

`cases.json` declares synthetic deterministic values, zeros, large/small
magnitudes, cancellation, residual-rounding stress, signed/zero weights, and
correctness-only row counts 0/1/47/49. Both outputs, unchanged inputs, and output
guard rows are checked. Residual output must be exact; the normalized output
requires both `rtol=0.008, atol=2e-6` and at least 60 dB SNR. These tolerances must
be checked against the original AITER run when materializing a bundle.

```bash
export DEBUG_CLR_GRAPH_PACKET_CAPTURE=false
python driver.py
python driver.py --bench-mode --warmup 10 --iters 30 --repeat 3
python driver.py --profile-run --profile-case residual_rmsnorm_48x6144
```

Graph capture is mandatory. NaN-dirty outputs are replayed and checked three
times before timing, so all-zero expected outputs also reject an empty graph.
Capture or validation failures exit nonzero; no eager fallback exists. Every
timed sample executes one graph replay containing 128 calls on the same fixed
inputs and distinct output buffers, and GPU event time is divided by 128. This amortizes the
host submission gap that dominates single-call graph/event timing for tiny
kernels; the fixed count is in `cases.json`. These are repeated calls on the
same tensors, not a measurement of the full layer sequence. Repeats and
per-sample distributions are printed as JSON comments; the scored aggregate is
the median of every sample. Only scored cases emit `case_ms`. The top-level
mean/median are aggregates across the scored case medians, not weighted online
latency estimates.

KernelForge compatibility: the upstream probe patches
`torch.cuda.CUDAGraph.replay`, which this harness calls directly; its required
replay count is met. `--help` advertises `--profile-case` without importing Torch
or AITER. `--profile-run` initializes tensors on CPU, transfers them, warms the
candidate, then captures/replays only the target without references, comparison
kernels, or timing output. The invocation spec must declare exactly
`tests.driver_contract.case_selectors=[{"CASE_ID":"residual_rmsnorm_48x6144"}]`.
KernelForge's preflight checks metric presence, so callers must also require
`details.correctness.passed == true`, not only `preflight.ok`.

On 2026-09-14 this template passed actual MI355X/gfx950 validation in the fresh
`forge-poc-219` container using PyTorch 2.9.1 / ROCm 7.2.0 and an isolated AITER
build. All 11 correctness cases passed; worst SNR was 94.092662 dB, residual
outputs were exact, and all nonempty cases passed dirty graph replay. With
`DEBUG_CLR_GRAPH_PACKET_CAPTURE=false`, the 10-warmup/30-iteration/3-repeat
baseline median was 0.003246594 ms per invocation. Repeat medians were
0.003130656, 0.003327219 and 0.003316281 ms. This is a microbenchmark observation,
not a speedup against the parent trace or an online latency claim.

The upstream graph/profile/case-set preflight passed in that capture mode and
counted 27 actual graph replays (5 required). An intentionally empty candidate
failed benchmark replay validation with exit 1; a BF16-add-before-norm candidate
failed correctness with exit 1 and worst SNR 51.015676 dB. The original target
symbol was confirmed through rocprof kernel tracing. Torch graph setup adds
two int64 RNG-state fill dispatches and ROCm initialization adds one copyBuffer
dispatch; the final three profile replays contain only the target kernel.
See `baseline_validation.json` for the environment, hashes and detailed gates.
