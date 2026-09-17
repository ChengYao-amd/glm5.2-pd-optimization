# Captured GLM-5.2 MXFP4 expert-chain task

This template consumes actual TP0 hidden states, routing IDs/weights, preshuffled
FP4 weights, E8M0 scales and original outputs from layers 3/39/77 of a nominal
M=192 TARGET_VERIFY graph. The source benchmark is the existing synthetic fake-PD
workload (TP4/DP4/EP1, concurrency 32, EAGLE six verification tokens).

The scope is **post-router expert execution**: sorting, activation quantization,
GEMM1+SiLU/quantization, GEMM2 and route accumulation. It excludes router projection,
top-k computation, collectives, and serving e2e behavior. Three selected layer
snapshots are equally scored; they do not represent exhaustive temporal/layer coverage.

`kernel.py` is the original AITER boundary wrapper. `candidate_config.csv` is copied
from the original GLM tuned table before preparation. Python/FlyDSL implementation
and dispatch tuning are editable; C++ quantization code remains fixed in this pilot.
Existing tracked files should be used because the current Forge implementer does
not permit new untracked helper files. Driver/reference/fixture_io/cases and goldens
are immutable once preparation passes.

Fixtures store raw bytes plus exact shape/stride/dtype and `is_shuffled`; they must
not be replaced by contiguous semantic BF16 weights. Native captured outputs are
first reproduced by the original isolated implementation. The original implementation
then creates frozen goldens for real/scaled/zero/hot-expert/tail inputs. Per-case SNR
floors use observed baseline variation minus 3 dB, capped at 60 dB, with an overall
floor of 35 dB and additional allclose/exact-zero/input-preservation checks.

Performance cases use only the three real M=192 snapshots. All five input variants
per layer are correctness cases; each is tested through dirty-output graph replays.
Inputs, weights, scales, route IDs and route weights are byte-checked for mutation.
The benchmark includes all necessary sorting/quantization/atomic-output initialization.
The independent zero-output negative test must fail.

`FLYDSL_RUNTIME_CACHE_DIR` and Triton cache are versioned by implementation/config
content because FlyDSL's native cache does not fingerprint all helper changes.
The driver verifies it imported the candidate AITER source tree. External fixture
hashes are frozen in the bundle manifest; size/mtime are checked throughout the run.

Interfaces: no flags = complete correctness; `--bench-mode --warmup N --iters N
--repeat N` = three scored cases; `--profile-run --profile-case <CASE_ID>` = one
target chain, without reference calculations. No eager fallback is allowed.
