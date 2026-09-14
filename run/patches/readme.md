# Shared SGLang patches

This directory is the single source of active patches for both `run_profiles`
and `performance_sweep`. The host-side `run/start_container.sh` creates the container
and applies all four patches to its editable SGLang checkout before reporting
that the container is ready. Neither `up.sh` nor `sweep.py` applies patches.

## Supported source and application order

The patches were validated with the versions pinned in the repository Dockerfile:

- Base image: `lmsysorg/sglang:v0.5.18-rocm720-mi35x`.
- SGLang: `402df1e1e453e1e85ec0f5ac4052d36598cc691a`.
- AITER: `2c71811b32c8ce2e1266aedaec199df7d90f597d`.
- Validation hardware: AMD Instinct MI355X, ROCm 7.2, TP4/DP4/EP1.
- Workload: GLM-5.2-MXFP4, fake decode, FP8 KV, EAGLE 5 steps / 6 draft tokens /
  top-k 1, simulated acceptance length 3.61.

All patch paths are relative to the SGLang checkout, which defaults to `/sglang`
and can be selected with `SGLANG_DIR` when starting the container. There are no
AITER source patches in this directory.

| Order | Patch | Target | Used by |
|---|---|---|---|
| 1 | `fake_decode.patch` | Fake handoff and DSA/KV configuration | Both servers |
| 2 | `fake_dsa_seed.patch` | Initial DSA seed for fake handoff | Both servers |
| 3 | `dp_sparse_graph.patch` | DSA graph selection with DP attention | Both servers |
| 4 | `client_details.patch` | Native serving benchmark client | Both clients |

The order is explicit in `start_container.sh`; it is not a filename glob.
`fake_dsa_seed.patch` uses the `get_disagg` import introduced by
`fake_decode.patch`, so the first two must stay in this order. Merely adding a
patch file here does not activate it: add it to that ordered list as well.

For each patch, startup first runs `git apply --reverse --check`. If this succeeds,
the patch is already present and is skipped. Otherwise startup runs
`git apply --check` followed by `git apply`. Missing patches, incompatible source,
or application failures produce a nonzero startup exit status. The container is
left available for inspection; startup does not print its ready message.

The application log is saved to `$WORKSPACE_DIR/patches.log` and also printed to
the terminal. These changes affect the container checkout, not the Docker image.
Recreating a container reapplies them. Rebuilds are only needed when image
dependencies or pinned sources change. After changing a patch, recreate the
container: reverse-checking cannot upgrade an arbitrary older or partially
applied version of that patch.

## 1. `fake_decode.patch`

**Problem.** A fake transfer does not carry the proposal distribution and remote
KV/indexer metadata that a real prefill worker would send. Treating it like a real
handoff can leave EAGLE rejection sampling without draft probabilities, or select
incompatible shared-top-k layouts.

**Files and changes.**

- `python/sglang/srt/speculative/eagle_disaggregation.py`: when the transfer
  backend is `fake` and rejection sampling is enabled, initialize a point-mass
  draft distribution on the supplied dummy draft token and set its probability
  to one.
- `python/sglang/srt/mem_cache/kv_cache_configurator.py`: allow fake decode to use
  the same shared-top-k index elision as the aggregated path, subject to the
  existing cache eligibility checks.
- `python/sglang/srt/layers/attention/dsa/utils.py`: permit the fused-top-k path
  for fake decode in the local physical-slot domain.

**Scope.** The added branches are gated on fake transfer/decode. They do not
replace proposal distributions or wire layouts for real PD transfers. Fake KV
and the synthetic proposal distribution are intended for performance experiments,
not model accuracy evaluation.

**Origin.** This patch was already present in the profiling workflow. Its archived
source is `Infera-glm-5.2-exp/packups/glm52_fake_tp4ep4_10k500_c16_c32.packup_20260910-055810/patches/`.

## 2. `fake_dsa_seed.patch`

**Problem.** Fake handoff leaves `dsa_topk_indices` unset. The EAGLE worker then
forces the new request out of its draft CUDA graph. With DP attention and overlap,
the resulting eager path can introduce host/device synchronization while other
ranks are already waiting on communication. The observed symptoms included
stalled warmup and requests stuck in metadata preparation or event waits.

**File.** `python/sglang/srt/speculative/eagle_disaggregation.py`.

**Change.** For a DSA model with fake transfer and no supplied DSA seed, construct
an initial seed from the request's allocated local physical KV slots. Positions
outside the request's sequence length are filled with `-1`; the result is an
`int32` tensor. The seed is included in the normal EAGLE input/relay so a new
request can keep using draft CUDA graphs.

**Scope.** The patch only generates a seed when it is missing and the transfer is
fake. It does not synthesize seeds for real transfers or replace an existing seed.
The chosen slots are valid synthetic inputs, not the result of a real prefill
indexer. Independent GPU checks cover physical indices, short-input padding,
proposal probabilities, and isolation of the real-transfer branch.

## 3. `dp_sparse_graph.patch`

**Problem.** Different DP ranks can have different local sequence lengths, and
some can be idle. Choosing dense/sparse DSA graph variants independently is unsafe
for their coordinated execution. The original dispatch can also read a GPU scalar
when there is no CPU sequence-length mirror, introducing synchronization into the
overlap path.

**File.** `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`.

**Change.** When DSA dual-graph support and DP attention are enabled, always select
the sparse graph on every rank. The sparse variant is correct for all sequence
lengths and this decision does not need a device-to-host length read.

**Scope and tradeoff.** The condition is DP attention, not the fake-transfer flag:
it also affects other DP-attention workloads using this patched DSA runner. It
leaves non-DP graph selection unchanged and retains CUDA Graph replay. DP attention
does not use the short-context dense variant after this change; both variants may
still be captured at startup. The validated 10k-input workload already requires
the sparse path. EAGLE compilation and the overlap scheduler remain enabled.

## 4. `client_details.patch`

**Problem.** The native serving benchmark's saved details are insufficient to
independently reconstruct request TPOT and distinguish SSE chunk gaps from the
per-token ITL values used by its statistics.

**File.** `python/sglang/benchmark/serving.py`.

**Change.** Add `raw_chunk_gaps` and `chunk_token_counts` to request output records,
record them for nonempty SGLang stream updates after the first chunk, and include
the following arrays in detailed JSONL output:

- `latencies`: per-request end-to-end client latency in seconds.
- `start_times`: per-request monotonic client start times.
- `successes`: per-request success flags.
- `raw_chunk_gaps`: elapsed seconds between recorded stream chunks.
- `chunk_token_counts`: newly generated tokens carried by those chunks.

The patch preserves the native ITL calculation: each gap is divided by the number
of new tokens in its chunk. It does not measure individual GPU token completion
times or change model execution. There is additional client bookkeeping and
detailed-output size.

**Application change.** This patch formerly targeted a temporary benchmark copy
inside each sweep result directory. Its file headers now target the installed
SGLang client, so it can be applied by the same startup `git apply` loop. Both
workflow clients now use the instrumented source. `sweep.py` still copies that
prepared client to `<OUT_DIR>/bench_with_details.py` to preserve the code used for
the run, but does not patch the copy again.

## Startup and verification

On the allocated host, from the repository root:

```bash
docker build -t rocm-llm-bench:kernelforge .
EXP=my-experiment NAME=dev-container bash run/start_container.sh
```

Then start the desired workflow inside the container, for example:

```bash
docker exec -it dev-container bash /run/run_profiles/up.sh
# Or use /run/performance_sweep/up.sh for the performance sweep server.
```

The `up.sh` scripts now assume successful container preparation. After updating an
existing checkout to this layout, prepare/recreate the container before launching
new runs. The scripts do not apply or repair source patches at server startup.

To check the patch state without starting a model:

```bash
docker exec dev-container bash -c '
  set -e
  source /run/env.bashrc
  for name in fake_decode fake_dsa_seed dp_sparse_graph client_details; do
    git -C "$SGLANG_DIR" apply --reverse --check "/run/patches/$name.patch"
    echo "Verified: $name.patch"
  done
'
```

The original GPU validation, request metrics, and trace checks are recorded under
`record/20260914-debug-validation/`. Patch relocation/startup validation is recorded
under `record/20260914-patch-centralization/`. Historical, superseded experiments
remain in their workspace archives and are not part of this patch list.

Selecting `flydsl` does not guarantee that every MLA call uses a FlyDSL kernel:
the pinned implementation supports 8/16 q heads for that path, while the default
TP4/DP4 configuration can use 64. Such calls use the existing fallback. These
patches do not extend kernel shape support or tune graph buckets.
