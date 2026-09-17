# GLM TP4 all-reduce measurement bundle

Copy `driver.py`, `reference.py`, `graph_harness.py`, and `cases.json` together
into the bundle's `repo/measurement/` directory and freeze their hashes before
optimization. All three scored cases have BF16 input/output `[192,6144]`, use
four ranks, and transfer 2,359,296 input bytes per rank. The fixture inputs are
deterministic synthetic values, not captured model activations.

The candidate is `CustomAllreduce.custom_all_reduce(x)`, the raw communicator
called by the production tensor-parallel wrapper. It returns an independent
output and must preserve input storage and values. Caller-side gather writeback
does not belong to this operator's timing. The kernel's third template parameter
is `is_broadcast_reg_outptr=false`; the original kernel accumulates in FP32 and
rounds to BF16. `false` does not disable accumulation.

Set the candidate root externally using `PYTHONPATH`. Set
`FORGE_EXPECTED_AITER_ROOT` (or `FORGE_AITER_ROOT`) to the same source snapshot
and set `AITER_JIT_DIR` to a task-specific directory outside the source tree,
preferably on node-local `/tmp`. `AITER_META_DIR`, if present, must name the same
root. The driver verifies imported Python/source locations and loaded custom
all-reduce extension. It hashes custom kernel/pybind sources, all files under
`csrc/include`, AITER distributed Python code, the relevant op wrappers, JIT
configuration, and toolchain environment. A fingerprint change forces a full,
serialized JIT rebuild. Cache reuse requires matching source and `.so` digests.
The framework should also record the full source snapshot identity and freeze
all dependencies outside the task's explicit editable range.

Required runtime: Linux, four mutually accessible ROCm GPUs, PyTorch distributed
with RCCL/Gloo, the candidate AITER Python/build dependencies, HIP compiler and
headers, the source's Opus/includes, and sufficient shared memory for IPC. The
original source uses AITER's real `graph_capture()` and communicator capture
registration; an unavailable communicator or disabled registered-input path is
a hard error. No SGLang import is required for this raw operator.

Example commands after setting the environment:

```bash
python measurement/driver.py --dump-json artifacts/correctness.json
python measurement/driver.py --bench-mode --warmup 10 --iters 30 --repeat 3 --dump-json artifacts/baseline.json
python measurement/driver.py --profile-run --profile-case moe_out_192x6144 --warmup 10 --iters 3
python -m torch.distributed.run --standalone --nproc-per-node=4 measurement/driver.py --profile-run --profile-case gather_192x6144 --iters 3
```

Without `RANK`, the driver self-launches four torchrun workers and accepts
`FORGE_NPROC_PER_NODE=4`. With `RANK`, it uses the launcher's process group
environment, requires `WORLD_SIZE=4`, and binds `LOCAL_RANK` (falling back to
`RANK` when omitted). It remains in its caller's process group for timeout
cleanup. Exactly four visible GPUs are required for the self-launch branch.

Correctness always runs the complete three-case suite. `--shape default` and
`--mode smoke|stability|determinism` support Forge preparer probes without
reducing coverage. Each case runs four numerical modes (nominal, zero,
cancellation, alternating large/small columns), eager repeats, two distinct
payloads per captured graph, dirty/zero outputs, skewed rank arrival, and queued
replays. The oracle gathers actual rank inputs, sums explicitly in FP32, and
casts to BF16. Every rank contributes to the minimum SNR, maximum difference,
and combined pass result. Additional exact checks enforce gather placement,
zero results, input preservation, out-of-place storage, and deterministic replay.
A failed semantic or numerical check returns a nonzero exit status.

Benchmarking captures a chain of `--iters` raw collectives and only times graph
replay. Each GPU records events, every sample takes the slowest rank, and each
case uses the median of five samples. `--repeat` repeats the entire three-case
sweep; final per-case values are medians across repeats. `mean_ms` is the
arithmetic mean of the three final case values; source identity, all rounds,
and raw samples are retained in `--dump-json`. The driver prints one
`case_ms: <id> <ms>` per frozen case and `mean_ms: <ms>` from rank zero.

Profiling has a separate execution path with initialization, eager warmup,
graph capture/IPC registration, and `--iters` replays; it does not run an oracle,
correctness comparisons, or benchmark measurement. Omitting `--profile-case`
selects only `gather_192x6144`. Unknown IDs fail argument parsing.

Initial calibration on 2026-09-14 used `forge-poc-025` on
`crsuse2-m2m-025`, PyTorch `2.9.1+rocm7.2.0.git7e1940d4`, HIP
`7.2.26015-fc0010cf6a`, and independently rebuilt original `/aiter` sources.
All 80 checks per case per rank were bit exact: SNR 200 dB (the exact-match
reporting cap), max difference zero. The frozen gate is SNR >=60 dB,
`rtol=1/128`, `atol=2^-16`, with the separate exact semantic checks above.
This leaves room for finite FP32 accumulation reordering while rejecting
BF16 accumulation error. Prepare must revalidate each candidate source
snapshot and measure its own baseline; calibration results here do not replace
that GPU preflight.

The same initial validation measured these graph latencies with warmup 10,
chain length 30, and three complete repeats:

| Case | Median latency (ms) | Three-repeat spread |
|---|---:|---:|
| `gather_192x6144` | 0.023946899 | 1.1150% |
| `moe_out_192x6144` | 0.023952234 | 0.7458% |
| `dense_out_192x6144` | 0.023964234 | 0.4732% |

Mean latency was 0.023954456 ms. Source/build fingerprint was
`58b21f259423bb791bae529e14a875909c9c0af2ab5ef124f238275be7196f8c`;
the independently compiled custom-allreduce `.so` SHA256 was
`d6aeeb5a5f73d5584ee9ec3340793f5ab7d4ad309ba539580bb464dc9173b840`.
Raw correctness and timing records were written to
`forge-poc-025:/tmp/forge-ar-correctness.json` and
`forge-poc-025:/tmp/forge-ar-benchmark.json`.

All three explicit profile selectors passed under external torchrun with four
ranks, warmup 2, and three graph replays. They emitted no SNR, allclose, case_ms,
or mean_ms records. An independent temporary worker that deliberately replaced
the candidate by `x.clone()` produced `allclose: False` and process exit 1;
an unknown profile-case ID was rejected with exit 2. That fault injection is
not part of the frozen driver interface.
