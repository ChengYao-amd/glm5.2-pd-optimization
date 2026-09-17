# Local DP fill/copy staging template

This is a **single GPU local component** extracted from 025
`residual_norm_quant_chain_001`, supported by 168 `graph_fill_copy_001` evidence.
It does not execute all-reduce, residual add, RMSNorm, quantization, or the full
four-rank 78-layer boundary chain. Simulating the four rank offsets checks local
placement, not distributed correctness. The parent family's 5.0131% share is
not the optimizable share of this component.

`kernel.py` is the only editable optimization file. It begins with the complete
original SGLang `memcpy_triton.py`, retaining `BLOCK_SIZE=8192`, and adds the
original local `fill_(0)` then `memcpy_triton` operations from gather/scatter.
There is no pre-optimized fusion or zero removal in this baseline.
`source_identity.json` records source and baseline hashes; `source/` preserves
the original bytes used for the extraction. The baseline wrappers omit only
the surrounding ForwardBatch/global-state lookup, gather's already-satisfied
rank branch (TP=DP=4, attention TP size=1), and the collective outside this scope.

The callable ABI is:

```python
gather(global_output, local_input, device_start_row, device_valid_rows) -> None
scatter(local_output, global_input, device_start_row, device_valid_rows) -> None
```

Tensor payloads are contiguous, distinct BF16 local `[48, 6144]` and global
`[192, 6144]`. Metadata consists of CUDA/HIP int64 scalar tensors; their values
can change between graph replays. Gather zeros every output row except the
copied valid local rows at the supplied global offset. Scatter zeros padding
rows after the copied global slice. Inputs and metadata must remain unchanged.
Both operations must launch on the current stream and write the supplied output
on every invocation. Caching outputs, skipping repeated calls, host reads of
metadata, specializing on validation values, or importing protected references
from candidate code are outside the contract.

Only `local_dp_staging_79g78s_48x6144` is scored. It captures 78 repetitions of
gather then scatter followed by the final gather: 79 gathers and 78 scatters,
matching the per-rank stable production frequency. It reuses a fixed pair of
nonaliasing input/output buffers for each role, with independent immutable
gather and scatter inputs. Intervening norms and collectives are omitted, so
the score measures a local staging frequency model and does not reconstruct
their dependencies or memory traffic. Full-batch rank 0 is the measured case;
four simulated ranks and tail cases are correctness coverage. Changing this
sequence, workload frequency, metadata, buffer reuse, or case table is forbidden.

`gather_48x6144` and `scatter_48x6144` are diagnostics. They emit
`diagnostic_ms`, never `case_ms`, because upstream Forge scores every `case_ms`
line and does not support excluding lines through a weight field. The single
scored case makes Forge's equal-case aggregation retain the 79:78 frequency.
Diagnostic timing captures 64 identical role invocations per graph and divides
by 64, preventing host graph-submission gaps from dominating these tiny cases.
Profiling still executes only the selected role per graph invocation.

Run in a GPU environment with PyTorch and Triton:

```bash
python driver.py
python driver.py --bench-mode --warmup 10 --iters 30 --repeat 3
python driver.py --diagnostic-case gather_48x6144 --warmup 10 --iters 30 --repeat 3
python driver.py --diagnostic-case scatter_48x6144 --warmup 10 --iters 30 --repeat 3
python driver.py --profile-run --profile-case gather_48x6144 --warmup 3 --iters 3
```

Correctness uses an independent PyTorch copy oracle and compares BF16 bits,
including signed zeros and NaN payloads. It covers ranks 0–3, valid rows
0/1/42/47/48 for padded 48, the observed 18/24 tail geometry, packed cumsum
offsets, nonzero dirty outputs, input preservation, output guards, changing
input values and device metadata after graph capture, repeated replay, and the
157-operation scored sequence. A failure exits nonzero. `SNR: 100.00 dB` is the
finite parser-compatible exact-match sentinel; actual acceptance is bitwise.

Benchmarking requires successful graph capture and dirty-output verification
before and after each repeat. There is no eager fallback. Each repeat brackets
`iters` graph replays with GPU events, divides by `iters`, and emits one sample;
the score is their median. Profiling runs only the selected target after
initialization/warmup, with no reference or correctness/benchmark reporting.

Freeze `driver.py`, `reference.py`, `graph_harness.py`, `cases.json`, source
identity/snapshots and this contract before optimization. Any candidate must
pass the complete correctness suite, retain the exact scored case IDs, and
pass the real graph replay and profile probes. Source baseline qualification
results belong in the bundle manifest; the final deployed boundary and e2e
serving path still require separate validation.
