## How to run

```bash
# Host: use the same image and container setup as run_profiles.
EXP=quick-test NAME=glm52-quick bash run/start_container.sh
docker exec -it glm52-quick bash

# Container: each command starts its own TP workers; no server is needed.
bash /run/quick_test/bench.sh --dry-run
bash /run/quick_test/bench.sh
bash /run/quick_test/profile.sh

# A short decode run; --max-steps 0 runs the full output length.
MAX_STEPS=10 bash /run/quick_test/bench.sh

# Profile measured iterations [20, 32), on all ranks, with graphs disabled.
PROFILE_START=20 PROFILE_STEPS=12 PROFILE_RANKS=all \
  bash /run/quick_test/profile.sh --disable-cuda-graph

# Longer-context TP8/DP8 sweep; EP=1 runs the corresponding no-EP sweep.
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TP=8 DP=8 EP=8 ISL=70000 OSL=10000 \
  bash /run/quick_test/sweep.sh 48 64 96 128

# Check one completed benchmark or collect a sweep; any failed point returns nonzero.
python3 /run/quick_test/collect.py "$WORKSPACE_DIR/quick_test/bench-<timestamp>"
python3 /run/quick_test/collect.py "$WORKSPACE_DIR/quick_test/sweep-<timestamp>" \
  --output "$WORKSPACE_DIR/quick_test/summary.csv"
```

## Configuration and outputs

- Scripts source `run/env.bashrc`, as `run_profiles` does. Defaults are TP4/DP4/EP1,
  batch 32, ISL 10000, OSL 500, FP8 KV and acceptance length 3.61. `DP` must be 1 or
  equal to `TP`; the global batch must divide evenly across attention DP shards.
  EAGLE uses 5 steps, 6 draft tokens and top-k 1.
- `WARMUP_STEPS` defaults to 10. Benchmark `MAX_STEPS` defaults to 0; profiling
  defaults to `PROFILE_START + PROFILE_STEPS`, with start 20 and 12 captured steps.
  `MAX_STEPS=0 bash /run/quick_test/profile.sh` profiles a window in a full run.
- `RESULTS_DIR` defaults to `$WORKSPACE_DIR/quick_test`. `OUT_DIR` selects a fresh
  directory for one run or sweep. Arguments after `bench.sh` / `profile.sh` override
  defaults; unknown arguments pass through to SGLang. Use environment variables for
  TP/DP/EP and batch settings. `--dry-run` prints configuration without loading the
  model or creating output; SGLang validates forwarded flags during a real launch.
- Each point saves `config.json`, `command.txt`, `status.json`, `result.json`, and
  `rank_N.json` / `steps_rank_N.jsonl`. `status.json` records worker exit status and
  wall time. Console output stays in the terminal; redirect it for long runs.
  Sweeps save per-point logs and `summary.csv`, stopping if a point fails.
- Profiles add `traces/decode-TP-N.trace.json.gz`, kernel tables, stage timers and
  per-rank metadata. Open traces in Perfetto. `--profile-with-stack` and
  `--profile-record-shapes` enable diagnostic detail; both default to false.
  `--profile-activities CPU,GPU,MEM` also saves a memory snapshot.
  `--profile-graph-capture-trace` adds graph construction traces as a kernel inventory.
- The implementation targets the SGLang commit pinned by the repository Dockerfile
  (`402df1e1e453e1e85ec0f5ac4052d36598cc691a`). Container setup and shared patches remain
  in `run/start_container.sh`; the quick scripts do not create a second container.

## Measurement basis

The harness loads real target/draft weights, physically initializes synthetic KV and
DSA keys/scales, bootstraps one prompt token, and warms up before timing. It then runs
the complete EAGLE decode loop at a fixed batch. Acceptance is simulated; all ranks
check matching sequence progress each iteration. TP replicas are counted once, while
attention DP shards contribute distinct requests. The last iteration pays its full
compute cost but counts only tokens remaining within OSL.

Loading, graph capture, KV initialization, bootstrap and warmup are excluded from
decode time. Synchronization, cross-rank progress checks and bookkeeping are included.
There is no scheduler, overlap scheduling or HTTP request arrival, so these numbers
measure internal decode cost and are not end-to-end serving throughput.

Profiled runs are marked `is_performance_measurement: false`; use a full non-profiled
run for throughput and TPOT. `collect.py` rejects profiled or incomplete points.
Equivalent sweep points must also have identical realized acceptance lengths.
Kernel tables exclude user annotations, whose GPU time overlaps nested kernels.
Device timers cover the entire measured loop, with a separate profile-window total.
An output length too short to reach the requested profile window returns an error
after saving available diagnostics. Graph packet capture follows `env.bashrc`
(`DEBUG_CLR_GRAPH_PACKET_CAPTURE=false`); keep this setting identical for comparisons.

## Additional tools

```bash
source /run/env.bashrc

# Fixed waves with request warmup; DP1 only. No HTTP server is launched.
python3 /run/quick_test/compare.py --model-path "$MODEL_PATH" \
  --result-dir "$WORKSPACE_DIR/quick_test/compare" \
  --tp-size 4 --ep-size 1 --batch-size 32 --num-requests 128 --warmup-requests 16 \
  --input-len 10000 --output-len 500 --initial-state fake-server \
  --mem-fraction-static 0.85 --enable-aiter-allreduce-fusion --enable-fused-qk-norm-rope

# One-GPU graph visibility probe; no model or SGLang dependency.
HIP_VISIBLE_DEVICES=0 python3 /run/quick_test/graph_probe.py --mode graph
HIP_VISIBLE_DEVICES=0 python3 /run/quick_test/graph_probe.py --mode eager
```

`compare.py` writes `comparison.json` and per-wave summaries. `fake-server` starts
from zero KV/draft state and counts one handoff token separately from computed decode
tokens; `synthetic` uses initialized KV and a real bootstrap. Neither reproduces a
server's recycled KV or scheduler. `WARMUP_STEPS` belongs to `bench.sh`; comparison
warmup uses `--warmup-requests` with up to 32 output tokens per request.

The reference directory's unit tests and historical regression scripts are omitted.
Container wrappers and duplicated EP sweep/verification scripts are consolidated into
the shared container setup, `sweep.sh` and `collect.py`.
