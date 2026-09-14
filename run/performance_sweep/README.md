## How to run

```bash
# Host: run from the repository root on an allocated GPU node.
# Build the image if needed, then create and enter the container.
docker build -t rocm-llm-bench:kernelforge .
EXP=my-performance NAME=glm52-performance bash run/start_container.sh
docker exec -it glm52-performance bash

# Container terminal 1: start the server in the foreground; CONC defaults to 64.
# Stop any existing server using the same GPUs and port first.
bash /run/performance_sweep/up.sh

# Container terminal 2: open another shell in the same container.
# Wait for readiness and sweep 1 2 4 8 16 32 40 64; use a new experiment directory.
OUT_DIR="$WORKSPACE_DIR/performance" bash /run/performance_sweep/sweep.sh

# Append repeats with unchanged server settings and input/output lengths.
# Choose a label that does not reuse existing round names.
OUT_DIR="$WORKSPACE_DIR/performance" bash /run/performance_sweep/sweep.sh \
  --label confirm --concurrencies 32 40 --repeats 2

# Measure one concurrency point; omitting OUT_DIR creates a timestamped directory.
bash /run/performance_sweep/sweep.sh --concurrencies 32

# Rebuild the summary from existing per-round metrics; no server is needed.
# This does not recompute per-round metrics from raw benchmark JSONL.
python3 /run/performance_sweep/analyze.py "$WORKSPACE_DIR/performance"
```

## Execution steps

- `start_container.sh` applies all shared patches, including client instrumentation. `up.sh` loads `/run/env.bashrc`, clears profiling-specific settings, and launches a fake-decode server with normal ROCm graph packet capture behavior. No profiler is started.

- Defaults are GLM-5.2-MXFP4, GPUs 0-3, TP4/DP4/EP1, FP8 KV, memory fraction 0.85, ISL 10000, and OSL 500. EAGLE uses 5 steps, 6 draft tokens, top-k 1, and simulated acceptance length 3.61.

- `sweep.sh` runs `sweep.py`, which waits for `/health`, checks server compatibility and capacity, saves the configuration, and copies the already-patched benchmark client to `bench_with_details.py` for reproducibility. It does not apply patches itself.

- The default concurrency points are `1 2 4 8 16 32 40 64`. Each point measures `max(128, 8 * concurrency)` requests, rounded up to a multiple of concurrency, after `max(16, concurrency)` warmup requests. Warmup output is capped at 32 tokens. Adjust counts with `--min-requests` and `--waves`, and lengths with `ISL` and `OSL`.

- Each round generates local synthetic seed conversations, cycling the original 160 templates as needed. The native random dataset loader tokenizes and repeats/truncates these prompts to the requested length; no dataset download is needed.

- After each round, the script checks request counts, success flags, token lengths, timing, SSE chunk consistency, and throughput. It reconstructs TPOT/ITL statistics and compares them with native SGLang values before saving `requests.csv` and `metrics.json`.

- The final summary combines all completed rounds. It is refreshed when a round fails, including a failed first round. An incomplete round excludes its concurrency from the maximum-passing-concurrency decision.

## Other notes

- Default results go to `$WORKSPACE_DIR/performance_sweep/<timestamp>`, where `WORKSPACE_DIR` normally resolves to `<repository>/workspace/<EXP>`. `OUT_DIR` selects a specific result directory. Use a separate workspace per node; compilation caches and node-local runtime scratch follow the same setup as [run_profiles](../run_profiles/README.md#other-notes).

- Each result directory contains `config.json`, server information, and a client snapshot. Each `rounds/<label>_c<concurrency>_r<repeat>/` directory contains `run.json`, `command.txt`, `prompts.json`, `benchmark.log`, `benchmark.jsonl`, `requests.csv`, and `metrics.json`. Cross-round outputs are `analysis/metrics.csv` and `analysis/summary.json`. Server console logs remain in the startup terminal unless redirected into the workspace.

- The readiness timeout defaults to 3600 seconds (`--ready-timeout`); each benchmark timeout defaults to 2400 seconds (`--benchmark-timeout`). Counts must be positive, OSL must be at least 2, concurrency values must be unique, and labels may contain letters, digits, underscores, or hyphens.

- Decode graph buckets are passed as `1 2 4 8 16 24 32 40 48 56 64`, clipped to the configured capacity and including its upper bound. SGLang interprets buckets as local DP-worker batch sizes and clips them again: default CONC64/DP4 captures `1 2 4 8 16`, so a local batch of 10 pads to 16. Override with `--cuda-graph-bs-decode` on `up.sh`, using local batch sizes.

- `up.sh` forwards extra arguments to SGLang. Changing `CONC`, `TP`, `DP`, `EP`, or other server settings requires a server restart. Use a fresh output directory when server settings or token lengths change.

- Per-request decode rate is `(output_tokens - 1) / (latency - TTFT)`. Targets of 70/80 tokens/s correspond to TPOT limits of 14.2857/12.5 ms. Goodput is the sum of all output tokens from qualifying requests divided by measurement duration. P50, P90, and at least 90% of requests passing are independent rules; every repeat at a concurrency must satisfy the selected rule.

- Native ITL divides each SSE chunk gap among its new tokens; raw chunk gaps are retained separately. TTFT/E2E exclude time waiting for a client concurrency slot. `accept_length` is cumulative over the server lifetime. Configured concurrency is a client limit, not a guarantee of a constant active batch size.

- See [patches/readme.md](../patches/readme.md) for patch scope and pinned source versions. The default 64-head MLA calls can use the existing FlyDSL fallback. Fake KV and simulated acceptance measure synthetic decode performance rather than model accuracy.
