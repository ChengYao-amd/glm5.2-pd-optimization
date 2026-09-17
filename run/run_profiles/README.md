## How to run

```bash
# Host: run from the repository root on an allocated GPU node.
# Build the image if needed, then create and enter the container.
docker build -t rocm-llm-bench:kernelforge .
EXP=my-profile NAME=glm52-profile bash run/start_container.sh
docker exec -it glm52-profile bash

# Container terminal 1: start the server in the foreground; Ctrl-C stops it.
# Stop any existing server using the same GPUs and port first.
bash /run/run_profiles/up.sh

# Container terminal 2: open another shell in the same container.
# Each benchmark/profile invocation creates a new output directory.
bash /run/run_profiles/bench.sh
bash /run/run_profiles/profile.sh

# Override profiling settings; OUT_DIR must be a fresh directory.
# Capacity/parallelism changes require a server restart and matching client settings.
OUT_DIR="$WORKSPACE_DIR/profile-long" PROFILE_STEPS=24 \
  bash /run/run_profiles/profile.sh

# Diagnostic capture: restart the dedicated server with these flags, then profile.
SGLANG_PROFILE_WITH_STACK=true SGLANG_PROFILE_RECORD_SHAPES=true \
  bash /run/run_profiles/up.sh
```

## Execution steps

- `start_container.sh` mounts the repository's `run/` directory at `/run`, prepares the workspace, and applies the shared patches. `up.sh` loads `/run/env.bashrc` and launches the server without modifying its source.

- The default server uses `/shared_nfs/models/GLM-5.2-MXFP4`, GPUs 0-3, TP4/DP4/EP1, FP8 KV, memory fraction 0.85, and capacity 32. EAGLE uses 5 steps, 6 draft tokens, top-k 1, and simulated acceptance length 3.61. HiCache is disabled.

- `bench.sh` calls `prepare_benchmark.py` to generate `prompts.json` and wait for `/health`. The helper preserves the original 160 synthetic conversation templates and cycles them when more requests are needed. The default readiness timeout is 3600 seconds; override it with `--ready-check-timeout-sec`.

- The native SGLang client shuffles the seed prompts, tokenizes the human text, and repeats or truncates its token IDs to the requested input length. Defaults are 16 warmup requests with up to 32 output tokens, followed by 128 measured requests with 10000 input tokens and 500 output tokens at concurrency 32.

- `profile.sh` runs the same benchmark with profiling enabled. After warmup, it captures 12 forwards by default and stops profiling automatically. Capture includes the batch ramp-up period; it does not wait for full concurrency.

- Each run saves `prompts.json` and `benchmark.jsonl`. Profiles also save per-TP-rank files under `traces/<timestamp>/*.trace.json.gz`, which can be opened in Perfetto. The wrappers return the native client's exit status; they do not perform additional request/trace validation or generate `trace-summary.json`.

## Other notes

- `EXP` defaults to `default`. `WORKSPACE_DIR` defaults to `<repository>/workspace/<EXP>`, and results default to `$WORKSPACE_DIR/run_profiles/bench-<timestamp>` or `profile-<timestamp>`. Use separate `workspace/<EXP>/<node>` directories for multiple nodes. Compilation caches live under `$WORKSPACE_DIR/cache`; redirect console logs into the workspace when needed.

- ROCm 7.2 on this cluster can crash with an NFS-backed `TMPDIR`. `start_container.sh` mounts the node-local `LOCAL_TMP_DIR` at `/tmp` and creates `$WORKSPACE_DIR/runtime-tmp` as a link to it. Access that link on the corresponding node; a local NVMe path can be supplied through `LOCAL_TMP_DIR`.

- The full `/shared_nfs` mount preserves Hugging Face snapshot-to-blob symlinks. Models outside that tree need an appropriate container mount.

- Patch descriptions, source versions, and application order are documented in [patches/readme.md](../patches/readme.md). Container preparation writes `$WORKSPACE_DIR/patches.log` and returns nonzero if patching fails. The patches retain EAGLE compilation, overlap scheduling, and CUDA Graph replay; the client patch adds request and SSE details to JSONL output.

- `DEBUG_CLR_GRAPH_PACKET_CAPTURE=false` is the default for exposing ROCm graph kernels in traces. For normal graph packet capture behavior, stop the server and restart it with `DEBUG_CLR_GRAPH_PACKET_CAPTURE=true bash /run/run_profiles/up.sh` before benchmarking. The two modes have different replay overhead.

- Shape/stack recording defaults to false and can be overridden when launching the server. Treat that trace as diagnostic: instrumentation changes timing, and graph replay alone does not reconstruct every Python callsite or input shape inside the graph.

- Some 64-head MLA calls in the default TP4/DP4 configuration exceed the pinned FlyDSL path's 8/16-head support and use its fallback. Inspect logs and traces for the actual kernels. Fake KV and simulated acceptance are for synthetic decode performance measurements, not model accuracy evaluation.
