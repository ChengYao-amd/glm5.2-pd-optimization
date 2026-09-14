#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../env.bashrc"
OUT_DIR="${OUT_DIR:-$RESULTS_DIR/bench-$(date -u +%Y%m%dT%H%M%S-%N)}"
mkdir -p "$(dirname "$OUT_DIR")" "$TMPDIR"
mkdir "$OUT_DIR" || { echo "Choose a fresh OUT_DIR: $OUT_DIR" >&2; exit 1; }

# Prepare the local dataset and wait for the server's PD warmup.
python3 "$HERE/prepare_benchmark.py" "$OUT_DIR/prompts.json" \
    --num-prompts "$NUM_PROMPTS" "$@"

exec python3 -m sglang.benchmark.serving \
    --backend sglang --host 127.0.0.1 --port "$PORT" \
    --model "$MODEL_PATH" --served-model-name glm52 \
    --dataset-name random --dataset-path "$OUT_DIR/prompts.json" --tokenize-prompt \
    --random-input-len "$ISL" --random-output-len "$OSL" --random-range-ratio 1 \
    --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONC" --warmup-requests "$WARMUP_REQUESTS" \
    --fake-prefill --output-details --output-file "$OUT_DIR/benchmark.jsonl" "$@"
