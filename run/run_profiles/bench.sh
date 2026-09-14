#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../env.bashrc"
OUT_DIR="${OUT_DIR:-$RESULTS_DIR/bench-$(date -u +%Y%m%dT%H%M%S-%N)}"
mkdir -p "$OUT_DIR"

# Recreate the archived synthetic prompts locally, without a dataset download.
python3 - "$OUT_DIR/prompts.json" "$NUM_PROMPTS" <<'PY'
import json, sys
with open(sys.argv[1], "w") as f:
    json.dump([{"conversations": [
        {"from": "human", "value": f"Example {i}. Describe an integer sequence and its next value."},
        {"from": "gpt", "value": "The next value depends on the rule."}
    ]} for i in range(max(160, int(sys.argv[2])))], f)
PY

exec python3 -m sglang.benchmark.serving \
    --backend sglang --host 127.0.0.1 --port "$PORT" \
    --model "$MODEL_PATH" --served-model-name glm52 \
    --dataset-name random --dataset-path "$OUT_DIR/prompts.json" --tokenize-prompt \
    --random-input-len "$ISL" --random-output-len "$OSL" --random-range-ratio 1 \
    --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONC" --warmup-requests "$WARMUP_REQUESTS" \
    --fake-prefill --output-details --output-file "$OUT_DIR/benchmark.jsonl" "$@"
