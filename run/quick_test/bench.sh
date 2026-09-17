#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
QUICK_RESULTS_DIR="${RESULTS_DIR:-}"
source "$HERE/../env.bashrc"
RESULTS_DIR="${QUICK_RESULTS_DIR:-$WORKSPACE_DIR/quick_test}"
OUT_DIR="${OUT_DIR:-$RESULTS_DIR/bench-$(date -u +%Y%m%dT%H%M%S-%N)}"

# The internal harness supports DP1 or one attention shard per TP rank.
if [ "$DP" -ne 1 ] && [ "$DP" -ne "$TP" ]; then
    echo "quick_test requires DP=1 or DP=TP" >&2
    exit 1
fi
PARALLEL=(--tp-size "$TP" --ep-size "$EP")
[ "$DP" -gt 1 ] && PARALLEL+=(--enable-dp-attention)

exec python3 "$HERE/decode.py" \
    --model-path "$MODEL_PATH" --result-dir "$OUT_DIR" "${PARALLEL[@]}" \
    --batch-size "$CONC" --input-len "$ISL" --output-len "$OSL" \
    --accept-length "$ACC_LEN" --warmup-steps "${WARMUP_STEPS:-10}" \
    --max-steps "${MAX_STEPS:-0}" --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --mem-fraction-static "$MEM_FRACTION" \
    --enable-aiter-allreduce-fusion --enable-fused-qk-norm-rope "$@"
