#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
OUT_DIR="${OUT_DIR:-${RESULTS_DIR:-$WORKSPACE_DIR/results}/$(date -u +%Y%m%dT%H%M%S-%N)}"
parallel=(--tp-size "$TP" --dp-size "$DP" --ep-size "$EP")
if (( DP > 1 )); then parallel+=(--enable-dp-attention); fi
exec python3 "$PREFILL_DIR/prefill.py" \
    --model-path "$MODEL_PATH" "${parallel[@]}" --result-dir "$OUT_DIR" \
    --batch-size "${BATCH_SIZE:-${CONC:-8}}" --input-len "${ISL:-32768}" \
    --cache-hit-rate "${CACHE_HIT_RATE:-0}" \
    --warmup-steps "${WARMUP_STEPS:-2}" --steps "${STEPS:-5}" \
    --mem-fraction-static "${MEM_FRACTION:-0.85}" \
    --chunked-prefill-size "${CHUNK_SIZE:-32768}" --max-running-requests 256 \
    --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e4m3}" \
    --dsa-prefill-backend "${DSA_PREFILL_BACKEND:-tilelang}" \
    --dsa-decode-backend tilelang --json-model-override-args '{"index_share_for_mtp_iteration":false}' \
    --enable-aiter-allreduce-fusion --enable-fused-qk-norm-rope \
    --disable-cuda-graph --disable-radix-cache --disable-overlap-schedule \
    --trust-remote-code "$@"
