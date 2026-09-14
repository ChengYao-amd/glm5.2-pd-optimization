#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export CONC="${CONC:-64}"
source "$HERE/../env.bashrc"

# Restore normal ROCm graph replay for performance measurements.
unset DEBUG_CLR_GRAPH_PACKET_CAPTURE SGLANG_PROFILE_V2
unset SGLANG_PROFILE_WITH_STACK SGLANG_PROFILE_RECORD_SHAPES

# run/start_container.sh prepares the shared SGLang patches.
mkdir -p "$AITER_JIT_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME" "$TMPDIR"
PARALLEL=(--tp "$TP" --ep-size "$EP" --dp "$DP")
[ "$DP" -gt 1 ] && PARALLEL+=(--enable-dp-attention)
GRAPH_BATCHES=()
for batch in 1 2 4 8 16 24 32 40 48 56 64; do
    [ "$batch" -lt "$CONC" ] && GRAPH_BATCHES+=("$batch")
done
GRAPH_BATCHES+=("$CONC")

exec python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" --served-model-name glm52 --trust-remote-code \
    --host 0.0.0.0 --port "$PORT" "${PARALLEL[@]}" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --dsa-prefill-backend flydsl --dsa-decode-backend flydsl --dsa-topk-backend aiter \
    --tool-call-parser glm47 --reasoning-parser glm45 \
    --chunked-prefill-size 32768 --mem-fraction-static "$MEM_FRACTION" \
    --max-running-requests "$CONC" --cuda-graph-max-bs-decode "$CONC" \
    --cuda-graph-bs-decode "${GRAPH_BATCHES[@]}" \
    --speculative-algorithm EAGLE --speculative-num-steps "$SPEC_STEPS" \
    --speculative-eagle-topk "$SPEC_TOPK" --speculative-num-draft-tokens "$SPEC_DRAFT" \
    --enable-aiter-allreduce-fusion --enable-fused-qk-norm-rope \
    --disaggregation-mode decode --disaggregation-transfer-backend fake \
    --watchdog-timeout 1800 --enable-metrics --decode-log-interval 10 "$@"
