#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
NAME="${NAME:-glm52-quick-prefill}"
image_id=$(docker image inspect --format '{{.Id}}' "$IMAGE")
[[ "$image_id" == "$EXPECTED_IMAGE_ID" ]] || {
    echo "Image mismatch: $image_id; expected $EXPECTED_IMAGE_ID" >&2; exit 1;
}
cache="$WORKSPACE_DIR/cache/$(hostname -s)/${image_id#sha256:}"
mkdir -p "$cache"
# A fresh name is required; never replace an existing container.
docker run -d --init --name "$NAME" --network host --ipc host --shm-size 32g \
    --device /dev/kfd --device /dev/dri --group-add video --group-add render \
    --cap-add SYS_PTRACE --cap-add IPC_LOCK --security-opt seccomp=unconfined \
    --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 \
    -v /shared_nfs:/shared_nfs -v "$MODEL_PATH:$MODEL_PATH:ro" \
    -v "$PREFILL_DIR:/quick_test_prefill:ro" -v "$cache:/aiter-jit" \
    -v "$WORKSPACE_DIR:$WORKSPACE_DIR" -w "$WORKSPACE_DIR" \
    -e IMAGE -e EXPECTED_IMAGE_ID -e MODEL_PATH -e WORKSPACE_DIR \
    -e TP -e DP -e EP -e HIP_VISIBLE_DEVICES -e HSA_NO_SCRATCH_RECLAIM \
    "$IMAGE" sleep infinity
echo "Ready: docker exec $NAME bash /quick_test_prefill/bench.sh"
