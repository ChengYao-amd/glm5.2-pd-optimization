#!/bin/bash
# Start a container 
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
source "$REPO/run/env.bashrc"
IMAGE="${IMAGE:-rocm-llm-bench:kernelforge}"
NAME="${NAME:-dev-container}"
# Keep Hugging Face snapshot -> ../../blobs symlinks valid inside the container.
VOLUME_DIR="/shared_nfs"

docker stop -t 10 "$NAME" >/dev/null 2>&1 || true
docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" -w / \
    --network host --ipc=host --shm-size 32g \
    --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
    --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
    -v "$REPO/forge_workspace:/forge_workspace" \
    -v "$REPO/run:/run" \
    -v "$VOLUME_DIR:$VOLUME_DIR" \
    -e PYTHONNOUSERSITE=1 -e MODEL_PATH \
    ${HIP_VISIBLE_DEVICES:+-e HIP_VISIBLE_DEVICES} \
    "$IMAGE" \
    sleep infinity
    
