#!/bin/bash
# Start a container and apply the shared SGLang patches.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
source "$REPO/run/env.bashrc"
IMAGE="${IMAGE:-rocm-llm-bench:kernelforge}"
NAME="${NAME:-dev-container}"
# Keep Hugging Face snapshot -> ../../blobs symlinks valid inside the container.
VOLUME_DIR="/shared_nfs"
LOCAL_TMP_DIR="${LOCAL_TMP_DIR:-/tmp/glm52-${USER:-user}/$EXP/$NAME}"
mkdir -p "$WORKSPACE_DIR/forge_workspace" "$LOCAL_TMP_DIR"
# ROCm crashes on NFS-backed TMPDIR; keep runtime scratch on this node's disk.
# Expose it through the experiment directory and a short container path for IPC.
ln -sfn "$LOCAL_TMP_DIR" "$WORKSPACE_DIR/runtime-tmp"

# Optional agent configuration. Credentials stay in mounted files, not Docker env.
AGENT_MOUNTS=()
if [ -n "${LLM_GATEWAY_DIR:-}" ]; then
    AGENT_MOUNTS+=(--mount "type=bind,src=$LLM_GATEWAY_DIR,dst=/llm_gateway,readonly")
    AGENT_MOUNTS+=(--mount "type=bind,src=$LLM_GATEWAY_DIR/config.toml,dst=/root/.codex/config.toml,readonly")
fi

docker stop -t 10 "$NAME" >/dev/null 2>&1 || true
docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --init --name "$NAME" -w "$WORKSPACE_DIR" \
    --network host --ipc=host --shm-size 32g \
    --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
    --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
    -v "$WORKSPACE_DIR/forge_workspace:/forge_workspace" \
    -v "$REPO/run:/run" \
    -v "$VOLUME_DIR:$VOLUME_DIR" \
    -v "$WORKSPACE_DIR:$WORKSPACE_DIR" \
    -v "$LOCAL_TMP_DIR:/tmp" \
    -v "$LOCAL_TMP_DIR:$LOCAL_TMP_DIR" \
    "${AGENT_MOUNTS[@]}" \
    -e PYTHONNOUSERSITE=1 -e PYTHONDONTWRITEBYTECODE=1 \
    -e MODEL_PATH -e SGLANG_DIR -e EXP -e WORKSPACE_DIR -e TMPDIR=/tmp \
    ${HIP_VISIBLE_DEVICES:+-e HIP_VISIBLE_DEVICES} \
    "$IMAGE" \
    sleep infinity

# Patch the container's editable SGLang installation before either workflow starts.
# fake_dsa_seed depends on the get_disagg import in fake_decode; keep this order.
docker exec -i "$NAME" bash -s -- "$SGLANG_DIR" <<'PATCH_CONTAINER' 2>&1 | tee "$WORKSPACE_DIR/patches.log"
set -euo pipefail
sglang_dir=$1
for patch_name in fake_decode fake_dsa_seed dp_sparse_graph client_details; do
    patch_file="/run/patches/$patch_name.patch"
    if git -C "$sglang_dir" apply --reverse --check "$patch_file" 2>/dev/null; then
        echo "Already applied: $patch_name.patch"
    else
        echo "Applying: $patch_name.patch"
        git -C "$sglang_dir" apply --check "$patch_file"
        git -C "$sglang_dir" apply "$patch_file"
    fi
done
echo "All SGLang patches are ready."
PATCH_CONTAINER

echo "Container $NAME is ready; patch log: $WORKSPACE_DIR/patches.log"
