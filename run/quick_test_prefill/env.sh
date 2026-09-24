#!/usr/bin/env bash
# Compute settings from yaocheng/2p1d-sweep; no dependency on that directory.
PREFILL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export IMAGE="${IMAGE:-infera-sglang:v0519-yihou-0917-nextnfix-hicache}"
export EXPECTED_IMAGE_ID="${EXPECTED_IMAGE_ID:-sha256:fd7220a57b7d3b58efd875c41f7a9ef46b93469581102d96cbeb6f5451e91d35}"
export MODEL_PATH="${MODEL_PATH:-/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4}"
# /shared_nfs is mounted read-only on some GPU nodes, including 137.
export WORKSPACE_DIR="${WORKSPACE_DIR:-/tmp/glm52-quick-prefill-${USER:-user}}"
export TP="${TP:-8}" DP="${DP:-8}" EP="${EP:-1}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-0}"
export NCCL_IB_DISABLE=1 NCCL_IGNORE_CPU_AFFINITY=1
export SGLANG_USE_AITER=1 SGLANG_OPT_USE_TOPK_V2=false
export SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1 AITER_USE_FLYDSL_MOE_SORTING=1
export SAFETENSORS_FAST_GPU=1 HIP_FORCE_DEV_KERNARG=1 PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export SGLANG_DP_USE_GATHERV=$((DP > 1))
export TMPDIR=/tmp
export AITER_JIT_DIR="${AITER_JIT_DIR:-/aiter-jit}"
