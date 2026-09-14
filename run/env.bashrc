# Shared defaults; override with VAR=value before invoking a script.
RUN_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export EXP="${EXP:-default}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd "$RUN_DIR/.." && pwd)/workspace/$EXP}"
export MODEL_PATH="${MODEL_PATH:-/shared_nfs/models/GLM-5.2-MXFP4}"
export SGLANG_DIR="${SGLANG_DIR:-/sglang}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3}"
export TP="${TP:-4}" EP="${EP:-1}" DP="${DP:-4}"
export PORT="${PORT:-31832}" CONC="${CONC:-32}"
export MEM_FRACTION="${MEM_FRACTION:-0.85}" KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
export SPEC_STEPS="${SPEC_STEPS:-5}" SPEC_DRAFT="${SPEC_DRAFT:-6}" SPEC_TOPK="${SPEC_TOPK:-1}"
export ACC_LEN="${ACC_LEN:-3.61}"

export ISL="${ISL:-10000}" OSL="${OSL:-500}"
export NUM_PROMPTS="${NUM_PROMPTS:-128}" WARMUP_REQUESTS="${WARMUP_REQUESTS:-16}"
export PROFILE_STEPS="${PROFILE_STEPS:-12}"
export RESULTS_DIR="${RESULTS_DIR:-$WORKSPACE_DIR/run_profiles}"

# Keep intermediate files with the experiment; use separate workspaces per node.
# ROCm's runtime temporary files must be on a local filesystem, not NFS.
export TMPDIR="${TMPDIR:-/tmp}"
export AITER_JIT_DIR="${AITER_JIT_DIR:-$WORKSPACE_DIR/cache/aiter}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$WORKSPACE_DIR/cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$WORKSPACE_DIR/cache/torch}"
export HF_HOME="${HF_HOME:-$WORKSPACE_DIR/cache/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORKSPACE_DIR/cache/xdg}"
export PYTHONPATH="$SGLANG_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export SGLANG_OPT_USE_TOPK_V2=false SGLANG_TIMEOUT_KEEP_ALIVE=900
export AITER_USE_FLYDSL_MOE_SORTING=1 SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1
export SGLANG_SIMULATE_ACC_LEN="$ACC_LEN"
export SGLANG_SIMULATE_ACC_METHOD=match-expected SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token

# ROCm 7.2 needs literal false to expose graph kernels; graphs remain enabled.
export DEBUG_CLR_GRAPH_PACKET_CAPTURE="${DEBUG_CLR_GRAPH_PACKET_CAPTURE:-false}"
export SGLANG_PROFILE_V2=0 SGLANG_PROFILE_WITH_STACK=false SGLANG_PROFILE_RECORD_SHAPES=false
