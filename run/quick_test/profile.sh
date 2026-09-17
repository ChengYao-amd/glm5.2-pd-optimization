#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
QUICK_RESULTS_DIR="${RESULTS_DIR:-}"
source "$HERE/../env.bashrc"
export RESULTS_DIR="${QUICK_RESULTS_DIR:-$WORKSPACE_DIR/quick_test}"
export OUT_DIR="${OUT_DIR:-$RESULTS_DIR/profile-$(date -u +%Y%m%dT%H%M%S-%N)}"
export MAX_STEPS="${MAX_STEPS:-$((${PROFILE_START:-20} + PROFILE_STEPS))}"
exec bash "$HERE/bench.sh" --profile \
    --profile-start-step "${PROFILE_START:-20}" --profile-num-steps "$PROFILE_STEPS" \
    --profile-ranks "${PROFILE_RANKS:-0}" "$@"
