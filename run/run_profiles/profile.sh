#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../env.bashrc"
export OUT_DIR="${OUT_DIR:-$RESULTS_DIR/profile-$(date -u +%Y%m%dT%H%M%S-%N)}"
exec bash "$HERE/bench.sh" --profile --profile-num-steps "$PROFILE_STEPS" \
    --profile-activities CPU GPU --profile-output-dir "$OUT_DIR/traces" "$@"
