#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../env.bashrc"
export OUT_DIR="${OUT_DIR:-$RESULTS_DIR/profile-$(date -u +%Y%m%dT%H%M%S-%N)}"
if [ -e "$OUT_DIR" ]; then
    echo "Choose a fresh OUT_DIR: $OUT_DIR" >&2; exit 1
fi
bash "$HERE/bench.sh" --profile --profile-num-steps "$PROFILE_STEPS" \
    --profile-activities CPU GPU --profile-output-dir "$OUT_DIR/traces" "$@"

# The native client can exit successfully even when the profiler request fails.
compgen -G "$OUT_DIR/traces/*/*.trace.json.gz" >/dev/null || {
    echo "No profile traces found in $OUT_DIR/traces" >&2; exit 1;
}
echo "Profile saved to $OUT_DIR"
