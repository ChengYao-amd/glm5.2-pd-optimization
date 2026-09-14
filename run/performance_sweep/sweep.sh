#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SWEEP_RESULTS_DIR="${RESULTS_DIR:-}"
source "$HERE/../env.bashrc"
export RESULTS_DIR="${SWEEP_RESULTS_DIR:-$WORKSPACE_DIR/performance_sweep}"
exec python3 "$HERE/sweep.py" "$@"
