#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export RESULTS_DIR="${RESULTS_DIR:-$HERE/results}"
source "$HERE/../env.bashrc"
exec python3 "$HERE/sweep.py" "$@"
