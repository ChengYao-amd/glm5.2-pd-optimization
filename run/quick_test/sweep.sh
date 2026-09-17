#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
QUICK_RESULTS_DIR="${RESULTS_DIR:-}"
source "$HERE/../env.bashrc"
export RESULTS_DIR="${QUICK_RESULTS_DIR:-$WORKSPACE_DIR/quick_test}"
OUT_DIR="${OUT_DIR:-$RESULTS_DIR/sweep-$(date -u +%Y%m%dT%H%M%S-%N)}"
mkdir -p "$(dirname "$OUT_DIR")"
mkdir "$OUT_DIR" || { echo "Choose a fresh OUT_DIR: $OUT_DIR" >&2; exit 1; }

for concurrency in "${@:-$CONC}"; do
    CONC="$concurrency" OUT_DIR="$OUT_DIR/c$concurrency" bash "$HERE/bench.sh" \
        > "$OUT_DIR/c$concurrency.log" 2>&1
done
exec python3 "$HERE/collect.py" "$OUT_DIR" --output "$OUT_DIR/summary.csv"
