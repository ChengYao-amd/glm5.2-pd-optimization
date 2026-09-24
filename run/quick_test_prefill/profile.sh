#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/bench.sh" --profile \
    --profile-steps "${PROFILE_STEPS:-1}" --profile-ranks "${PROFILE_RANKS:-0}" "$@"
