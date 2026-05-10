#!/usr/bin/env bash
# Launcher for an overnight hyperparameter sweep.
#
# Writes everything to logs/search/<timestamp>/:
#   trials.jsonl   per-trial records
#   logs/*.log     captured stdout/stderr for each trial subprocess
#   report.md      end-of-run markdown summary
#
# Tail the live log with:
#     tail -F logs/search/<timestamp>/orchestrator.log
set -euo pipefail

SPACE="${SPACE:-configs/search/overnight_v1.py}"
BUDGET="${BUDGET:-8h}"
STRATEGY="${STRATEGY:-random}"
MAX_TRIALS="${MAX_TRIALS:-}"
TRIAL_TIMEOUT="${TRIAL_TIMEOUT:-}"
SEED="${SEED:-0}"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="logs/search/${STAMP}"
mkdir -p "${OUT}"

ORCH_LOG="${OUT}/orchestrator.log"
echo "Search run:  ${OUT}"
echo "Orchestrator log:  ${ORCH_LOG}"

EXTRA_ARGS=()
[[ -n "${MAX_TRIALS}" ]] && EXTRA_ARGS+=(--max-trials "${MAX_TRIALS}")
[[ -n "${TRIAL_TIMEOUT}" ]] && EXTRA_ARGS+=(--trial-timeout "${TRIAL_TIMEOUT}")

exec python -m pixels_to_predictions.search run \
    --space "${SPACE}" \
    --out "${OUT}" \
    --strategy "${STRATEGY}" \
    --budget "${BUDGET}" \
    --seed "${SEED}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${ORCH_LOG}"
