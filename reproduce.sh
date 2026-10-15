#!/usr/bin/env bash
# Reproduce the paper's tables and figures from the checkpoint pool.
#
#   bash reproduce.sh                   reports from existing results (no GPU, minutes)
#   bash reproduce.sh --from eval       re-run the evals, then the reports
#   bash reproduce.sh --from ood        re-fit the OOD detectors, then evals + reports
#   bash reproduce.sh --from selection  re-score the pool and re-select, then all of the above
#
# Stages: pool -> selection -> ood -> eval -> reports. Starting from a stage
# recomputes it and everything after it; earlier stages are read from
# evaluations/results/. Replaced outputs are moved to
# archive/cross_scenario/superseded/ (never deleted).
#
# Options: --workers N (eval processes per cell, default 4), --cells a,b,...,
#          --no-figures (skip the evals' diagnostic plots; CSVs unaffected)
# Retraining the pool: see evaluations/paper/cross_scenario/run.py.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-$REPO/.venv/bin/python}"
cd "$REPO"

FROM=reports
WORKERS=4
CELLS=()
NOFIG=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) FROM="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --cells) CELLS=(--cells "$2"); shift 2 ;;
        --no-figures) NOFIG=(--no-figures); shift ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
case "$FROM" in
    selection|ood|eval|reports) ;;
    *) echo "--from must be one of: selection ood eval reports" >&2; exit 2 ;;
esac
[[ -x "$PY" ]] || { echo "python not found at $PY (run: uv sync)" >&2; exit 1; }

stage() { printf '\n=== %s ===\n' "$1"; }
SECONDS=0

if [[ "$FROM" == selection ]]; then
    stage "selection: score the pool, select cells"
    "$PY" -m evaluations.paper.cross_scenario.selection
fi

if [[ "$FROM" == selection || "$FROM" == ood ]]; then
    stage "ood + evals"
    "$PY" -m evaluations.paper.cross_scenario.run --force --workers "$WORKERS" "${CELLS[@]}" "${NOFIG[@]}" \
        --steps ood,eval,eval_finetune,eval_leader_only
    stage "alpha sweep evals"
    "$PY" -m evaluations.paper.alpha_sweep.run --force --workers "$WORKERS" "${NOFIG[@]}"
elif [[ "$FROM" == eval ]]; then
    stage "evals"
    "$PY" -m evaluations.paper.cross_scenario.run --force --workers "$WORKERS" "${CELLS[@]}" "${NOFIG[@]}" \
        --steps eval,eval_finetune,eval_leader_only
    stage "alpha sweep evals"
    "$PY" -m evaluations.paper.alpha_sweep.run --force --workers "$WORKERS" "${NOFIG[@]}"
fi

stage "reports"
"$PY" -m evaluations.paper.cross_scenario.analyze
"$PY" -m evaluations.paper.alpha_sweep.analyze
"$PY" -m evaluations.paper.appendix.generate_hyperparam_tables
"$PY" -m evaluations.paper.appendix.generate_env_tables
"$PY" -m evaluations.paper.figures.env_overviews
"$PY" -m evaluations.paper.figures.warehouse_along_wall

printf '\nreproduce.sh --from %s finished in %dm%02ds\n' "$FROM" $((SECONDS / 60)) $((SECONDS % 60))
