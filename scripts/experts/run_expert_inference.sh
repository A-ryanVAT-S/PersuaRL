#!/usr/bin/env bash
# Stage 1b -- annotate a corpus with the trained experts.
#
#   bash scripts/experts/run_expert_inference.sh                 # all four
#   bash scripts/experts/run_expert_inference.sh --expert intent # just one
#   bash scripts/experts/run_expert_inference.sh --set data.dialogues_path=data/deal/dialogues.csv
#
# Overwrites data/insuredial/expert_outputs/*.csv. Back them up first if you
# want to keep the shipped answers.

source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

warn_missing_dir "${MODELS_ROOT}/experts" \
  "no trained experts under ${MODELS_ROOT}/experts -- run train_all_experts.sh first"

log "annotating with the expert modules (one loaded at a time)"
gpu_summary
run_module persuarl.cli.run_experts --config "${REPO_ROOT}/configs/experts/inference.yaml" "$@"
