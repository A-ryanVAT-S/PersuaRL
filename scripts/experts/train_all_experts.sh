#!/usr/bin/env bash
# Stage 1 -- fine-tune all four expert modules, in sequence.
#
#   bash scripts/experts/train_all_experts.sh
#   bash scripts/experts/train_all_experts.sh Qwen/Qwen2.5-3B-Instruct
#
# ~3 h total on an A100 80GB. Sequential, not parallel: four 3B models at once
# will not fit alongside anything else on a single card.
#
# OPTIONAL. The repo ships precomputed expert answers for all 13,383 turns, so
# you can skip to Stage 2 and use those. Train these if you want a different
# backbone, a different label space, or to annotate a new corpus.

source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

MODEL_ARG=()
if [[ $# -gt 0 && "$1" != -* ]]; then
  MODEL_ARG+=("$1")
  shift
fi

for expert in engagement intent keyterm sentiment; do
  log "=== expert ${expert} ==="
  bash "${REPO_ROOT}/scripts/experts/train_expert.sh" "${expert}" "${MODEL_ARG[@]}" "$@"
done

log "all four experts trained -> ${MODELS_ROOT}/experts/"
log "next: bash scripts/experts/run_expert_inference.sh   (to regenerate cached answers)"
