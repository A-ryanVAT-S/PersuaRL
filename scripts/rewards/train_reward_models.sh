#!/usr/bin/env bash
# Stage 2 -- train the R1/R2 classifiers and build their class prototypes.
#
#   bash scripts/rewards/train_reward_models.sh
#   BASE_MODEL=answerdotai/ModernBERT-base bash scripts/rewards/train_reward_models.sh
#   bash scripts/rewards/train_reward_models.sh --dimension intent
#
# ~15 min on an A100. Each dimension produces a classifier directory containing
# both the model and its prototypes.pt -- they are built in one command, from
# one checkpoint, on purpose: prototypes scored against a different classifier
# than they were built from produce plausible-looking garbage.

source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

EXTRA=()
[[ -n "${BASE_MODEL:-}" ]] && { EXTRA+=(--set "base_model_id=${BASE_MODEL}"); log "classifier backbone: ${BASE_MODEL}"; }

for dimension in engagement intent; do
  file="${DATA_ROOT}/processed/classifiers/${dimension}.csv"
  [[ -f "${file}" ]] || die "${file} not found -- run: bash scripts/prepare_data.sh"
  rows=$(( $(wc -l < "${file}") - 1 ))
  if [[ "${rows}" -lt 100 ]]; then
    warn "${dimension} classifier data has only ${rows} rows; R${dimension} will be weak."
    warn "See data/README.md#known-issues-in-the-shipped-files"
  fi
done

log "training reward classifiers + prototypes"
gpu_summary
run_module persuarl.cli.train_reward_models \
  --config "${REPO_ROOT}/configs/rewards/classifiers.yaml" "${EXTRA[@]}" "$@"

log "reward models written to ${MODELS_ROOT}/reward_models/"
