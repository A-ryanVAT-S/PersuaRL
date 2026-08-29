#!/usr/bin/env bash
# Stage 1 -- fine-tune ONE expert module.
#
#   bash scripts/experts/train_expert.sh intent
#   bash scripts/experts/train_expert.sh intent Qwen/Qwen2.5-3B-Instruct
#   bash scripts/experts/train_expert.sh keyterm --set train.epochs=5
#
# ~40 min per expert on an A100 80GB with the default 4-bit QLoRA settings.

source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

EXPERT="${1:-}"
[[ -n "${EXPERT}" ]] || die "usage: train_expert.sh <engagement|intent|keyterm|sentiment> [model_id] [--set k=v ...]"
shift

case "${EXPERT}" in
  engagement|intent|keyterm|sentiment) ;;
  *) die "unknown expert '${EXPERT}' (expected engagement, intent, keyterm or sentiment)" ;;
esac

# An unprefixed second argument is a model id; anything starting with - is a flag.
EXTRA=()
if [[ $# -gt 0 && "$1" != -* ]]; then
  EXTRA+=(--set "model.id=$1")
  log "backbone override: $1"
  shift
fi

DATASET="${DATA_ROOT}/processed/experts/${EXPERT}.csv"
[[ -f "${DATASET}" ]] || die "${DATASET} not found -- run: bash scripts/prepare_data.sh"

log "training the ${EXPERT} expert"
gpu_summary
run_module persuarl.cli.train_expert \
  --config "${REPO_ROOT}/configs/experts/${EXPERT}.yaml" \
  --expert "${EXPERT}" \
  "${EXTRA[@]}" "$@"

log "adapter written to ${MODELS_ROOT}/experts/${EXPERT}"
