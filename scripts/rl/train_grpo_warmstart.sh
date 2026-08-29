#!/usr/bin/env bash
# Warm-start GRPO -- the "Single -> SFT -> RL" progression of Table 2.
# Merges an SFT adapter into the base weights, then trains a fresh LoRA with GRPO.
#
#   bash scripts/rl/train_grpo_warmstart.sh
#   bash scripts/rl/train_grpo_warmstart.sh meta-llama/Llama-3.2-3B-Instruct outputs/sft_baseline
#
# Merging (rather than stacking a second adapter on a live one) is deliberate:
# with two active adapters the RL gradient works against the SFT adapter as well
# as the base model, and the effective learning rate is not what you configured.

source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

EXTRA=()
if [[ $# -gt 0 && "$1" != -* ]]; then
  slug="$(echo "$1" | tr '/' '_' | tr '[:upper:]' '[:lower:]')"
  EXTRA+=(--set "model.id=$1" --set "output_dir=${MODELS_ROOT}/grpo_warmstart_${slug}")
  log "policy: $1"
  shift
fi

ADAPTER="${MODELS_ROOT}/sft_baseline"
if [[ $# -gt 0 && "$1" != -* ]]; then
  ADAPTER="$1"; shift
fi
EXTRA+=(--set "model.adapter_path=${ADAPTER}")

require_dir "${ADAPTER}" \
  "SFT adapter not found at ${ADAPTER} -- run: bash scripts/sft/train_baseline_sft.sh"

log "warm-start GRPO from ${ADAPTER}"
gpu_summary
run_module persuarl.cli.train_single_grpo \
  --config "${REPO_ROOT}/configs/rl/single_grpo_warmstart.yaml" "${EXTRA[@]}" "$@"
