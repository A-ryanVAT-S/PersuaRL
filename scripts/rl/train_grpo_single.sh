#!/usr/bin/env bash
# Single-model GRPO baseline -- no Selector, no experts (Table 13, D.3.3).
# The policy generates the reply directly, scored by the same composite reward.
#
#   bash scripts/rl/train_grpo_single.sh
#   bash scripts/rl/train_grpo_single.sh meta-llama/Llama-3.2-3B-Instruct
#   bash scripts/rl/train_grpo_single.sh Qwen/Qwen2.5-3B-Instruct --no-judge

source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

EXTRA=()
if [[ $# -gt 0 && "$1" != -* ]]; then
  slug="$(echo "$1" | tr '/' '_' | tr '[:upper:]' '[:lower:]')"
  EXTRA+=(--set "model.id=$1" --set "output_dir=${MODELS_ROOT}/grpo_single_${slug}")
  log "policy: $1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-judge) EXTRA+=(--set "rewards.judge.model_id=None"); shift ;;
    *)          EXTRA+=("$1"); shift ;;
  esac
done

log "single-model GRPO (cold start from the instruct checkpoint)"
gpu_summary
run_module persuarl.cli.train_single_grpo \
  --config "${REPO_ROOT}/configs/rl/single_grpo.yaml" "${EXTRA[@]}"
