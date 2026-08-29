#!/usr/bin/env bash
# The SFT column of Table 2, and the warm start for single-model GRPO.
# No experts, no routing: (history, user utterance) -> reply.
#
#   bash scripts/sft/train_baseline_sft.sh
#   bash scripts/sft/train_baseline_sft.sh Qwen/Qwen2.5-3B-Instruct
#   bash scripts/sft/train_baseline_sft.sh mistralai/Mistral-Small-24B-Instruct-2501

source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

EXTRA=()
if [[ $# -gt 0 && "$1" != -* ]]; then
  EXTRA+=(--set "model.id=$1")
  # Keep each backbone's adapter in its own directory so runs do not clobber
  # each other when you sweep the Table 2 rows.
  slug="$(echo "$1" | tr '/' '_' | tr '[:upper:]' '[:lower:]')"
  EXTRA+=(--set "output_dir=${MODELS_ROOT}/sft_baseline_${slug}")
  log "baseline backbone: $1 -> ${MODELS_ROOT}/sft_baseline_${slug}"
  shift
fi

log "single-model SFT baseline"
gpu_summary
run_module persuarl.cli.train_sft \
  --config "${REPO_ROOT}/configs/sft/baseline.yaml" "${EXTRA[@]}" "$@"
