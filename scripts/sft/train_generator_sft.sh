#!/usr/bin/env bash
# Stage 3 -- expert-conditioned Generator SFT. The warm start PersuaRL needs.
#
#   bash scripts/sft/train_generator_sft.sh
#   bash scripts/sft/train_generator_sft.sh microsoft/Phi-3-mini-128k-instruct
#   bash scripts/sft/train_generator_sft.sh --set data.route_source=random
#
# ~3-4 h on an A100 80GB.
#
# DO NOT SKIP THIS. An un-adapted Generator cannot exploit the <expert> blocks,
# so every route produces a comparably mediocre response, every rollout in a
# GRPO group scores alike, and the Selector gets no gradient (Table 15).

source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

EXTRA=()
if [[ $# -gt 0 && "$1" != -* ]]; then
  EXTRA+=(--set "model.id=$1")
  log "generator backbone: $1"
  shift
fi

[[ -f "${DATA_ROOT}/insuredial/dialogues.csv" ]] || die "InsureDial not found under ${DATA_ROOT}"

log "generator SFT (expert-conditioned)"
gpu_summary
run_module persuarl.cli.train_sft \
  --config "${REPO_ROOT}/configs/sft/generator.yaml" "${EXTRA[@]}" "$@"

log "generator adapter -> ${MODELS_ROOT}/generator_sft"
log "next: bash scripts/rl/train_persuarl.sh"
