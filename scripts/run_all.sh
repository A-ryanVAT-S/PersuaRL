#!/usr/bin/env bash
# The whole pipeline, end to end. Roughly 30-35 h on one A100 80GB.
#
#   bash scripts/run_all.sh
#   bash scripts/run_all.sh --skip-experts          # use the shipped cached outputs
#   bash scripts/run_all.sh --selector Qwen/Qwen2.5-3B-Instruct \
#                           --generator microsoft/Phi-3-mini-128k-instruct
#
# Each stage is also runnable on its own -- see the individual scripts. This
# exists so a fresh clone has one command that produces the paper's numbers.

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

SKIP_EXPERTS=0
SELECTOR=""
GENERATOR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-experts) SKIP_EXPERTS=1; shift ;;
    --selector)     SELECTOR="$2"; shift 2 ;;
    --generator)    GENERATOR="$2"; shift 2 ;;
    *)              die "unknown flag: $1" ;;
  esac
done

log "STAGE 0/6  data preparation"
bash "${REPO_ROOT}/scripts/prepare_data.sh"

if [[ "${SKIP_EXPERTS}" -eq 0 ]]; then
  log "STAGE 1/6  expert modules"
  bash "${REPO_ROOT}/scripts/experts/train_all_experts.sh"
  bash "${REPO_ROOT}/scripts/experts/run_expert_inference.sh"
else
  log "STAGE 1/6  skipped -- using the cached expert outputs in ${DATA_ROOT}"
fi

log "STAGE 2/6  reward classifiers and prototypes"
bash "${REPO_ROOT}/scripts/rewards/train_reward_models.sh"

log "STAGE 3/6  generator SFT (warm start)"
bash "${REPO_ROOT}/scripts/sft/train_generator_sft.sh" ${GENERATOR:+"${GENERATOR}"}

log "STAGE 4/6  PersuaRL selector GRPO"
PERSUARL_ARGS=()
[[ -n "${SELECTOR}"  ]] && PERSUARL_ARGS+=(--selector "${SELECTOR}")
[[ -n "${GENERATOR}" ]] && PERSUARL_ARGS+=(--generator "${GENERATOR}")
bash "${REPO_ROOT}/scripts/rl/train_persuarl.sh" "${PERSUARL_ARGS[@]}"

log "STAGE 5/6  inference on the held-out test split"
bash "${REPO_ROOT}/scripts/inference/run_pipeline.sh" --out results/persuarl.csv

log "STAGE 6/6  automatic metrics"
bash "${REPO_ROOT}/scripts/eval/compute_metrics.sh" results/persuarl.csv

log "pipeline complete. Summary: results/metrics/persuarl_metrics.summary.json"
