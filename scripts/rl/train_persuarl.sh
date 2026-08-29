#!/usr/bin/env bash
# Stage 4 -- PersuaRL. GRPO over the expert-selection policy, with a
# co-adapting Generator. This is the main experiment.
#
#   bash scripts/rl/train_persuarl.sh
#   bash scripts/rl/train_persuarl.sh --selector meta-llama/Llama-3.2-3B-Instruct
#   bash scripts/rl/train_persuarl.sh --generator Qwen/Qwen2.5-3B-Instruct
#   bash scripts/rl/train_persuarl.sh --freeze-generator      # ablation D.3.4
#   bash scripts/rl/train_persuarl.sh --ablate intent         # ablation Table 12
#   bash scripts/rl/train_persuarl.sh --no-judge              # dev; frees ~14 GB
#   bash scripts/rl/train_persuarl.sh --set train.num_generations=4
#
# ~25-28 h on an A100 80GB (3B selector + 3B generator + 7B judge resident).
#
# Watch, in the console or TensorBoard:
#   reward/total     should climb and plateau around 0.65-0.67
#   invalid_routes   should stay at ~0 (non-zero => constraint not installed)
#   route histogram  several routes recurring, not one dominating

source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

SELECTOR=""
GENERATOR=""
EXTRA=()
RUN_TAG="$(timestamp)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --selector)          SELECTOR="$2"; shift 2 ;;
    --generator)         GENERATOR="$2"; shift 2 ;;
    --freeze-generator)  EXTRA+=(--set "train.freeze_generator=True"); shift ;;
    --no-judge)          EXTRA+=(--set "rewards.judge.model_id=None"); shift ;;
    --ablate)
      # Zero one reward weight. Remaining weights are NOT renormalised,
      # matching the paper's ablation protocol.
      case "$2" in
        engagement|intent|contextual|repetition|judge) ;;
        *) die "--ablate expects one of: engagement, intent, contextual, repetition, judge" ;;
      esac
      EXTRA+=(--set "rewards.weights.$2=0.0")
      RUN_TAG="${RUN_TAG}_ablate-$2"
      shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

[[ -n "${SELECTOR}"  ]] && { EXTRA+=(--set "selector.id=${SELECTOR}");   log "selector:  ${SELECTOR}"; }
[[ -n "${GENERATOR}" ]] && { EXTRA+=(--set "generator.id=${GENERATOR}"); log "generator: ${GENERATOR}"; }

# --- prerequisite checks -------------------------------------------------
# Each of these degrades the run rather than breaking it, so warn loudly and
# continue -- but they are, in order, the two most common causes of "training
# runs but the reward never improves".

warn_missing_dir "${MODELS_ROOT}/generator_sft" \
  "no generator SFT adapter at ${MODELS_ROOT}/generator_sft.
     An un-adapted generator cannot use the expert blocks, so every route scores
     alike and the selector gets no gradient (Table 15).
     Run: bash scripts/sft/train_generator_sft.sh"

for dimension in engagement intent; do
  warn_missing_dir "${MODELS_ROOT}/reward_models/${dimension}_classifier" \
    "no ${dimension} reward classifier -- R$( [[ ${dimension} == engagement ]] && echo 1 || echo 2 ) will score 0.
     Run: bash scripts/rewards/train_reward_models.sh"
done

LOG_FILE="${REPO_ROOT}/logs/persuarl_${RUN_TAG}.log"
mkdir -p "$(dirname "${LOG_FILE}")"

log "starting PersuaRL selector GRPO (log: ${LOG_FILE})"
gpu_summary
run_module persuarl.cli.train_selector \
  --config "${REPO_ROOT}/configs/rl/persuarl.yaml" \
  --log-file "${LOG_FILE}" \
  "${EXTRA[@]}"

log "selector  -> ${MODELS_ROOT}/persuarl/selector"
log "generator -> ${MODELS_ROOT}/persuarl/generator"
log "routes    -> ${MODELS_ROOT}/persuarl/selector/best_routes.csv"
log "next: bash scripts/inference/run_pipeline.sh"
