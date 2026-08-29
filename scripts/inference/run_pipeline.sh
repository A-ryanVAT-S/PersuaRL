#!/usr/bin/env bash
# Stage 5 -- Selector -> Experts -> Generator over the held-out test split.
#
#   bash scripts/inference/run_pipeline.sh
#   bash scripts/inference/run_pipeline.sh --out results/persuarl_llama.csv
#   bash scripts/inference/run_pipeline.sh --mode all           # AllExpert
#   bash scripts/inference/run_pipeline.sh --mode prompting     # prompted, no RL
#   bash scripts/inference/run_pipeline.sh --live-experts       # call expert LMs
#   bash scripts/inference/run_pipeline.sh --limit 20           # smoke run
#
# History is built from the model's own replies, not the reference transcript,
# so errors compound across a dialogue -- which is what you want to measure.

source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

OUT="results/persuarl.csv"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)   OUT="$2"; shift 2 ;;
    --mode)
      case "$2" in
        persuarl|all|prompting) ;;
        *) die "--mode expects persuarl, all or prompting" ;;
      esac
      EXTRA+=(--set "routing_mode=$2"); shift 2 ;;
    --live-experts) EXTRA+=(--set "expert_source=live"); shift ;;
    --limit)        EXTRA+=(--set "max_conversations=$2"); shift 2 ;;
    --verbose)      EXTRA+=(--set "verbose=True"); shift ;;
    *)              EXTRA+=("$1"); shift ;;
  esac
done

warn_missing_dir "${MODELS_ROOT}/persuarl/selector" \
  "no trained selector at ${MODELS_ROOT}/persuarl/selector -- the run will use the
     base backbone, which is the untrained-model ablation (Table 17), not PersuaRL"

mkdir -p "$(dirname "${REPO_ROOT}/${OUT}")"

log "running inference -> ${OUT}"
gpu_summary
run_module persuarl.cli.run_pipeline \
  --config "${REPO_ROOT}/configs/inference/persuarl.yaml" \
  --set "output_path=${OUT}" \
  "${EXTRA[@]}"

log "next: bash scripts/eval/compute_metrics.sh ${OUT}"
