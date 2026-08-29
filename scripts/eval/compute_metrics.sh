#!/usr/bin/env bash
# Stage 6 -- automatic metrics for Table 2.
#
#   bash scripts/eval/compute_metrics.sh results/persuarl.csv
#   bash scripts/eval/compute_metrics.sh results/persuarl.csv --with-judge
#   bash scripts/eval/compute_metrics.sh results/persuarl.csv --with-ppl meta-llama/Llama-3.2-3B-Instruct
#
# BLEU-2, METEOR, BERTScore-F1, Distinct-2 and ROUGE-1 always. Perplexity and
# LLM-as-a-judge opt in, because each loads an extra model.

source "$(dirname "${BASH_SOURCE[0]}")/../_lib.sh"

INPUT="${1:-results/persuarl.csv}"
shift || true
[[ -f "${INPUT}" ]] || die "results file not found: ${INPUT}"

BASENAME="$(basename "${INPUT}" .csv)"
OUTPUT="results/metrics/${BASENAME}_metrics.csv"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-judge)
      EXTRA+=(--set "judge_model_id=prometheus-eval/prometheus-7b-v2.0"); shift ;;
    --with-ppl)
      [[ -n "${2:-}" ]] || die "--with-ppl needs a model id"
      EXTRA+=(--set "perplexity_model_id=$2"); shift 2 ;;
    --out) OUTPUT="$2"; shift 2 ;;
    *)     EXTRA+=("$1"); shift ;;
  esac
done

mkdir -p "$(dirname "${REPO_ROOT}/${OUTPUT}")"

log "scoring ${INPUT}"
run_module persuarl.cli.evaluate \
  --config "${REPO_ROOT}/configs/eval/default.yaml" \
  --set "input_path=${INPUT}" \
  --set "output_path=${OUTPUT}" \
  "${EXTRA[@]}"

log "per-turn scores -> ${OUTPUT}"
log "summary         -> ${OUTPUT%.csv}.summary.json"
