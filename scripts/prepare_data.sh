#!/usr/bin/env bash
# Stage 0 -- split the shipped expert answers back into (utterance, label, reason).
#
#   bash scripts/prepare_data.sh
#
# Fast, CPU-only. Writes data/processed/, which is gitignored and regenerable.

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

[[ -f "${DATA_ROOT}/insuredial/dialogues.csv" ]] || \
  die "dialogues.csv not found under ${DATA_ROOT}/insuredial -- see data/README.md"

log "preparing derived training files from ${DATA_ROOT}/insuredial"
run_module persuarl.cli.prepare_data --config "${REPO_ROOT}/configs/data.yaml" "$@"

log "done. Derived files:"
find "${DATA_ROOT}/processed" -name '*.csv' -exec ls -lh {} \; 2>/dev/null | awk '{print "  " $5, $9}'
