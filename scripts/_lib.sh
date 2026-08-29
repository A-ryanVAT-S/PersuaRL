#!/usr/bin/env bash
# Shared helpers for every script in scripts/. Sourced, never executed.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
#
# Provides: REPO_ROOT, DATA_ROOT, MODELS_ROOT, PYTHON, log/warn/die,
# run_module, require_dir, gpu_summary.

set -euo pipefail

# --------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------

# Resolve the repo root from this file, so scripts work from any cwd.
_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_LIB_DIR}/.." && pwd)"
export REPO_ROOT

# Load .env if present. Existing environment variables win, matching the
# behaviour of persuarl.utils.env.load_dotenv on the Python side.
if [[ -f "${REPO_ROOT}/.env" ]]; then
  while IFS='=' read -r _key _value; do
    [[ -z "${_key}" || "${_key}" == \#* ]] && continue
    _key="$(echo "${_key}" | xargs)"
    _value="$(echo "${_value}" | xargs | sed -e 's/^"//' -e "s/^'//" -e 's/"$//' -e "s/'$//")"
    [[ -z "${!_key:-}" ]] && export "${_key}=${_value}"
  done < "${REPO_ROOT}/.env"
fi

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
MODELS_ROOT="${MODELS_ROOT:-${REPO_ROOT}/outputs}"
PYTHON="${PYTHON:-python}"
export DATA_ROOT MODELS_ROOT PYTHON

# Make `src/` importable without requiring `pip install -e .` first.
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Tokenizers fork warnings drown out the training log otherwise.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

if [[ -t 1 ]]; then
  _BLUE=$'\033[0;34m'; _YELLOW=$'\033[0;33m'; _RED=$'\033[0;31m'; _RESET=$'\033[0m'
else
  _BLUE=""; _YELLOW=""; _RED=""; _RESET=""
fi

log()  { printf '%s[persuarl]%s %s\n' "${_BLUE}"   "${_RESET}" "$*"; }
warn() { printf '%s[warn]%s %s\n'     "${_YELLOW}" "${_RESET}" "$*" >&2; }
die()  { printf '%s[error]%s %s\n'    "${_RED}"    "${_RESET}" "$*" >&2; exit 1; }

# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

require_dir() {
  # require_dir <path> <message>
  # A missing prerequisite should stop the run here, not 40 minutes in.
  [[ -d "$1" ]] || die "$2"
}

warn_missing_dir() {
  # warn_missing_dir <path> <message>
  # For prerequisites that degrade the run rather than break it.
  [[ -d "$1" ]] || warn "$2"
}

gpu_summary() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  GPU /'
  else
    warn "nvidia-smi not found -- training on CPU will be impractically slow"
  fi
}

# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------

run_module() {
  # run_module <module> [args...]
  # Echo the command before running it, so a log file records exactly what ran.
  local module="$1"; shift
  log "python -m ${module} $*"
  "${PYTHON}" -m "${module}" "$@"
}

timestamp() { date +%Y%m%d_%H%M%S; }
