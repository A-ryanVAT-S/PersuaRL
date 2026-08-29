#!/usr/bin/env bash
# One-time environment setup: dependencies, NLTK data, .env, sanity checks.
#
#   bash scripts/setup_env.sh
#   bash scripts/setup_env.sh --cuda cu118      # match your driver
#   bash scripts/setup_env.sh --no-torch        # torch already installed

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

CUDA_TAG="cu121"
INSTALL_TORCH=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda)     CUDA_TAG="$2"; shift 2 ;;
    --no-torch) INSTALL_TORCH=0; shift ;;
    *)          die "unknown flag: $1" ;;
  esac
done

log "python: $(${PYTHON} --version 2>&1)"

if [[ "${INSTALL_TORCH}" -eq 1 ]]; then
  # torch must come first and from the CUDA-specific index; pip cannot infer
  # the right wheel from requirements.txt and will give you a CPU build.
  log "installing torch (${CUDA_TAG})"
  "${PYTHON}" -m pip install --upgrade pip
  "${PYTHON}" -m pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
fi

log "installing PersuaRL and its dependencies"
"${PYTHON}" -m pip install -e "${REPO_ROOT}[quant,logging,dev]"

log "downloading NLTK tokenizer data"
# punkt_tab is the newer name; try both so either NLTK version works.
"${PYTHON}" - <<'PY'
import nltk
for resource in ("punkt", "punkt_tab"):
    try:
        nltk.download(resource, quiet=True)
    except Exception as error:
        print(f"  note: could not fetch {resource} ({error}); R4 will fall back "
              f"to whitespace tokenisation")
PY

if [[ ! -f "${REPO_ROOT}/.env" ]]; then
  cp "${REPO_ROOT}/.env.example" "${REPO_ROOT}/.env"
  log "created .env from .env.example -- add your HF_TOKEN if you need gated models"
fi

log "verifying the install"
"${PYTHON}" - <<'PY'
import importlib

for module in ("torch", "transformers", "trl", "peft", "datasets", "bert_score", "persuarl"):
    try:
        loaded = importlib.import_module(module)
        print(f"  ok   {module:<14} {getattr(loaded, '__version__', '')}")
    except ImportError as error:
        print(f"  FAIL {module:<14} {error}")

import torch
print(f"  cuda available: {torch.cuda.is_available()}")
PY

gpu_summary
log "setup complete. Next: bash scripts/prepare_data.sh"
