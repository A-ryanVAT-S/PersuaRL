# Convenience targets. Everything here just calls a script in scripts/.
#
#   make setup      install deps + NLTK data
#   make data       build the derived training files
#   make experts    fine-tune all four expert modules
#   make rewards    train the R1/R2 classifiers + prototypes
#   make sft        expert-conditioned Generator SFT
#   make train      PersuaRL selector GRPO (the main experiment)
#   make infer      run the pipeline on the test split
#   make eval       automatic metrics
#   make all        every stage, in order
#   make test       unit tests (no GPU, no model downloads)
#   make lint       ruff

.PHONY: setup data experts rewards sft train infer eval all test lint clean help
.DEFAULT_GOAL := help

PYTHON ?= python
RESULTS ?= results/persuarl.csv

help:
	@grep -E '^#   \S' Makefile | sed 's/^#   //'

setup:
	bash scripts/setup_env.sh

data:
	bash scripts/prepare_data.sh

experts:
	bash scripts/experts/train_all_experts.sh

rewards:
	bash scripts/rewards/train_reward_models.sh

sft:
	bash scripts/sft/train_generator_sft.sh

train:
	bash scripts/rl/train_persuarl.sh

infer:
	bash scripts/inference/run_pipeline.sh --out $(RESULTS)

eval:
	bash scripts/eval/compute_metrics.sh $(RESULTS)

all:
	bash scripts/run_all.sh

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
