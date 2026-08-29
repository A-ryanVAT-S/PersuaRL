"""Train the PersuaRL Selector with GRPO -- the main experiment.

    python -m persuarl.cli.train_selector --config configs/rl/persuarl.yaml

Common variations, no code changes required:

    # different Selector / Generator backbones
    --set selector.id=meta-llama/Llama-3.2-3B-Instruct
    --set generator.id=microsoft/Phi-3-mini-128k-instruct

    # freeze the Generator (ablation D.3.4)
    --set train.freeze_generator=True

    # reward ablation (Table 12): drop R1 by zeroing its weight
    --set rewards.weights.engagement=0.0
"""

from __future__ import annotations

import json

from ..training.grpo_selector import train_selector
from ..utils.logging import get_logger
from ._common import base_parser, bootstrap

LOGGER = get_logger(__name__)


def main() -> None:
    parser = base_parser(__doc__ or "train the PersuaRL selector with GRPO")
    args = parser.parse_args()
    config = bootstrap(args)
    summary = train_selector(config)
    LOGGER.info("PersuaRL training complete: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
