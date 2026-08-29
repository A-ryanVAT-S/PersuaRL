"""Supervised fine-tuning: the Generator, or a single-model SFT baseline.

    # Generator warm start (expert-conditioned)
    python -m persuarl.cli.train_sft --config configs/sft/generator.yaml

    # SFT baseline for Table 2 (no experts)
    python -m persuarl.cli.train_sft --config configs/sft/baseline.yaml

    # Same config, different backbone
    python -m persuarl.cli.train_sft --config configs/sft/baseline.yaml \
        --set model.id=Qwen/Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import json

from ..training.sft import train_sft
from ..utils.logging import get_logger
from ._common import base_parser, bootstrap

LOGGER = get_logger(__name__)


def main() -> None:
    parser = base_parser(__doc__ or "run supervised fine-tuning")
    args = parser.parse_args()
    config = bootstrap(args)
    metrics = train_sft(config)
    LOGGER.info("SFT complete: %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
