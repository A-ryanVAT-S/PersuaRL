"""Single-model GRPO baselines -- no Selector, no experts.

    # GRPO straight from the instruct checkpoint (Table 13)
    python -m persuarl.cli.train_single_grpo --config configs/rl/single_grpo.yaml

    # GRPO warm-started from an SFT adapter (the Single -> SFT -> RL column)
    python -m persuarl.cli.train_single_grpo --config configs/rl/single_grpo_warmstart.yaml
"""

from __future__ import annotations

import json

from ..training.grpo_single import train_single_model_grpo
from ..utils.logging import get_logger
from ._common import base_parser, bootstrap

LOGGER = get_logger(__name__)


def main() -> None:
    parser = base_parser(__doc__ or "train a single-model GRPO baseline")
    args = parser.parse_args()
    config = bootstrap(args)
    summary = train_single_model_grpo(config)
    LOGGER.info("single-model GRPO complete: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
