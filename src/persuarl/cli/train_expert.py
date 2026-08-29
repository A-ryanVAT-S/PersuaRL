"""Fine-tune one expert module (or all four in sequence).

    python -m persuarl.cli.train_expert --config configs/experts/intent.yaml
    python -m persuarl.cli.train_expert --config configs/experts/base.yaml --expert all

The backbone comes from the config, so the same command trains a Llama, Qwen or
Phi expert without touching code.
"""

from __future__ import annotations

import json

from ..constants import EXPERT_KEYS
from ..experts.training import train_expert
from ..utils.logging import get_logger
from ._common import base_parser, bootstrap

LOGGER = get_logger(__name__)


def main() -> None:
    parser = base_parser(__doc__ or "train a PersuaRL expert module")
    parser.add_argument(
        "--expert",
        default=None,
        choices=[*EXPERT_KEYS, "all"],
        help="which expert to train; defaults to the config's `expert` key",
    )
    args = parser.parse_args()
    config = bootstrap(args)

    requested = args.expert or config.get("expert", "all")
    targets = list(EXPERT_KEYS) if requested == "all" else [requested]

    results: dict[str, dict[str, float]] = {}
    for expert in targets:
        results[expert] = train_expert(config, expert)

    LOGGER.info("expert training complete:\n%s", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
