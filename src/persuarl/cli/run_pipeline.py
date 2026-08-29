"""Run the full Selector -> Experts -> Generator pipeline over the test split.

    python -m persuarl.cli.run_pipeline --config configs/inference/persuarl.yaml

Ablation modes (D.3.2), all from the same config:

    --set routing_mode=all         # AllExpert: every expert, every turn
    --set routing_mode=prompting   # prompted selection, no RL
    --set expert_source=live       # call the expert LMs instead of the cache
"""

from __future__ import annotations

from ..inference.pipeline import run_inference
from ..utils.logging import get_logger
from ..utils.seeding import seed_everything
from ._common import base_parser, bootstrap

LOGGER = get_logger(__name__)


def main() -> None:
    parser = base_parser(__doc__ or "run the PersuaRL inference pipeline")
    args = parser.parse_args()
    config = bootstrap(args)
    seed_everything(int(config.get("seed", 42)))
    frame = run_inference(config)
    LOGGER.info("inference complete: %d turns generated", len(frame))


if __name__ == "__main__":
    main()
