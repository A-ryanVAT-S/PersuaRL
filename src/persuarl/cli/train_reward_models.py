"""Train the R1/R2 reward classifiers and build their class prototypes.

    python -m persuarl.cli.train_reward_models --config configs/rewards/classifiers.yaml

Writes, per dimension, a HuggingFace classifier directory plus a
``prototypes.pt`` tensor inside it. Both paths then go into the `rewards:`
block of your RL config.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..constants import ENGAGEMENT_LABELS, INTENT_LABELS
from ..rewards.classifiers import ClassifierSpec, train_classifier
from ..utils.logging import get_logger
from ..utils.seeding import seed_everything
from ._common import base_parser, bootstrap

LOGGER = get_logger(__name__)

_LABEL_SETS = {
    "engagement": ENGAGEMENT_LABELS,
    "intent": INTENT_LABELS,
}


def main() -> None:
    parser = base_parser(__doc__ or "train PersuaRL reward classifiers")
    parser.add_argument(
        "--dimension",
        default="all",
        choices=["engagement", "intent", "all"],
        help="which reward classifier to train",
    )
    args = parser.parse_args()
    config = bootstrap(args)
    seed_everything(int(config.get("seed", 42)))

    targets = ["engagement", "intent"] if args.dimension == "all" else [args.dimension]
    output_root = Path(config.get("output_dir"))
    results: dict[str, dict[str, float]] = {}

    for dimension in targets:
        section = config.section(dimension)
        spec = ClassifierSpec(
            name=dimension,
            labels=_LABEL_SETS[dimension],
            dataset_path=section.get("dataset_path"),
            text_column=section.get("text_column", "utterance"),
            label_column=section.get("label_column", "label"),
            # Engagement annotations are free text ("Logical appeal"); intent
            # annotations are already canonical identifiers.
            normalize_labels=dimension == "engagement",
        )
        results[dimension] = train_classifier(
            spec,
            base_model_id=section.get("base_model_id", config.get("base_model_id", "bert-base-uncased")),
            output_dir=output_root / f"{dimension}_classifier",
            epochs=int(config.get("train.epochs", 2)),
            batch_size=int(config.get("train.batch_size", 16)),
            learning_rate=float(config.get("train.learning_rate", 2e-5)),
            weight_decay=float(config.get("train.weight_decay", 0.01)),
            max_length=int(config.get("train.max_length", 512)),
            seed=int(config.get("seed", 42)),
            fp16=bool(config.get("train.fp16", True)),
        )

    LOGGER.info("reward classifiers complete:\n%s", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
