"""Score an inference CSV with the automatic metrics of Table 2.

    python -m persuarl.cli.evaluate --config configs/eval/default.yaml
    python -m persuarl.cli.evaluate --config configs/eval/default.yaml \
        --set input_path=results/persuarl_llama3b.csv
"""

from __future__ import annotations

from ..evaluation.runner import evaluate_file
from ..utils.logging import get_logger
from ._common import base_parser, bootstrap

LOGGER = get_logger(__name__)


def main() -> None:
    parser = base_parser(__doc__ or "compute automatic metrics")
    args = parser.parse_args()
    config = bootstrap(args)

    evaluate_file(
        config.get("input_path"),
        config.get("output_path"),
        candidate_column=config.get("candidate_column", "model_reply"),
        reference_column=config.get("reference_column", "reference_reply"),
        perplexity_model_id=config.get("perplexity_model_id", None),
        bertscore_model=config.get("bertscore_model", "bert-base-uncased"),
        judge_model_id=config.get("judge_model_id", None),
        device=config.get("device", None),
    )


if __name__ == "__main__":
    main()
