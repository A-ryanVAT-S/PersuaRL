"""Annotate a dialogue corpus with the trained expert modules.

Regenerates ``data/insuredial/expert_outputs/*.csv`` from your own expert
checkpoints, or annotates a new corpus so PersuaRL can run on it:

    python -m persuarl.cli.run_experts --config configs/experts/inference.yaml
    python -m persuarl.cli.run_experts --config configs/experts/inference.yaml --expert intent
"""

from __future__ import annotations

import gc
from pathlib import Path

import torch

from ..constants import EXPERT_KEYS
from ..data.dataset import load_dialogues
from ..experts.inference import ExpertRunner, GenerationSettings, annotate_corpus
from ..utils.logging import get_logger
from ._common import base_parser, bootstrap

LOGGER = get_logger(__name__)


def main() -> None:
    parser = base_parser(__doc__ or "run expert inference over a corpus")
    parser.add_argument("--expert", default="all", choices=[*EXPERT_KEYS, "all"])
    args = parser.parse_args()
    config = bootstrap(args)

    dialogues = load_dialogues(config.get("data.dialogues_path"))
    output_dir = Path(config.get("output_dir"))
    targets = list(EXPERT_KEYS) if args.expert == "all" else [args.expert]

    settings = GenerationSettings(
        max_new_tokens=int(config.get("generation.max_new_tokens", 96)),
        temperature=float(config.get("generation.temperature", 0.8)),
        top_p=float(config.get("generation.top_p", 0.95)),
        top_k=int(config.get("generation.top_k", 40)),
    )

    for expert in targets:
        section = config.section(f"experts.{expert}")
        runner = ExpertRunner.from_pretrained(
            expert,
            section.get("id", config.get("experts.id")),
            section.get("adapter_path", None),
            dtype=section.get("dtype", "bfloat16"),
            device_map=section.get("device_map", "auto"),
            settings=settings,
        )
        annotate_corpus(
            runner,
            dialogues,
            batch_size=int(config.get("batch_size", 8)),
            output_path=output_dir / f"{expert}.csv",
        )

        # Free the adapter before loading the next expert -- four 3B models at
        # once will not fit alongside anything else on a single 80GB card.
        del runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    LOGGER.info("expert annotation complete -> %s", output_dir)


if __name__ == "__main__":
    main()
