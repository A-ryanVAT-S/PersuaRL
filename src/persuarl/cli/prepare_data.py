"""Derive the expert / classifier training files from the shipped CSVs.

``data/insuredial/expert_outputs/*.csv`` hold each expert's *rendered* answer as
prose ("The persuasion strategy is Logical appeal and the reason is ..."). The
expert LMs and the reward classifiers both need that split back into
``(utterance, label, reason)``. This command does the split, using each expert's
own pattern from :mod:`persuarl.experts.registry`, and writes:

    data/processed/experts/<expert>.csv     conversation_id, turn_no, utterance,
                                            new_agent_reply, label, reason
    data/processed/classifiers/engagement.csv   utterance, label
    data/processed/classifiers/intent.csv       utterance, label

Run it once after cloning::

    python -m persuarl.cli.prepare_data --config configs/data.yaml

Rows whose answer does not parse are reported and dropped, not guessed at.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..constants import (
    COL_AGENT_REPLY,
    COL_CONVERSATION_ID,
    COL_TURN_NO,
    ENGAGEMENT_LABELS,
    EXPERT_ANSWER_COLUMN,
    EXPERT_KEYS,
    INTENT_LABELS,
)
from ..data.dataset import load_dialogues, load_expert_outputs
from ..experts.registry import parse_answer
from ..utils.logging import get_logger
from ._common import base_parser, bootstrap

LOGGER = get_logger(__name__)


def _canonical_engagement(label: str) -> str:
    """Map free-text strategy annotations onto the six canonical labels."""
    cleaned = str(label).strip().lower().replace(" appeal", "").replace("-based", "")
    return cleaned if cleaned in ENGAGEMENT_LABELS else ""


def _canonical_intent(label: str) -> str:
    """Intent annotations are already identifiers; just normalise whitespace."""
    cleaned = str(label).strip().replace(" ", "_")
    for known in INTENT_LABELS:
        if cleaned.lower() == known.lower():
            return known
    return ""


def build_expert_training_file(
    expert: str,
    dialogues: pd.DataFrame,
    answers_path: str | Path,
    output_path: Path,
) -> pd.DataFrame:
    """Split one expert's rendered answers into label/reason training rows."""
    answers = load_expert_outputs(expert, answers_path)
    answer_column = EXPERT_ANSWER_COLUMN.format(expert=expert)

    merged = dialogues.merge(answers, on=[COL_CONVERSATION_ID, COL_TURN_NO], how="inner")

    labels: list[str] = []
    reasons: list[str] = []
    unparsed = 0
    for raw in merged[answer_column]:
        parsed = parse_answer(expert, raw)
        if parsed is None:
            unparsed += 1
            labels.append("")
            reasons.append("")
        else:
            labels.append(parsed["label"])
            reasons.append(parsed["reason"])

    merged["label"] = labels
    merged["reason"] = reasons

    before = len(merged)
    merged = merged[merged["label"].astype(bool)]
    if unparsed:
        LOGGER.warning("%s: %d/%d answers did not parse and were dropped", expert, unparsed, before)

    result = merged.rename(columns={"user_utterance": "utterance"})[
        [COL_CONVERSATION_ID, COL_TURN_NO, "utterance", COL_AGENT_REPLY, "label", "reason"]
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    LOGGER.info("%-10s -> %s (%d rows)", expert, output_path, len(result))
    return result


def build_classifier_file(
    expert: str,
    expert_frame: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Reduce an expert training file to the ``(utterance, label)`` pairs the
    reward classifier needs, with labels normalised to the closed set."""
    canonicalise = _canonical_engagement if expert == "engagement" else _canonical_intent

    frame = expert_frame[["utterance", "label"]].copy()
    frame["label"] = frame["label"].map(canonicalise)

    before = len(frame)
    frame = frame[frame["label"].astype(bool)]
    dropped = before - len(frame)
    if dropped:
        # Multi-label annotations ("logical and emotional") have no single-label
        # mapping; the classifier is single-label by construction.
        LOGGER.warning("%s classifier: dropped %d rows with non-canonical labels", expert, dropped)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    LOGGER.info("%-10s classifier data -> %s (%d rows, %d classes)",
                expert, output_path, len(frame), frame["label"].nunique())
    return frame


def main() -> None:
    parser = base_parser(__doc__ or "prepare PersuaRL training files")
    args = parser.parse_args()
    config = bootstrap(args)

    dialogues = load_dialogues(config.get("data.dialogues_path"))
    expert_paths = config.section("data.expert_outputs").as_dict()
    expert_out = Path(config.get("output.experts_dir"))
    classifier_out = Path(config.get("output.classifiers_dir"))

    expert_frames: dict[str, pd.DataFrame] = {}
    for expert in EXPERT_KEYS:
        path = expert_paths.get(expert)
        if not path:
            LOGGER.warning("no expert_outputs path configured for %s; skipping", expert)
            continue
        expert_frames[expert] = build_expert_training_file(
            expert, dialogues, path, expert_out / f"{expert}.csv"
        )

    # Only engagement and intent have reward classifiers (R1 and R2).
    for expert in ("engagement", "intent"):
        if expert in expert_frames:
            build_classifier_file(expert, expert_frames[expert], classifier_out / f"{expert}.csv")

    LOGGER.info("data preparation complete")


if __name__ == "__main__":
    main()
