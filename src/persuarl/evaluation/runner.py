"""Scoring an inference CSV with every automatic metric.

Input is whatever ``persuarl.cli.run_pipeline`` wrote (or any CSV with a
candidate column and a reference column). Output is the same table plus one
column per metric, and a printed summary of the corpus means.

The optional LLM-as-a-judge column reuses the *training* judge
(:class:`persuarl.rewards.judge.JudgeScorer`), so the LLM-J number in the
results table and the R5 signal during training come from the same rubric.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..utils.logging import get_logger
from .metrics import MetricSummary, bertscore_f1, bleu2, distinct_n, meteor, perplexity, rouge1

LOGGER = get_logger(__name__)


def evaluate_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    candidate_column: str = "model_reply",
    reference_column: str = "reference_reply",
    context_column: str = "dialogue_context",
    utterance_column: str = "user_utterance",
    perplexity_model_id: str | None = None,
    bertscore_model: str = "bert-base-uncased",
    judge_model_id: str | None = None,
    device: str | None = None,
) -> MetricSummary:
    """Score every row of ``input_path`` and write the annotated CSV."""
    frame = pd.read_csv(input_path)
    for column in (candidate_column, reference_column):
        if column not in frame.columns:
            raise ValueError(
                f"{input_path}: expected column {column!r}; found {list(frame.columns)}"
            )

    candidates = frame[candidate_column].fillna("").astype(str).tolist()
    references = frame[reference_column].fillna("").astype(str).tolist()
    LOGGER.info("scoring %d generated replies from %s", len(candidates), input_path)

    # -- per-row lexical metrics --------------------------------------------
    frame["BLEU-2"] = [
        bleu2(candidate, reference) for candidate, reference in zip(candidates, references)
    ]
    frame["ROUGE-1"] = [
        rouge1(candidate, reference) for candidate, reference in zip(candidates, references)
    ]
    frame["METEOR"] = [
        meteor(reference, candidate) for candidate, reference in zip(candidates, references)
    ]
    frame["Distinct-2"] = [distinct_n(candidate, 2) for candidate in candidates]

    # -- batched BERTScore ---------------------------------------------------
    frame["BERTScore-F1"] = bertscore_f1(
        candidates, references, model_type=bertscore_model, device=device
    )

    # -- optional: perplexity under a reference LM ---------------------------
    if perplexity_model_id:
        from ..models.loader import load_causal_lm, load_tokenizer

        LOGGER.info("computing perplexity under %s", perplexity_model_id)
        tokenizer = load_tokenizer(perplexity_model_id)
        model = load_causal_lm(perplexity_model_id, dtype="auto", device_map="auto")
        model.eval()
        frame["PPL"] = [
            perplexity(candidate, model, tokenizer)
            for candidate in tqdm(candidates, desc="perplexity")
        ]
        del model

    # -- optional: LLM-as-a-judge -------------------------------------------
    if judge_model_id:
        from ..rewards.judge import JudgeScorer

        LOGGER.info("scoring with judge %s", judge_model_id)
        judge = JudgeScorer.from_pretrained(judge_model_id)
        contexts = (
            frame[context_column].fillna("").astype(str).tolist()
            if context_column in frame.columns else [""] * len(frame)
        )
        utterances = (
            frame[utterance_column].fillna("").astype(str).tolist()
            if utterance_column in frame.columns else [""] * len(frame)
        )
        _, raw = judge.score(candidates, contexts, utterances)
        frame["LLM-Judge"] = raw
        LOGGER.info("judge stats: %s", judge.stats())

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    metric_columns = [
        column
        for column in ("BLEU-2", "METEOR", "BERTScore-F1", "Distinct-2", "ROUGE-1", "PPL", "LLM-Judge")
        if column in frame.columns
    ]
    # nanmean: perplexity is NaN for empty replies, and one NaN would otherwise
    # take the whole column with it.
    summary = MetricSummary(
        {column: float(np.nanmean(frame[column].to_numpy(dtype=float))) for column in metric_columns}
    )

    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    LOGGER.info("wrote per-turn scores -> %s", output_path)
    LOGGER.info("wrote summary -> %s", summary_path)
    print("\n=== Average metrics ===")
    print(summary)
    return summary


__all__ = ["evaluate_file"]
