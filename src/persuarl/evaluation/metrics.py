"""Automatic metrics for Table 2.

Six numbers per generated reply:

BLEU-2       n-gram precision against the reference, weights (0.5, 0.5),
             smoothing method 1 (short dialogue turns make raw BLEU degenerate).
METEOR       unigram F-mean with a length penalty.
BERTScore-F1 contextual-embedding similarity to the reference.
Distinct-2   distinct bigrams over total bigrams -- diversity, no reference.
ROUGE-1      unigram F-measure against the reference.
Perplexity   the reply's PPL under a reference LM -- fluency, no reference text.

BERTScore is batched over the whole file rather than computed per row: the
original notebook called ``score([one], [one])`` inside the loop, which reloads
and re-runs the scorer per example and dominates the runtime.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


# --------------------------------------------------------------------------
# Reference-based lexical metrics
# --------------------------------------------------------------------------


def bleu2(candidate: str, reference: str) -> float:
    """Sentence-level BLEU with uniform weights over 1- and 2-grams."""
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    candidate_tokens = str(candidate).strip().split()
    reference_tokens = str(reference).strip().split()
    if not candidate_tokens or not reference_tokens:
        return 0.0

    return float(
        sentence_bleu(
            [reference_tokens],
            candidate_tokens,
            weights=(0.5, 0.5),
            smoothing_function=SmoothingFunction().method1,
        )
    )


def rouge1(candidate: str, reference: str) -> float:
    """ROUGE-1 F-measure with Porter stemming."""
    from rouge_score import rouge_scorer

    if not str(candidate).strip() or not str(reference).strip():
        return 0.0
    scorer = _rouge_scorer_singleton(rouge_scorer)
    return float(scorer.score(str(reference), str(candidate))["rouge1"].fmeasure)


_ROUGE_SCORER = None


def _rouge_scorer_singleton(module):
    """One scorer for the whole run -- constructing it per row is measurable."""
    global _ROUGE_SCORER
    if _ROUGE_SCORER is None:
        _ROUGE_SCORER = module.RougeScorer(["rouge1"], use_stemmer=True)
    return _ROUGE_SCORER


def _normalize(text: str) -> list[str]:
    """Lower-case, strip punctuation, split on whitespace."""
    if not isinstance(text, str):
        return []
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).split()


def meteor(reference: str, hypothesis: str, alpha: float = 0.5) -> float:
    """Simplified METEOR: unigram precision/recall with a fragmentation penalty.

    This is the variant used to produce the paper's MT column -- exact-match
    unigrams only, no stemming or synonym stages -- kept as-is so the reported
    numbers stay reproducible. Values are not comparable to full METEOR.
    """
    reference_tokens = _normalize(reference)
    hypothesis_tokens = _normalize(hypothesis)
    if not reference_tokens or not hypothesis_tokens:
        return 0.0

    overlap = len(set(reference_tokens) & set(hypothesis_tokens))
    precision = overlap / len(hypothesis_tokens)
    recall = overlap / len(reference_tokens)
    if precision == 0 and recall == 0:
        return 0.0

    penalty = alpha * (len(reference_tokens) / (len(reference_tokens) + len(hypothesis_tokens)))
    return float(precision * recall / (precision + (1 - alpha) * recall + penalty))


# --------------------------------------------------------------------------
# Reference-free metrics
# --------------------------------------------------------------------------


def distinct_n(text: str, n: int = 2) -> float:
    """Ratio of unique n-grams to total n-grams -- lexical diversity."""
    words = str(text).strip().split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


@torch.no_grad()
def perplexity(text: str, model, tokenizer, *, max_length: int = 512) -> float:
    """Token-level perplexity of ``text`` under ``model``.

    Note this scores fluency *of the generated text under a reference LM*; it is
    not the model's own training perplexity, and a very short reply can score
    deceptively low.
    """
    if not str(text).strip():
        return float("nan")
    inputs = tokenizer(str(text), return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(model.device)
    if input_ids.shape[1] < 2:
        return float("nan")
    loss = model(input_ids=input_ids, labels=input_ids).loss
    return float(math.exp(loss.item()))


# --------------------------------------------------------------------------
# Batched BERTScore
# --------------------------------------------------------------------------


def bertscore_f1(
    candidates: Sequence[str],
    references: Sequence[str],
    *,
    model_type: str = "bert-base-uncased",
    device: str | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    """BERTScore-F1 for the whole file at once."""
    from bert_score import BERTScorer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    scores = np.zeros(len(candidates), dtype=np.float64)

    keep = [
        index
        for index, (candidate, reference) in enumerate(zip(candidates, references))
        if str(candidate).strip() and str(reference).strip()
    ]
    if not keep:
        LOGGER.warning("no candidate/reference pairs to score with BERTScore")
        return scores

    scorer = BERTScorer(model_type=model_type, lang="en", device=device)
    _, _, f1 = scorer.score(
        [str(candidates[i]) for i in keep],
        [str(references[i]) for i in keep],
        batch_size=batch_size,
    )
    for position, index in enumerate(keep):
        scores[index] = float(f1[position])
    return scores


@dataclass
class MetricSummary:
    """Corpus-level means, i.e. the row you paste into the results table."""

    values: dict[str, float]

    def __str__(self) -> str:
        width = max(len(name) for name in self.values)
        lines = [f"{name:<{width}} : {value:.4f}" for name, value in self.values.items()]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, float]:
        return dict(self.values)


__all__ = [
    "MetricSummary",
    "bertscore_f1",
    "bleu2",
    "distinct_n",
    "meteor",
    "perplexity",
    "rouge1",
]
