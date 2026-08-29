"""R3 -- contextual appropriateness, and R4 -- non-repetitiveness.

**R3** (Eq. 5) measures whether the response is semantically anchored to both
the full dialogue context and the user's latest turn, using BERTScore-F1::

    R3 = min( (BS_F1(x_i, y_i) + 2 * BS_F1(u_i, y_i)) / 3 , 1 )

The latest utterance is weighted twice because relevance to what the user *just*
said dominates perceived appropriateness. Dividing by 3 renormalises the
weighted sum back into [0, 1], and the ``min(., 1)`` clips the occasional
outlier that would otherwise dominate a GRPO group's advantage.

**R4** (Eq. 6) is one minus the Jaccard overlap between the current response and
the previous agent turn -- a cheap, model-free penalty on saying the same thing
twice, which is the failure mode SFT models fall into over long dialogues.

BERTScore is the expensive part of the reward (a full BERT forward per
candidate), so :class:`ContextualScorer` keeps one scorer instance alive rather
than paying model-load cost on every call, which is what the original
``bert_scorer(...)`` -per-batch did.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


# --------------------------------------------------------------------------
# R4 -- non-repetitiveness
# --------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """Word-level tokens for the Jaccard computation.

    NLTK's ``punkt`` gives slightly better splits, but it is an optional
    dependency and a download away, so we fall back to whitespace splitting
    rather than crashing a training run several hours in.
    """
    try:
        from nltk.tokenize import word_tokenize

        return set(word_tokenize(text.lower()))
    except Exception:
        return set(text.lower().split())


def jaccard_distance(current: str, previous: str) -> float:
    """1 - |A ∩ B| / |A ∪ B|. Returns 0.0 when there is no previous turn.

    Returning 0 (not 1) for the first turn of a conversation is deliberate: an
    opening line cannot demonstrate non-repetitiveness, and handing it a free
    full-credit reward would bias the policy toward whatever route it happened
    to pick on turn one.
    """
    if not current or not previous:
        return 0.0
    left, right = _tokenize(current), _tokenize(previous)
    union = len(left | right)
    if not union:
        return 0.0
    return 1.0 - len(left & right) / union


def non_repetitiveness_rewards(
    responses: Sequence[str],
    previous_replies: Sequence[str],
) -> np.ndarray:
    """Vectorised R4 over a batch."""
    return np.asarray(
        [jaccard_distance(response, previous) for response, previous in zip(responses, previous_replies)],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------
# R3 -- contextual appropriateness
# --------------------------------------------------------------------------


@dataclass
class ContextualScorer:
    """BERTScore-based relevance to (dialogue context, latest user utterance).

    ``reference_weight`` lets you fall back to the simpler
    "BERTScore against the ground-truth reply" formulation used by the
    single-model GRPO baselines: set ``use_reference=True`` and R3 is computed
    against ``new_agent_reply`` instead of the context pair.
    """

    model_type: str = "bert-base-uncased"
    device: str = "cuda"
    batch_size: int = 32
    utterance_weight: float = 2.0
    use_reference: bool = False

    _scorer: object | None = None

    def _get_scorer(self):
        """Lazily build and cache the ``bert_score.BERTScorer``."""
        if self._scorer is None:
            from bert_score import BERTScorer

            LOGGER.info("loading BERTScorer (%s) on %s", self.model_type, self.device)
            self._scorer = BERTScorer(
                model_type=self.model_type,
                lang="en",
                device=self.device,
                rescale_with_baseline=False,
            )
        return self._scorer

    def _f1(self, candidates: Sequence[str], references: Sequence[str]) -> np.ndarray:
        """BERTScore-F1 for aligned candidate/reference lists, empty pairs -> 0."""
        scores = np.zeros(len(candidates), dtype=np.float64)
        keep = [
            index
            for index, (cand, ref) in enumerate(zip(candidates, references))
            if str(cand).strip() and str(ref).strip()
        ]
        if not keep:
            return scores

        try:
            _, _, f1 = self._get_scorer().score(
                [str(candidates[i]) for i in keep],
                [str(references[i]) for i in keep],
                batch_size=self.batch_size,
            )
        except Exception as error:  # OOM or a tokenizer edge case
            LOGGER.warning("BERTScore failed for this batch (%s); scoring it 0", error)
            return scores

        for position, index in enumerate(keep):
            scores[index] = float(f1[position])
        return scores

    def score(
        self,
        responses: Sequence[str],
        contexts: Sequence[str],
        user_utterances: Sequence[str],
        references: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Compute R3 for a batch, clipped to [0, 1]."""
        if self.use_reference:
            if references is None:
                raise ValueError("use_reference=True requires ground-truth references")
            return np.clip(self._f1(responses, references), 0.0, 1.0)

        context_f1 = self._f1(responses, contexts)
        utterance_f1 = self._f1(responses, user_utterances)

        # (1 * context + 2 * utterance) / 3, per Eq. 5.
        total_weight = 1.0 + self.utterance_weight
        combined = (context_f1 + self.utterance_weight * utterance_f1) / total_weight
        return np.clip(combined, 0.0, 1.0)


__all__ = [
    "ContextualScorer",
    "jaccard_distance",
    "non_repetitiveness_rewards",
]
