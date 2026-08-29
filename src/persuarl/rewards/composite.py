"""The composite reward ``R = sum_k beta_k * R_k``.

One class, :class:`PersuasiveRewardModel`, scores a batch of candidate
responses and returns a full per-rollout breakdown. Both training regimes use
it unchanged:

* **PersuaRL** (``training/grpo_selector.py``) -- the policy emits a route, the
  frozen Generator turns it into a response, and *that* response is scored.
* **Single-model GRPO baselines** (``training/grpo_single.py``) -- the policy's
  own completion is the response.

Keeping one implementation is the whole point: in the original scripts the
baseline and the main method each had their own copy of the reward, and they
had already drifted (different R1/R2 formulations, different judge prompts),
which makes the ablation table not quite an ablation.

Default weights are the paper's tuned setting (Table 11)::

    beta_1 = 0.15  engagement strategy consistency  (R1)
    beta_2 = 0.15  intent consistency               (R2)
    beta_3 = 0.20  contextual appropriateness       (R3)
    beta_4 = 0.15  non-repetitiveness               (R4)
    beta_5 = 0.35  LLM-as-a-judge                   (R5)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..utils.logging import get_logger
from .contextual import ContextualScorer, non_repetitiveness_rewards

LOGGER = get_logger(__name__)

#: Canonical reward names, in the order they appear in the paper.
REWARD_NAMES: tuple[str, ...] = ("engagement", "intent", "contextual", "repetition", "judge")


@dataclass
class RewardWeights:
    """The five ``beta_k``. They should sum to 1 -- we check and warn if not."""

    engagement: float = 0.15   # beta_1, R1
    intent: float = 0.15       # beta_2, R2
    contextual: float = 0.20   # beta_3, R3
    repetition: float = 0.15   # beta_4, R4
    judge: float = 0.35        # beta_5, R5

    def __post_init__(self) -> None:
        total = self.total()
        if abs(total - 1.0) > 1e-6:
            LOGGER.warning(
                "reward weights sum to %.4f, not 1.0 -- rewards will not be in [0, 1] "
                "and are not comparable to the paper's numbers", total
            )

    def total(self) -> float:
        return sum(getattr(self, name) for name in REWARD_NAMES)

    def as_array(self) -> np.ndarray:
        return np.asarray([getattr(self, name) for name in REWARD_NAMES], dtype=np.float64)

    @classmethod
    def from_config(cls, section) -> RewardWeights:
        return cls(
            engagement=float(section.get("engagement", 0.15)),
            intent=float(section.get("intent", 0.15)),
            contextual=float(section.get("contextual", 0.20)),
            repetition=float(section.get("repetition", 0.15)),
            judge=float(section.get("judge", 0.35)),
        )

    def without(self, *names: str) -> RewardWeights:
        """Zero out one or more components -- the reward ablation of Table 12.

        The remaining weights are *not* renormalised, matching the paper: the
        ablation asks what each term contributes, not how the model does with a
        rebalanced objective.
        """
        weights = {name: getattr(self, name) for name in REWARD_NAMES}
        for name in names:
            if name not in weights:
                raise ValueError(f"unknown reward component {name!r}; expected one of {REWARD_NAMES}")
            weights[name] = 0.0
        # Bypass __post_init__: an ablated objective deliberately sums to < 1,
        # and warning about that on every construction is just noise.
        instance = object.__new__(type(self))
        for key, value in weights.items():
            setattr(instance, key, value)
        return instance


@dataclass
class RewardBreakdown:
    """Per-component scores for a batch. Every array has length ``batch_size``."""

    engagement: np.ndarray
    intent: np.ndarray
    contextual: np.ndarray
    repetition: np.ndarray
    judge: np.ndarray
    judge_raw: np.ndarray
    total: np.ndarray

    def component(self, name: str) -> np.ndarray:
        return getattr(self, name)

    def means(self) -> dict[str, float]:
        """Batch means, for the TensorBoard/W&B scalars."""
        summary = {f"reward/{name}": float(np.mean(self.component(name))) for name in REWARD_NAMES}
        summary["reward/judge_raw"] = float(np.mean(self.judge_raw))
        summary["reward/total"] = float(np.mean(self.total))
        return summary

    def format_row(self, index: int) -> str:
        """Compact one-line view of a single rollout, for the console log."""
        return (
            f"R={self.total[index]:.3f} | "
            f"judge={self.judge_raw[index]:.0f}/5 "
            f"eng={self.engagement[index]:.2f} "
            f"int={self.intent[index]:.2f} "
            f"ctx={self.contextual[index]:.2f} "
            f"rep={self.repetition[index]:.2f}"
        )


@dataclass
class PersuasiveRewardModel:
    """Scores candidate responses on all five persuasion dimensions.

    Every scorer is optional. A missing engagement scorer contributes 0 to R1
    rather than crashing, which is what makes ``--ablate`` runs and
    reward-model-free smoke tests work off the same code path. Ablating a
    component means zeroing its *weight*; a missing scorer just means the
    component is unavailable.
    """

    weights: RewardWeights = field(default_factory=RewardWeights)
    engagement_scorer: object | None = None     # PrototypeScorer, R1
    intent_scorer: object | None = None         # PrototypeScorer, R2
    contextual_scorer: ContextualScorer | None = None  # R3
    judge: object | None = None                 # JudgeScorer / ConstantJudge, R5

    def __post_init__(self) -> None:
        missing = [
            name for name, scorer in (
                ("R1 engagement", self.engagement_scorer),
                ("R2 intent", self.intent_scorer),
                ("R3 contextual", self.contextual_scorer),
                ("R5 judge", self.judge),
            )
            if scorer is None
        ]
        if missing:
            LOGGER.warning("reward components unavailable (scored as 0): %s", ", ".join(missing))

    def score(
        self,
        responses: Sequence[str],
        *,
        user_utterances: Sequence[str],
        contexts: Sequence[str],
        previous_replies: Sequence[str],
        references: Sequence[str] | None = None,
    ) -> RewardBreakdown:
        """Score a batch of generated responses.

        ``responses`` may contain empty strings (an invalid route produced no
        response); those score 0 on every component that consults the text.
        """
        size = len(responses)
        zeros = np.zeros(size, dtype=np.float64)

        # -- R1 / R2: prototype consistency against the user's turn -----------
        engagement = (
            self.engagement_scorer.score(user_utterances, responses).numpy()
            if self.engagement_scorer is not None else zeros.copy()
        )
        intent = (
            self.intent_scorer.score(user_utterances, responses).numpy()
            if self.intent_scorer is not None else zeros.copy()
        )

        # -- R3: contextual appropriateness (BERTScore) -----------------------
        contextual = (
            self.contextual_scorer.score(responses, contexts, user_utterances, references)
            if self.contextual_scorer is not None else zeros.copy()
        )

        # -- R4: non-repetitiveness (lexical, no model) -----------------------
        repetition = non_repetitiveness_rewards(responses, previous_replies)

        # -- R5: LLM-as-a-judge ----------------------------------------------
        if self.judge is not None:
            judge, judge_raw = self.judge.score(responses, contexts, user_utterances)
        else:
            judge, judge_raw = zeros.copy(), zeros.copy()

        # An empty response cannot earn credit anywhere.
        empty = np.asarray([not str(r).strip() for r in responses])
        for array in (engagement, intent, contextual, repetition, judge):
            array[empty] = 0.0

        total = (
            self.weights.engagement * engagement
            + self.weights.intent * intent
            + self.weights.contextual * contextual
            + self.weights.repetition * repetition
            + self.weights.judge * judge
        )

        return RewardBreakdown(
            engagement=engagement,
            intent=intent,
            contextual=contextual,
            repetition=repetition,
            judge=judge,
            judge_raw=judge_raw,
            total=np.clip(total, 0.0, 1.0),
        )


def build_reward_model(config, *, device: str = "cuda") -> PersuasiveRewardModel:
    """Assemble a :class:`PersuasiveRewardModel` from a ``rewards:`` config block.

    Loading is lazy in the sense that anything the config leaves unset is simply
    not loaded -- useful when you want R1-R4 on a single GPU without the 7B
    judge resident.
    """
    from .consistency import PrototypeScorer
    from .judge import ConstantJudge, JudgeScorer

    weights = RewardWeights.from_config(config.section("weights"))
    for name in config.get("ablate", []) or []:
        LOGGER.info("ablating reward component: %s", name)
        weights = weights.without(name)

    engagement_scorer = None
    if config.get("engagement.classifier_path", None):
        engagement_scorer = PrototypeScorer.from_pretrained(
            config.get("engagement.classifier_path"),
            config.get("engagement.prototypes_path"),
            device=device,
            lambda_confidence=float(config.get("engagement.lambda_confidence", 0.1)),
            name="engagement",
        )

    intent_scorer = None
    if config.get("intent.classifier_path", None):
        intent_scorer = PrototypeScorer.from_pretrained(
            config.get("intent.classifier_path"),
            config.get("intent.prototypes_path"),
            device=device,
            lambda_confidence=float(config.get("intent.lambda_confidence", 0.1)),
            name="intent",
        )

    contextual_scorer = ContextualScorer(
        model_type=config.get("contextual.embedding_model", "bert-base-uncased"),
        device=device,
        batch_size=int(config.get("contextual.batch_size", 32)),
        utterance_weight=float(config.get("contextual.utterance_weight", 2.0)),
        use_reference=bool(config.get("contextual.use_reference", False)),
    )

    judge_id = config.get("judge.model_id", None)
    if judge_id in (None, "", "none"):
        LOGGER.info("no judge model configured; R5 uses a constant neutral score")
        judge = ConstantJudge()
    else:
        judge = JudgeScorer.from_pretrained(
            judge_id,
            dtype=config.get("judge.dtype", "float16"),
            device_map=config.get("judge.device_map", "auto"),
            max_new_tokens=int(config.get("judge.max_new_tokens", 200)),
            batch_size=int(config.get("judge.batch_size", 8)),
        )

    return PersuasiveRewardModel(
        weights=weights,
        engagement_scorer=engagement_scorer,
        intent_scorer=intent_scorer,
        contextual_scorer=contextual_scorer,
        judge=judge,
    )


__all__ = [
    "REWARD_NAMES",
    "PersuasiveRewardModel",
    "RewardBreakdown",
    "RewardWeights",
    "build_reward_model",
]
