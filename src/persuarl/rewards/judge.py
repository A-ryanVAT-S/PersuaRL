"""R5 -- LLM-as-a-judge reward.

Prometheus-2 (7B) scores each generated response 1-5 against a persuasion
rubric (see :mod:`persuarl.data.prompts`). The integer is mapped to
``(score - 1) / 4`` so R5 shares the [0, 1] range of the other rewards.

This is the most expensive reward by an order of magnitude -- one 7B generation
per rollout, i.e. ``batch_size * num_generations`` per step -- and also the
highest-weighted (``beta_5 = 0.35``). Two consequences worth knowing:

* The judge is loaded **frozen** and in ``eval`` mode, and decoding is greedy
  (``do_sample=False``). A stochastic judge would inject variance straight into
  the advantage estimate.
* Parsing is defensive. A malformed judge output falls back to the neutral
  score 3 rather than aborting a multi-hour run; the fallback rate is logged so
  you can tell a flaky batch from a broken prompt.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from ..data.prompts import JUDGE_SYSTEM_PROMPT, build_judge_prompt
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

#: Prometheus is asked to end with "[RESULT] <n>"; tolerate whitespace variants.
_RESULT_PATTERN = re.compile(r"\[RESULT\]\s*([1-5])")

#: Neutral score used when parsing fails. 3/5 -> 0.5 after normalisation.
NEUTRAL_SCORE = 3


def parse_judge_score(text: str) -> tuple[int, bool]:
    """Extract the 1-5 verdict. Returns ``(score, parsed_ok)``."""
    match = _RESULT_PATTERN.search(text or "")
    if match:
        return int(match.group(1)), True
    return NEUTRAL_SCORE, False


def normalize_score(score: int) -> float:
    """1..5 -> 0.0..1.0."""
    return (score - 1.0) / 4.0


@dataclass
class JudgeScorer:
    """Frozen Prometheus-style judge wrapped as a batched reward function."""

    model: torch.nn.Module
    tokenizer: object
    max_new_tokens: int = 200
    max_prompt_tokens: int = 2048
    batch_size: int = 8

    #: Running tally of unparseable outputs, reported by :meth:`stats`.
    _failures: int = field(default=0, init=False)
    _calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.model.eval()
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding: a right-padded batch would make the model continue from
        # pad tokens instead of from the end of the rubric.
        self.tokenizer.padding_side = "left"

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        dtype: str = "float16",
        device_map: str = "auto",
        **kwargs,
    ) -> JudgeScorer:
        from ..models.loader import load_causal_lm, load_tokenizer

        LOGGER.info("loading judge model %s", model_id)
        tokenizer = load_tokenizer(model_id, padding_side="left")
        model = load_causal_lm(model_id, dtype=dtype, device_map=device_map)
        return cls(model=model, tokenizer=tokenizer, **kwargs)

    def _build_prompts(
        self,
        responses: Sequence[str],
        contexts: Sequence[str],
        user_utterances: Sequence[str],
    ) -> list[str]:
        prompts: list[str] = []
        for response, context, utterance in zip(responses, contexts, user_utterances):
            instruction = (
                f"Given the conversation history:\n{context or 'N/A'}\n\n"
                f"Respond to the user's latest utterance:\n{utterance}"
            )
            messages = [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": build_judge_prompt(instruction, response or "(empty response)")},
            ]
            prompts.append(
                self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
        return prompts

    @torch.no_grad()
    def score(
        self,
        responses: Sequence[str],
        contexts: Sequence[str],
        user_utterances: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Judge a batch.

        Returns ``(normalized, raw)`` -- normalized in [0, 1] for the composite
        reward, raw 1-5 integers for the training log (they are much easier to
        eyeball than 0.63).
        """
        prompts = self._build_prompts(responses, contexts, user_utterances)
        raw_scores: list[int] = []
        failures = 0

        for start in range(0, len(prompts), self.batch_size):
            chunk = prompts[start:start + self.batch_size]
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_prompt_tokens,
            ).to(self.model.device)

            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            # Slice off the prompt; keeping it would let "[RESULT]" text from the
            # rubric itself get parsed as the verdict.
            completions = self.tokenizer.batch_decode(
                generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )

            for completion in completions:
                score, ok = parse_judge_score(completion)
                raw_scores.append(score)
                failures += 0 if ok else 1

        self._calls += len(prompts)
        self._failures += failures
        if failures:
            LOGGER.debug("judge: %d/%d outputs unparseable, defaulted to %d",
                         failures, len(prompts), NEUTRAL_SCORE)

        raw = np.asarray(raw_scores, dtype=np.float64)
        return np.asarray([normalize_score(s) for s in raw_scores], dtype=np.float64), raw

    def stats(self) -> dict[str, float]:
        """Parse-failure rate so far. A rate above a few percent means the judge
        prompt and the judge checkpoint have drifted apart -- check both."""
        return {
            "judge_calls": float(self._calls),
            "judge_parse_failures": float(self._failures),
            "judge_failure_rate": self._failures / self._calls if self._calls else 0.0,
        }


class ConstantJudge:
    """Stand-in judge returning a fixed score.

    Used by ``--ablate R5`` and by the smoke tests, so a run can exercise the
    full reward path without a 7B model resident in memory.
    """

    def __init__(self, score: int = NEUTRAL_SCORE) -> None:
        self.score = score

    def score_batch(self, size: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.full(size, normalize_score(self.score), dtype=np.float64),
            np.full(size, float(self.score), dtype=np.float64),
        )

    def score(self, responses, contexts, user_utterances):  # noqa: D102 - matches JudgeScorer
        return self.score_batch(len(responses))

    def stats(self) -> dict[str, float]:
        return {}


__all__ = ["ConstantJudge", "JudgeScorer", "normalize_score", "parse_judge_score"]
