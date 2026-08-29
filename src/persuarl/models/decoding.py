"""Constrained decoding for the Selector.

The Selector's output must be a legal route and nothing else. Two strategies
live here:

:class:`SingleTokenChoiceProcessor`
    Used during **GRPO training**. The action is one letter, so we mask every
    logit except the route tokens on the first step and force EOS afterwards.
    ``max_completion_length=4`` in the trainer config is then generous.

:class:`AllowedSequencesProcessor`
    Used when you want the Selector to emit a full ``<route>{...}</route>``
    JSON string (the original inference format, kept for backwards
    compatibility with checkpoints trained that way). It walks a prefix trie of
    the allowed strings.

Both are stateless across ``generate`` calls *except* for the prompt boundary,
which they capture on the first call. That is why :func:`constrained_generate`
builds a **fresh** processor per call -- reusing one across batches of different
prompt lengths was a real bug in the original script.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from ..routes import ROUTE_LABELS


class SingleTokenChoiceProcessor(LogitsProcessor):
    """Force the first generated token into ``allowed_token_ids``, then stop.

    Implementation note: we overwrite ``scores`` with ``-inf`` everywhere except
    the allowed ids rather than adding a bias, so the renormalised distribution
    is exactly the policy's distribution *restricted* to the action space. That
    keeps the log-probs GRPO computes consistent with the actions it observes.
    """

    def __init__(self, allowed_token_ids: Sequence[int], eos_token_id: int) -> None:
        if not allowed_token_ids:
            raise ValueError("allowed_token_ids is empty; the Selector would have no actions")
        self.allowed_token_ids = list(allowed_token_ids)
        self.eos_token_id = eos_token_id
        self._prompt_length: int | None = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self._prompt_length is None:
            self._prompt_length = input_ids.shape[1]

        generated_so_far = input_ids.shape[1] - self._prompt_length
        mask = torch.full_like(scores, float("-inf"))

        if generated_so_far == 0:
            # First step: only route letters are legal, at their original scores.
            index = torch.tensor(self.allowed_token_ids, device=scores.device)
            mask[:, index] = scores[:, index]
        else:
            # The action is complete; the only legal continuation is stopping.
            mask[:, self.eos_token_id] = 0.0

        return mask


class AllowedSequencesProcessor(LogitsProcessor):
    """Restrict generation to a finite set of exact strings, via prefix matching."""

    def __init__(self, tokenizer, allowed_strings: Sequence[str], eos_token_id: int) -> None:
        self.allowed_sequences = [
            tokenizer.encode(text, add_special_tokens=False) for text in allowed_strings
        ]
        self.eos_token_id = eos_token_id
        self._prompt_length: int | None = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self._prompt_length is None:
            self._prompt_length = input_ids.shape[1]

        generated = input_ids.shape[1] - self._prompt_length
        output = torch.full_like(scores, float("-inf"))

        for row in range(input_ids.shape[0]):
            produced = input_ids[row, self._prompt_length:].tolist()

            # Already spelled out a complete allowed string -> force EOS.
            if any(seq == produced for seq in self.allowed_sequences):
                output[row, self.eos_token_id] = 0.0
                continue

            live = [seq for seq in self.allowed_sequences if seq[:generated] == produced]
            if not live:
                # Off-trie (shouldn't happen, but sampling plus a shared prefix
                # can get here); cut the rollout short rather than free-run.
                output[row, self.eos_token_id] = 0.0
                continue

            for seq in live:
                if generated < len(seq):
                    next_token = seq[generated]
                    output[row, next_token] = scores[row, next_token]

        return output


def route_token_ids(tokenizer, labels: Sequence[str] = ROUTE_LABELS) -> list[int]:
    """Map route letters to single vocabulary ids, verifying they stay single-token.

    Most BPE vocabularies encode a bare capital letter as one token, but the
    leading-space variant (``" A"``) is a *different* id, and some tokenizers
    split unusual letters. We check and fail loudly, because a multi-token
    action label would silently break the single-token constraint above.
    """
    ids: list[int] = []
    for label in labels:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"route label {label!r} encodes to {len(encoded)} tokens under "
                f"{tokenizer.name_or_path}; use AllowedSequencesProcessor instead"
            )
        ids.append(encoded[0])

    if len(set(ids)) != len(ids):
        raise ValueError("route labels collide in this tokenizer's vocabulary")
    return ids


def build_selector_logits_processor(tokenizer, allowed_token_ids: Sequence[int]) -> LogitsProcessorList:
    """Fresh processor list for one ``generate`` call. Never cache this."""
    return LogitsProcessorList(
        [SingleTokenChoiceProcessor(allowed_token_ids, tokenizer.eos_token_id)]
    )


def patch_generate_with_constraint(model, tokenizer, allowed_token_ids: Sequence[int]):
    """Wrap ``model.generate`` so TRL's internal rollouts are constrained too.

    ``GRPOTrainer`` calls ``generate`` itself, and there is no hook to inject a
    logits processor per call -- so we wrap the bound method. The wrapper builds
    a new processor every call (see the class docstring) and defers to any
    processor the caller passed explicitly.
    """
    original_generate = model.generate

    def constrained_generate(*args, **kwargs):
        kwargs.setdefault(
            "logits_processor",
            build_selector_logits_processor(tokenizer, allowed_token_ids),
        )
        return original_generate(*args, **kwargs)

    model.generate = constrained_generate
    return model


__all__ = [
    "AllowedSequencesProcessor",
    "SingleTokenChoiceProcessor",
    "build_selector_logits_processor",
    "patch_generate_with_constraint",
    "route_token_ids",
]
