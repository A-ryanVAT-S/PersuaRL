"""Turning :class:`DialogueTurn` objects into model inputs.

Two things happen here and nowhere else:

1. **Prompt construction** -- the Selector prompt, the Generator prompt (with
   its ``<expert>...</expert>`` analysis block) and the plain baseline prompt.
2. **Prompt masking** -- tokenising ``prompt + completion`` and setting the
   prompt's labels to ``IGNORE_INDEX`` so SFT only ever backprops through the
   agent's reply. Every SFT variant in this repo reuses
   :func:`tokenize_with_prompt_mask`; getting the mask wrong in one copy of the
   code and not another was the single easiest way to make two runs
   incomparable in the original scripts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ..constants import EXPERT_KEYS, IGNORE_INDEX
from ..routes import Route
from .dataset import DialogueTurn
from .prompts import (
    BASELINE_SYSTEM_PROMPT,
    GENERATOR_SYSTEM_PROMPT,
    SELECTOR_ANSWER_CUE,
    SELECTOR_SYSTEM_PROMPT,
)

# --------------------------------------------------------------------------
# Selector
# --------------------------------------------------------------------------


def build_selector_prompt(turn: DialogueTurn) -> str:
    """Raw-text prompt asking the Selector for one route letter.

    Deliberately *not* chat-templated. The Selector emits a single constrained
    token; a chat template would add role scaffolding that the constrained
    logits processor then has to fight, and the extra tokens cost throughput on
    every one of the ``num_generations`` rollouts per turn.
    """
    parts = [SELECTOR_SYSTEM_PROMPT, "\n\n"]
    if turn.history:
        parts.append(f"### Conversation History:\n{turn.history.strip()}\n\n")
    parts.append(f"### Current User Utterance:\n{turn.user_utterance}\n\n")
    parts.append(SELECTOR_ANSWER_CUE)
    return "".join(parts)


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------


def format_internal_analysis(
    expert_answers: Mapping[str, str],
    experts: Iterable[str],
) -> str:
    """Render the selected experts' outputs as an XML-ish analysis block.

    Tag names come from the expert keys, so the Generator can tell an intent
    label from a sentiment label without positional guessing::

        <engagement>The persuasion strategy is Logical appeal ...</engagement>
        <keyterm>The keyterm extracted are Roadside assistance</keyterm>
    """
    blocks = [
        f"<{expert}>{str(expert_answers.get(expert, '')).strip()}</{expert}>"
        for expert in experts
    ]
    return "\n".join(blocks)


def build_generator_messages(
    turn: DialogueTurn,
    experts: Sequence[str],
    *,
    system_prompt: str = GENERATOR_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """Chat messages for the Generator, conditioned on the selected experts only."""
    analysis = format_internal_analysis(turn.expert_answers, experts)
    user_content = (
        f"### Conversation History:\n{turn.history.strip() or 'N/A'}\n\n"
        f"### Current User Utterance:\n{turn.user_utterance}\n\n"
        f"### Internal Analysis:\n{analysis or 'No analysis available.'}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def build_generator_messages_for_route(
    turn: DialogueTurn,
    route: Route,
    **kwargs: Any,
) -> list[dict[str, str]]:
    """Convenience wrapper: expand a :class:`Route` into its expert list."""
    return build_generator_messages(turn, route.experts, **kwargs)


# --------------------------------------------------------------------------
# Baselines (single-shot and SFT, no expert conditioning)
# --------------------------------------------------------------------------


def build_baseline_messages(
    turn: DialogueTurn,
    *,
    system_prompt: str = BASELINE_SYSTEM_PROMPT,
    include_history: bool = True,
) -> list[dict[str, str]]:
    """Chat messages for the single-shot / SFT baselines described in C.1."""
    if include_history:
        user_content = (
            f"Conversation History:\n{turn.history.strip()}\n\n"
            f"Current User Utterance:\n{turn.user_utterance}"
        )
    else:
        user_content = turn.user_utterance
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


# --------------------------------------------------------------------------
# Tokenisation with prompt masking
# --------------------------------------------------------------------------


def render_prompt(tokenizer, messages: Sequence[Mapping[str, str]]) -> str:
    """Apply the backbone's chat template with a generation cue appended.

    Every supported backbone (Llama 3.x, Qwen 2.5, Phi-3, Mistral) ships a
    template, so we never hand-roll role markers -- that is what made the
    original per-model scripts diverge.
    """
    return tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
    )


def tokenize_with_prompt_mask(
    tokenizer,
    messages: Sequence[Mapping[str, str]],
    completion: str,
    *,
    max_length: int = 1536,
) -> dict[str, list[int]]:
    """Tokenise ``prompt + completion``, masking the prompt out of the labels.

    Returns ``input_ids`` / ``attention_mask`` / ``labels`` ready for
    ``DataCollatorForSeq2Seq``. Truncation is applied to the concatenation, so a
    very long history eats into the completion rather than shifting the mask
    boundary -- we re-tokenise the prompt separately to find that boundary and
    clamp it to the truncated length.
    """
    prompt_text = render_prompt(tokenizer, messages)
    eos = tokenizer.eos_token or ""
    full_text = prompt_text + str(completion) + eos

    encoded_full = tokenizer(full_text, truncation=True, max_length=max_length, padding=False)
    encoded_prompt = tokenizer(prompt_text, truncation=True, max_length=max_length, padding=False)

    # min(): if truncation cut into the prompt there is no completion left to
    # learn from, and an unclamped slice would mask past the end of the sequence.
    prompt_length = min(len(encoded_prompt["input_ids"]), len(encoded_full["input_ids"]))

    labels = list(encoded_full["input_ids"])
    for index in range(prompt_length):
        labels[index] = IGNORE_INDEX

    return {
        "input_ids": encoded_full["input_ids"],
        "attention_mask": encoded_full["attention_mask"],
        "labels": labels,
    }


def has_trainable_tokens(example: Mapping[str, Sequence[int]]) -> bool:
    """True if at least one label survived masking.

    Fully-masked examples contribute a NaN loss under some collators; we drop
    them at dataset build time instead of guarding every training step.
    """
    return any(label != IGNORE_INDEX for label in example["labels"])


__all__ = [
    "build_selector_prompt",
    "build_generator_messages",
    "build_generator_messages_for_route",
    "build_baseline_messages",
    "format_internal_analysis",
    "render_prompt",
    "tokenize_with_prompt_mask",
    "has_trainable_tokens",
    "EXPERT_KEYS",
]
