"""Data loading, conversation-level splitting and prompt construction.

The split tests matter most: a row-level split would leak test conversations
into training and quietly inflate every lexical metric in the results table.
"""

from __future__ import annotations

import pandas as pd
import pytest

from persuarl.constants import EXPERT_ANSWER_COLUMN, EXPERT_KEYS, IGNORE_INDEX
from persuarl.data.dataset import iter_dialogue_turns, load_expert_outputs, merge_expert_outputs
from persuarl.data.formatting import build_selector_prompt, format_internal_analysis
from persuarl.data.splits import split_by_conversation
from persuarl.experts.registry import parse_answer
from persuarl.routes import ROUTE_LABELS


@pytest.fixture
def dialogues() -> pd.DataFrame:
    """Six turns across three conversations."""
    return pd.DataFrame(
        {
            "conversation_id": [1, 1, 2, 2, 3, 3],
            "turn_no": [1, 3, 1, 3, 1, 3],
            "user_utterance": [f"user turn {i}" for i in range(6)],
            "new_agent_reply": [f"agent turn {i}" for i in range(6)],
        }
    )


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------


def test_split_never_puts_a_conversation_in_two_places(dialogues):
    split = split_by_conversation(dialogues, train_ratio=0.34, validation_ratio=0.33)
    train, validation, test = (set(split.train_ids), set(split.validation_ids), set(split.test_ids))

    assert not (train & validation) and not (train & test) and not (validation & test)
    assert train | validation | test == set(dialogues["conversation_id"])


def test_split_is_deterministic_for_a_given_seed(dialogues):
    first = split_by_conversation(dialogues, seed=7)
    second = split_by_conversation(dialogues, seed=7)
    assert list(first.train_ids) == list(second.train_ids)


def test_split_does_not_disturb_the_global_numpy_rng(dialogues):
    """We use a local Generator so callers' RNG state is untouched."""
    import numpy as np

    np.random.seed(0)
    expected = np.random.rand()

    np.random.seed(0)
    split_by_conversation(dialogues, seed=123)
    assert np.random.rand() == expected


def test_zero_validation_ratio_gives_an_85_15_style_split(dialogues):
    split = split_by_conversation(dialogues, train_ratio=0.67, validation_ratio=0.0)
    assert len(split.validation_ids) == 0
    assert len(split.test_ids) > 0


def test_split_rejects_ratios_that_leave_no_test_set(dialogues):
    with pytest.raises(ValueError):
        split_by_conversation(dialogues, train_ratio=0.9, validation_ratio=0.2)


# --------------------------------------------------------------------------
# Turn iteration
# --------------------------------------------------------------------------


def test_history_accumulates_within_a_conversation_and_resets_between(dialogues):
    turns = list(iter_dialogue_turns(dialogues, experts=()))

    first_of_conversation = [turn for turn in turns if turn.turn_no == 1]
    assert all(turn.history == "" for turn in first_of_conversation)

    second = next(turn for turn in turns if turn.conversation_id == "1" and turn.turn_no == 3)
    assert "user turn 0" in second.history and "agent turn 0" in second.history
    assert second.previous_agent_reply == "agent turn 0"


def test_history_is_teacher_forced_from_reference_replies(dialogues):
    """Training context must not depend on decoding order."""
    turns = list(iter_dialogue_turns(dialogues, experts=()))
    second = next(t for t in turns if t.conversation_id == "2" and t.turn_no == 3)
    assert second.history.endswith("Agent: agent turn 2")


# --------------------------------------------------------------------------
# Expert merging
# --------------------------------------------------------------------------


def test_merge_joins_on_keys_not_row_order(dialogues):
    """Shuffled expert rows must still land on the right turn."""
    column = EXPERT_ANSWER_COLUMN.format(expert="sentiment")
    answers = pd.DataFrame(
        {
            "conversation_id": [2, 1, 3, 1, 3, 2],
            "turn_no": [3, 1, 1, 3, 3, 1],
            column: [f"answer-{i}" for i in [3, 0, 4, 1, 5, 2]],
        }
    )
    merged = merge_expert_outputs(dialogues, {"sentiment": _write(answers)})

    row = merged[(merged["conversation_id"] == 1) & (merged["turn_no"] == 1)].iloc[0]
    assert row[column] == "answer-0"


def test_merge_fails_loudly_when_keys_do_not_overlap(dialogues):
    column = EXPERT_ANSWER_COLUMN.format(expert="sentiment")
    answers = pd.DataFrame({"conversation_id": [99], "turn_no": [1], column: ["x"]})
    with pytest.raises(ValueError, match="zero rows"):
        merge_expert_outputs(dialogues, {"sentiment": _write(answers)})


def test_unambiguous_answer_column_is_renamed(tmp_path):
    path = tmp_path / "keyterm.csv"
    pd.DataFrame(
        {"conversation_id": [1], "turn_no": [1], "utterance": ["hi"], "the_answer": ["terms"]}
    ).to_csv(path, index=False)

    frame = load_expert_outputs("keyterm", path)
    assert EXPERT_ANSWER_COLUMN.format(expert="keyterm") in frame.columns


_TMP: list = []


def _write(frame: pd.DataFrame) -> str:
    """Persist a frame to a temp CSV and keep the handle alive for the test."""
    import tempfile

    handle = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="")
    frame.to_csv(handle.name, index=False)
    handle.close()
    _TMP.append(handle.name)
    return handle.name


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------


def test_selector_prompt_lists_every_action(dialogues):
    turn = next(iter_dialogue_turns(dialogues, experts=()))
    prompt = build_selector_prompt(turn)
    for label in ROUTE_LABELS:
        assert f"- {label}:" in prompt
    assert turn.user_utterance in prompt


def test_selector_prompt_omits_an_empty_history(dialogues):
    turn = next(iter_dialogue_turns(dialogues, experts=()))
    assert "### Conversation History" not in build_selector_prompt(turn)


def test_analysis_block_is_tagged_per_expert():
    analysis = format_internal_analysis(
        {"intent": "Ask_Coverage_Details", "sentiment": "positive"},
        ["intent", "sentiment"],
    )
    assert "<intent>Ask_Coverage_Details</intent>" in analysis
    assert "<sentiment>positive</sentiment>" in analysis


def test_analysis_block_includes_only_the_selected_experts():
    analysis = format_internal_analysis(
        {key: f"{key}-answer" for key in EXPERT_KEYS}, ["keyterm"]
    )
    assert "keyterm-answer" in analysis
    assert "engagement-answer" not in analysis


# --------------------------------------------------------------------------
# Expert answer parsing (the shipped CSVs use prose, not Label:/Reason:)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expert,text,label",
    [
        ("engagement", "The persuasion strategy is Logical appeal and the reason is it cites facts.", "Logical appeal"),
        ("engagement", "Label: Emotional appeal\nReason: reassures the user.", "Emotional appeal"),
        ("sentiment", "The sentiment is negative", "negative"),
        ("keyterm", "The keyterm extracted are Roadside assistance", "Roadside assistance"),
        ("intent", "Label: Confirm_Interest\nReason: the user agrees.", "Confirm_Interest"),
    ],
)
def test_expert_answers_parse_in_both_formats(expert, text, label):
    parsed = parse_answer(expert, text)
    assert parsed is not None and parsed["label"] == label


def test_unparseable_expert_answer_returns_none():
    assert parse_answer("sentiment", "???") is None


# --------------------------------------------------------------------------
# Prompt masking
# --------------------------------------------------------------------------


class _StubTokenizer:
    """Whitespace tokenizer with a chat template -- no downloads in unit tests."""

    eos_token = "</s>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return " ".join(m["content"] for m in messages) + " ASSISTANT:"

    def __call__(self, text, truncation=False, max_length=None, padding=False):
        ids = [hash(token) % 1000 for token in text.split()]
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_prompt_tokens_are_masked_out_of_the_labels():
    from persuarl.data.formatting import tokenize_with_prompt_mask

    tokenizer = _StubTokenizer()
    encoded = tokenize_with_prompt_mask(
        tokenizer,
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello there"}],
        "the completion",
    )

    labels = encoded["labels"]
    assert len(labels) == len(encoded["input_ids"])
    # Everything before the completion is masked; the completion is not.
    assert labels[0] == IGNORE_INDEX
    assert labels[-1] != IGNORE_INDEX


def test_masking_is_clamped_when_truncation_eats_the_completion():
    from persuarl.data.formatting import has_trainable_tokens, tokenize_with_prompt_mask

    tokenizer = _StubTokenizer()
    encoded = tokenize_with_prompt_mask(
        tokenizer,
        [{"role": "user", "content": "a b c d e f g h"}],
        "completion",
        max_length=3,
    )

    assert len(encoded["labels"]) == len(encoded["input_ids"]) == 3
    # Fully masked -> the example is droppable rather than a NaN loss.
    assert not has_trainable_tokens(encoded)
