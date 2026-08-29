"""Loading InsureDial and stitching the expert-output CSVs onto it.

The on-disk layout is::

    data/insuredial/
        dialogues.csv                  conversation_id, turn_no, user_utterance, new_agent_reply
        expert_outputs/engagement.csv  conversation_id, turn_no, utterance, ..., engagement_answer
        expert_outputs/intent.csv      ... intent_answer
        expert_outputs/keyterm.csv     ... keyterm_answer
        expert_outputs/sentiment.csv   ... sentiment_answer

The expert files were exported at different times with slightly different
headers, so :func:`load_expert_outputs` normalises them and
:func:`merge_expert_outputs` joins on ``(conversation_id, turn_no)``.

    A note on the original scripts: they concatenated the expert frames
    positionally with ``pd.concat(axis=1)``, which is only correct while every
    file has identical row order. We key-join instead -- same result on the
    shipped data, but it fails loudly instead of silently misaligning if you
    regenerate one expert's outputs on a subset of turns.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..constants import (
    COL_AGENT_REPLY,
    COL_CONVERSATION_ID,
    COL_EXPERT_UTTERANCE,
    COL_TURN_NO,
    COL_USER_UTTERANCE,
    EXPERT_ANSWER_COLUMN,
    EXPERT_KEYS,
)
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class DialogueTurn:
    """One (user utterance -> agent reply) pair with its rolling context.

    ``history`` is the transcript *before* this turn, built from ground-truth
    agent replies during training (teacher forcing) and from generated replies
    during inference. ``previous_agent_reply`` is the immediately preceding
    agent turn, used by the non-repetitiveness reward R4.
    """

    conversation_id: str
    turn_no: int
    user_utterance: str
    agent_reply: str
    history: str = ""
    previous_agent_reply: str = ""
    expert_answers: dict[str, str] = field(default_factory=dict)

    def expert_answer(self, expert: str) -> str:
        return self.expert_answers.get(expert, "")


def load_dialogues(path: str | Path) -> pd.DataFrame:
    """Read the main dialogue CSV and sort it into conversation/turn order."""
    frame = pd.read_csv(path)
    missing = {COL_CONVERSATION_ID, COL_TURN_NO, COL_USER_UTTERANCE} - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing required column(s) {sorted(missing)}")

    frame[COL_TURN_NO] = pd.to_numeric(frame[COL_TURN_NO])
    if COL_AGENT_REPLY not in frame.columns:
        # Out-of-domain eval sets (e.g. DEAL) may ship without reference replies.
        LOGGER.warning("%s has no %r column; reference-based rewards will be 0", path, COL_AGENT_REPLY)
        frame[COL_AGENT_REPLY] = ""

    frame[COL_USER_UTTERANCE] = frame[COL_USER_UTTERANCE].fillna("").astype(str)
    frame[COL_AGENT_REPLY] = frame[COL_AGENT_REPLY].fillna("").astype(str)
    frame = frame.sort_values([COL_CONVERSATION_ID, COL_TURN_NO]).reset_index(drop=True)

    LOGGER.info(
        "loaded %s: %d turns across %d conversations",
        path, len(frame), frame[COL_CONVERSATION_ID].nunique(),
    )
    return frame


def load_expert_outputs(expert: str, path: str | Path) -> pd.DataFrame:
    """Read one expert's precomputed outputs, normalised to three columns.

    Returns ``[conversation_id, turn_no, <expert>_answer]``. If the answer column
    is named something else but is unambiguous, we rename it and log the fact.
    """
    frame = pd.read_csv(path)
    answer_column = EXPERT_ANSWER_COLUMN.format(expert=expert)

    if answer_column not in frame.columns:
        ignorable = {
            "unnamed: 0", "index", COL_CONVERSATION_ID, COL_TURN_NO,
            "speaker", COL_EXPERT_UTTERANCE, COL_USER_UTTERANCE, COL_AGENT_REPLY,
        }
        candidates = [c for c in frame.columns if c.lower() not in ignorable]
        if len(candidates) == 1:
            LOGGER.info("%s: renaming %r -> %r", path, candidates[0], answer_column)
            frame = frame.rename(columns={candidates[0]: answer_column})
        else:
            raise ValueError(
                f"{path}: could not find {answer_column!r}; ambiguous candidates {candidates}. "
                f"Rename the column explicitly and re-run."
            )

    for required in (COL_CONVERSATION_ID, COL_TURN_NO):
        if required not in frame.columns:
            raise ValueError(f"{path}: missing join key {required!r}")

    frame[COL_CONVERSATION_ID] = pd.to_numeric(frame[COL_CONVERSATION_ID])
    frame[COL_TURN_NO] = pd.to_numeric(frame[COL_TURN_NO])
    frame[answer_column] = frame[answer_column].fillna("").astype(str)

    # Duplicated (conversation, turn) keys would fan the join out; keep the first.
    keys = [COL_CONVERSATION_ID, COL_TURN_NO]
    duplicates = int(frame.duplicated(subset=keys).sum())
    if duplicates:
        LOGGER.warning("%s: dropping %d duplicate (conversation, turn) rows", path, duplicates)
        frame = frame.drop_duplicates(subset=keys, keep="first")

    return frame[keys + [answer_column]]


def merge_expert_outputs(
    dialogues: pd.DataFrame,
    expert_paths: Mapping[str, str | Path],
    *,
    how: str = "inner",
) -> pd.DataFrame:
    """Left-to-right join every expert's answers onto the dialogue frame.

    ``how="inner"`` (the default, and what training uses) keeps only turns that
    every expert covered, so a route can never reference a missing answer. Use
    ``how="left"`` to keep uncovered turns with empty answers instead.
    """
    merged = dialogues.copy()
    merged[COL_CONVERSATION_ID] = pd.to_numeric(merged[COL_CONVERSATION_ID])

    for expert, path in expert_paths.items():
        before = len(merged)
        expert_frame = load_expert_outputs(expert, path)
        merged = merged.merge(expert_frame, on=[COL_CONVERSATION_ID, COL_TURN_NO], how=how)
        LOGGER.info("merged %-10s %6d -> %6d turns", expert, before, len(merged))

    if merged.empty:
        raise ValueError(
            "expert merge produced zero rows -- the CSVs do not share "
            "(conversation_id, turn_no) keys. Check that they were generated "
            "from the same dialogue file."
        )

    for expert in expert_paths:
        column = EXPERT_ANSWER_COLUMN.format(expert=expert)
        merged[column] = merged[column].fillna("").astype(str)

    return merged.sort_values([COL_CONVERSATION_ID, COL_TURN_NO]).reset_index(drop=True)


def iter_dialogue_turns(
    frame: pd.DataFrame,
    *,
    experts: Sequence[str] = EXPERT_KEYS,
    include_history: bool = True,
) -> Iterator[DialogueTurn]:
    """Yield :class:`DialogueTurn` objects with history accumulated per conversation.

    History is teacher-forced from ground-truth replies: at turn *t* the model
    sees the reference transcript up to *t-1*, never its own earlier outputs.
    That keeps training examples independent of decoding order, which is what
    makes GRPO's group-relative advantages comparable within a batch.
    """
    for conversation_id, group in frame.groupby(COL_CONVERSATION_ID, sort=True):
        history_parts: list[str] = []
        previous_reply = ""

        for _, row in group.sort_values(COL_TURN_NO).iterrows():
            user_utterance = str(row[COL_USER_UTTERANCE])
            agent_reply = str(row.get(COL_AGENT_REPLY, ""))

            yield DialogueTurn(
                conversation_id=str(conversation_id),
                turn_no=int(row[COL_TURN_NO]),
                user_utterance=user_utterance,
                agent_reply=agent_reply,
                history="\n".join(history_parts) if include_history else "",
                previous_agent_reply=previous_reply,
                expert_answers={
                    expert: str(row.get(EXPERT_ANSWER_COLUMN.format(expert=expert), ""))
                    for expert in experts
                },
            )

            history_parts.append(f"User: {user_utterance}")
            history_parts.append(f"Agent: {agent_reply}")
            previous_reply = agent_reply
