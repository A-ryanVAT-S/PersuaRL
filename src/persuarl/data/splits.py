"""Conversation-level splitting.

The one rule this file exists to enforce: **split on ``conversation_id``, never
on rows.** Turns inside a dialogue share personas, vehicle details and phrasing,
so a row-level split leaks the test set into training and inflates every
lexical metric. The paper's 80/5/15 split is conversation-level, and so is the
85/15 split used by the RL scripts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..constants import COL_CONVERSATION_ID, DEFAULT_SEED


@dataclass(frozen=True)
class ConversationSplit:
    """Three disjoint frames plus the conversation ids that produced them."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_ids: np.ndarray
    validation_ids: np.ndarray
    test_ids: np.ndarray

    def describe(self) -> str:
        return (
            f"train={len(self.train)} turns / {len(self.train_ids)} convs | "
            f"val={len(self.validation)} turns / {len(self.validation_ids)} convs | "
            f"test={len(self.test)} turns / {len(self.test_ids)} convs"
        )


def split_by_conversation(
    frame: pd.DataFrame,
    *,
    train_ratio: float = 0.80,
    validation_ratio: float = 0.05,
    seed: int = DEFAULT_SEED,
    id_column: str = COL_CONVERSATION_ID,
) -> ConversationSplit:
    """Shuffle conversation ids once, then slice train/val/test from them.

    Whatever is left after ``train_ratio + validation_ratio`` becomes the test
    split, so ``0.85 / 0.0`` reproduces the 85/15 split the RL scripts use and
    ``0.80 / 0.05`` reproduces the paper's Table 1.
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if not 0 <= validation_ratio < 1 or train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must leave room for a test split")

    ids = frame[id_column].unique().copy()
    # A local Generator, not np.random.seed(), so callers can't be surprised by
    # our reseeding of the global RNG halfway through their script.
    np.random.default_rng(seed).shuffle(ids)

    train_end = int(len(ids) * train_ratio)
    val_end = train_end + int(len(ids) * validation_ratio)
    train_ids, val_ids, test_ids = ids[:train_end], ids[train_end:val_end], ids[val_end:]

    def subset(subset_ids: np.ndarray) -> pd.DataFrame:
        return frame[frame[id_column].isin(subset_ids)].copy()

    return ConversationSplit(
        train=subset(train_ids),
        validation=subset(val_ids),
        test=subset(test_ids),
        train_ids=train_ids,
        validation_ids=val_ids,
        test_ids=test_ids,
    )
