"""Training entry points: SFT, PersuaRL selector GRPO, single-model GRPO."""

from .grpo_selector import SelectorRewardFunction, train_selector
from .grpo_single import train_single_model_grpo
from .sft import train_sft

__all__ = [
    "SelectorRewardFunction",
    "train_selector",
    "train_single_model_grpo",
    "train_sft",
]
