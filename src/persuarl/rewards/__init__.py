"""The composite persuasion reward: R1-R5 plus the selector-shaping penalties."""

from .composite import (
    REWARD_NAMES,
    PersuasiveRewardModel,
    RewardBreakdown,
    RewardWeights,
    build_reward_model,
)
from .consistency import PrototypeScorer
from .contextual import ContextualScorer, jaccard_distance, non_repetitiveness_rewards
from .judge import ConstantJudge, JudgeScorer, parse_judge_score
from .penalties import (
    PenaltyBreakdown,
    PenaltyConfig,
    RoutingStatistics,
    compute_penalties,
    penalty_config_from,
)

__all__ = [
    "REWARD_NAMES",
    "ConstantJudge",
    "ContextualScorer",
    "JudgeScorer",
    "PenaltyBreakdown",
    "PenaltyConfig",
    "PersuasiveRewardModel",
    "PrototypeScorer",
    "RewardBreakdown",
    "RewardWeights",
    "RoutingStatistics",
    "build_reward_model",
    "compute_penalties",
    "jaccard_distance",
    "non_repetitiveness_rewards",
    "parse_judge_score",
    "penalty_config_from",
]
