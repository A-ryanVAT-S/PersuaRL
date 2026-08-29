"""Reward arithmetic and penalty shaping.

The expensive parts (BERTScore, the 7B judge, the BERT classifiers) are not
exercised here -- these tests cover the composition logic around them, which is
where an off-by-one in a weight or a cap actually changes results.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from persuarl.constants import EXPERT_KEYS
from persuarl.rewards.composite import REWARD_NAMES, PersuasiveRewardModel, RewardWeights
from persuarl.rewards.contextual import jaccard_distance, non_repetitiveness_rewards
from persuarl.rewards.judge import normalize_score, parse_judge_score
from persuarl.rewards.penalties import (
    PenaltyConfig,
    RoutingStatistics,
    complexity_penalty,
    compute_penalties,
    load_balance_terms,
    repetition_penalty,
)
from persuarl.routes import ROUTES, route_from_mask

# --------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------


def test_default_weights_match_the_paper():
    """Table 11's tuned setting: 0.15 / 0.15 / 0.20 / 0.15 / 0.35."""
    weights = RewardWeights()
    assert (weights.engagement, weights.intent, weights.contextual,
            weights.repetition, weights.judge) == (0.15, 0.15, 0.20, 0.15, 0.35)
    assert math.isclose(weights.total(), 1.0)


def test_without_zeroes_a_component_and_does_not_renormalise():
    """The ablation asks what a term contributes, not how a rebalanced objective does."""
    ablated = RewardWeights().without("judge")
    assert ablated.judge == 0.0
    assert math.isclose(ablated.total(), 0.65)


def test_without_rejects_unknown_component():
    with pytest.raises(ValueError, match="unknown reward component"):
        RewardWeights().without("nonexistent")


# --------------------------------------------------------------------------
# R4: non-repetitiveness
# --------------------------------------------------------------------------


def test_identical_replies_score_zero():
    assert jaccard_distance("we cover your battery", "we cover your battery") == 0.0


def test_disjoint_replies_score_one():
    assert jaccard_distance("alpha beta", "gamma delta") == pytest.approx(1.0)


def test_first_turn_has_no_previous_reply():
    """Returning 0 (not 1) keeps turn one from earning free credit."""
    assert jaccard_distance("anything", "") == 0.0


def test_non_repetitiveness_is_batched_elementwise():
    scores = non_repetitiveness_rewards(["a b", "c d"], ["a b", ""])
    assert scores.shape == (2,)
    assert scores[0] == 0.0 and scores[1] == 0.0


# --------------------------------------------------------------------------
# R5: judge parsing
# --------------------------------------------------------------------------


def test_judge_score_is_parsed_from_the_result_tag():
    score, ok = parse_judge_score("Feedback: strong empathy. [RESULT] 4")
    assert (score, ok) == (4, True)


def test_malformed_judge_output_falls_back_to_neutral():
    score, ok = parse_judge_score("the model rambled without a verdict")
    assert (score, ok) == (3, False)


@pytest.mark.parametrize("raw,expected", [(1, 0.0), (3, 0.5), (5, 1.0)])
def test_judge_normalisation_spans_the_unit_interval(raw, expected):
    assert normalize_score(raw) == pytest.approx(expected)


# --------------------------------------------------------------------------
# Penalties
# --------------------------------------------------------------------------


def test_complexity_penalty_is_linear_in_route_size():
    config = PenaltyConfig()
    single = next(route for route in ROUTES if route.size == 1)
    every = route_from_mask({key: 1 for key in EXPERT_KEYS})
    assert complexity_penalty(single, config) == pytest.approx(0.025)
    assert complexity_penalty(every, config) == pytest.approx(0.100)


def test_repetition_penalty_is_inactive_during_warmup():
    """Early on, every route looks overused relative to a tiny history."""
    config = PenaltyConfig()
    route = ROUTES[0]
    counts = {route.as_json(): 5}
    assert repetition_penalty(route, counts, observed=5, config=config) == 0.0


def test_repetition_penalty_grows_with_overuse_and_is_capped():
    config = PenaltyConfig()
    route = ROUTES[0]
    observed = 150

    balanced = repetition_penalty(route, {route.as_json(): observed // 15}, observed, config)
    collapsed = repetition_penalty(route, {route.as_json(): observed}, observed, config)

    assert balanced < collapsed
    assert collapsed == pytest.approx(config.repetition_max)


def test_load_balance_penalises_the_overused_expert():
    config = PenaltyConfig()
    route = route_from_mask({"engagement": 1, "intent": 0, "keyterm": 0, "sentiment": 0})
    skewed = {"engagement": 900, "intent": 10, "keyterm": 10, "sentiment": 10}

    penalty, bonus = load_balance_terms(route, skewed, config)
    assert penalty == pytest.approx(config.load_balance_max)
    assert bonus == 0.0


def test_load_balance_rewards_the_underused_expert():
    config = PenaltyConfig()
    route = route_from_mask({"engagement": 0, "intent": 0, "keyterm": 0, "sentiment": 1})
    skewed = {"engagement": 900, "intent": 300, "keyterm": 300, "sentiment": 1}

    penalty, bonus = load_balance_terms(route, skewed, config)
    assert penalty == 0.0
    assert bonus > 0.0


def test_invalid_route_carries_no_penalties():
    """It already scores 0; stacking penalties would only muddy the log."""
    from collections import Counter

    breakdown = compute_penalties(None, Counter(), Counter(), PenaltyConfig())
    assert breakdown.total == 0.0


def test_penalty_total_nets_the_bonus_out():
    from collections import Counter

    route = ROUTES[-1]
    breakdown = compute_penalties(route, Counter(), Counter(), PenaltyConfig())
    assert breakdown.total == pytest.approx(
        breakdown.complexity + breakdown.repetition + breakdown.load_balance - breakdown.load_bonus
    )


# --------------------------------------------------------------------------
# Routing statistics
# --------------------------------------------------------------------------


def test_statistics_track_expert_usage():
    statistics = RoutingStatistics(history_size=10)
    route = route_from_mask({"engagement": 1, "intent": 1, "keyterm": 0, "sentiment": 0})

    statistics.commit([route, route])
    usage = statistics.usage_summary()

    assert usage["engagement"] == 2 and usage["intent"] == 2
    assert usage["keyterm"] == 0 and usage["sentiment"] == 0


def test_route_history_is_a_sliding_window():
    statistics = RoutingStatistics(history_size=3)
    statistics.commit(list(ROUTES[:5]))
    assert len(list(statistics.recent_routes)) == 3


# --------------------------------------------------------------------------
# Composite
# --------------------------------------------------------------------------


def test_composite_scores_only_the_available_components():
    """With no scorers loaded, only R4 (which needs no model) is non-zero."""
    model = PersuasiveRewardModel(weights=RewardWeights())
    breakdown = model.score(
        ["a completely different reply"],
        user_utterances=["what does it cover?"],
        contexts=[""],
        previous_replies=["some earlier reply"],
        references=["reference"],
    )

    assert breakdown.repetition[0] > 0.0
    assert breakdown.engagement[0] == 0.0
    # Only beta_4 contributes.
    assert breakdown.total[0] == pytest.approx(0.15 * breakdown.repetition[0])


def test_empty_response_scores_zero_everywhere():
    model = PersuasiveRewardModel(weights=RewardWeights())
    breakdown = model.score(
        ["", "a real reply"],
        user_utterances=["q", "q"],
        contexts=["", ""],
        previous_replies=["prior", "prior"],
    )
    assert breakdown.total[0] == 0.0
    assert all(breakdown.component(name)[0] == 0.0 for name in REWARD_NAMES)


def test_breakdown_means_are_reported_for_every_component():
    model = PersuasiveRewardModel(weights=RewardWeights())
    breakdown = model.score(
        ["reply one", "reply two"],
        user_utterances=["a", "b"],
        contexts=["", ""],
        previous_replies=["x", "y"],
    )
    means = breakdown.means()
    assert all(f"reward/{name}" in means for name in REWARD_NAMES)
    assert np.isfinite(means["reward/total"])
