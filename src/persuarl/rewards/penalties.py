"""Selector-shaping penalties (Appendix C.5).

The five ``R_k`` terms score the *response*. These three terms score the
*routing decision*, and they are what stop the Selector from degenerating:

Complexity penalty
    ``alpha * N``. Linear in the number of activated experts. Without it the
    policy learns that "select everything" is weakly dominant -- more context
    rarely hurts a single response, so the reward alone never pays for
    restraint. This is what makes PersuaRL a *selector* rather than AllExpert.

Route repetition penalty
    Compares how often a route has been chosen against uniform usage and
    penalises overuse. This is exploration pressure: GRPO's advantages are
    computed within a group, so once the policy collapses onto one route every
    rollout in a group scores identically and the gradient vanishes.

Load-balance penalty (and its mirrored bonus)
    Per-expert version of the same idea, measured as an expert's usage relative
    to the mean usage of the *other* experts. Keeps all four experts
    contributing instead of letting the Generator over-fit to, say, engagement
    signals alone.

All three read from shared counters that live across GRPO steps -- see
:class:`RoutingStatistics`, which is deliberately backed by
``multiprocessing.Manager`` objects so dataloader workers observe the same
history.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, MutableMapping, MutableSequence
from dataclasses import dataclass

from ..constants import EXPERT_KEYS
from ..routes import NUM_ROUTES, Route

_EPS = 1e-6


@dataclass
class PenaltyConfig:
    """Coefficients from Appendix C.5. Defaults reproduce the paper."""

    complexity_alpha: float = 0.025
    """``alpha``: reward cost per activated expert."""

    repetition_beta: float = 0.2
    """``beta``: scaling on (usage ratio - 1) for the route repetition penalty."""

    repetition_max: float = 0.15
    """``P_max``: cap on the route repetition penalty."""

    repetition_warmup: int = 16
    """Rollouts to observe before repetition is penalised at all.

    Early in training every route looks 'overused' relative to a tiny history,
    which would penalise the policy for noise.
    """

    load_balance_gamma: float = 0.4
    """``gamma``: strength of the per-expert overuse penalty."""

    load_balance_max: float = 0.15
    """Cap on the load-balance penalty."""

    load_balance_bonus_gamma: float = 0.05
    """Strength of the mirrored bonus for *under*-used experts."""

    load_balance_bonus_max: float = 0.08
    """Cap on the load-balance bonus."""

    history_size: int = 200
    """Sliding window of recent routes used for the repetition ratio."""


@dataclass
class PenaltyBreakdown:
    """Per-rollout penalty detail, surfaced in the training log."""

    complexity: float = 0.0
    repetition: float = 0.0
    load_balance: float = 0.0
    load_bonus: float = 0.0

    @property
    def total(self) -> float:
        """Net adjustment to subtract from the base reward."""
        return self.complexity + self.repetition + self.load_balance - self.load_bonus


class RoutingStatistics:
    """Usage counters shared across GRPO steps (and dataloader workers).

    ``expert_counts`` and ``recent_routes`` may be plain dict/list or
    ``multiprocessing.Manager`` proxies. Proxy containers do not support
    in-place mutation of nested values, which is why every update here
    reassigns rather than mutating.
    """

    def __init__(
        self,
        expert_counts: MutableMapping[str, int] | None = None,
        recent_routes: MutableSequence[str] | None = None,
        *,
        history_size: int = 200,
    ) -> None:
        self.expert_counts = expert_counts if expert_counts is not None else {}
        for key in EXPERT_KEYS:
            if key not in self.expert_counts:
                self.expert_counts[key] = 0
        self.recent_routes = recent_routes if recent_routes is not None else []
        self.history_size = history_size

    @classmethod
    def shared(cls, manager, *, history_size: int = 200) -> RoutingStatistics:
        """Build an instance backed by ``multiprocessing.Manager`` proxies."""
        counts = manager.dict({key: 0 for key in EXPERT_KEYS})
        return cls(counts, manager.list(), history_size=history_size)

    def snapshot(self) -> tuple[Counter, Counter]:
        """Copy the counters once per batch.

        Penalties for a batch are computed against the state *before* the batch,
        plus the batch's own running updates -- reading the proxies once keeps
        the whole batch consistent and avoids a proxy round-trip per rollout.
        """
        return Counter(list(self.recent_routes)), Counter(dict(self.expert_counts))

    def commit(self, routes: list[Route]) -> None:
        """Fold a batch's accepted routes into the shared history."""
        if not routes:
            return

        history = list(self.recent_routes) + [route.as_json() for route in routes]
        self.recent_routes[:] = history[-self.history_size:]

        for route in routes:
            for expert in route.experts:
                self.expert_counts[expert] = self.expert_counts.get(expert, 0) + 1

    def usage_summary(self) -> dict[str, int]:
        return dict(self.expert_counts)


def complexity_penalty(route: Route, config: PenaltyConfig) -> float:
    """``alpha * N`` -- see Appendix C.5.1."""
    return config.complexity_alpha * route.size


def repetition_penalty(
    route: Route,
    route_counts: Mapping[str, int],
    observed: int,
    config: PenaltyConfig,
) -> float:
    """``min(beta * max(0, F - 1), P_max)`` where ``F`` is usage over ideal usage."""
    if observed < config.repetition_warmup:
        return 0.0

    ideal = observed / NUM_ROUTES
    # +1 counts the rollout being scored, so a route's first use is not free.
    ratio = (route_counts.get(route.as_json(), 0) + 1) / (ideal + _EPS)
    return min(max(ratio - 1.0, 0.0) * config.repetition_beta, config.repetition_max)


def load_balance_terms(
    route: Route,
    expert_counts: Mapping[str, int],
    config: PenaltyConfig,
) -> tuple[float, float]:
    """Per-expert overuse penalty and underuse bonus, averaged over the route.

    For expert ``k``, ``R_k`` is its usage divided by the mean usage of the
    other experts. ``R_k > 1`` (overused) contributes ``gamma * (R_k - 1)^2``
    to the penalty; ``R_k < 1`` contributes ``gamma_bonus * (1 - R_k)^2`` to the
    bonus. Both are averaged over the route's experts and then capped, so a
    4-expert route is not penalised four times over for one imbalanced expert.
    """
    if route.size == 0:
        return 0.0, 0.0

    total_usage = float(sum(expert_counts.get(key, 0) for key in EXPERT_KEYS))
    others = max(len(EXPERT_KEYS) - 1, 1)

    penalty = 0.0
    bonus = 0.0
    for expert in route.experts:
        own = float(expert_counts.get(expert, 0))
        others_mean = (total_usage - own) / others + _EPS
        ratio = own / others_mean
        if ratio > 1.0:
            penalty += config.load_balance_gamma * (ratio - 1.0) ** 2
        else:
            bonus += config.load_balance_bonus_gamma * (1.0 - ratio) ** 2

    return (
        min(penalty / route.size, config.load_balance_max),
        min(bonus / route.size, config.load_balance_bonus_max),
    )


def compute_penalties(
    route: Route | None,
    route_counts: Counter,
    expert_counts: Counter,
    config: PenaltyConfig,
) -> PenaltyBreakdown:
    """All three penalty terms for one rollout.

    An unparseable route (``None``) carries no penalties -- it already gets a
    base reward of 0, and stacking penalties on top would push the clipped
    reward to the same floor while muddying the log.
    """
    if route is None:
        return PenaltyBreakdown()

    observed = sum(route_counts.values())
    load_penalty, load_bonus = load_balance_terms(route, expert_counts, config)

    return PenaltyBreakdown(
        complexity=complexity_penalty(route, config),
        repetition=repetition_penalty(route, route_counts, observed, config),
        load_balance=load_penalty,
        load_bonus=load_bonus,
    )


def penalty_config_from(section) -> PenaltyConfig:
    """Build a :class:`PenaltyConfig` from a ``rewards.penalties`` config block."""
    return PenaltyConfig(
        complexity_alpha=float(section.get("complexity_alpha", 0.025)),
        repetition_beta=float(section.get("repetition_beta", 0.2)),
        repetition_max=float(section.get("repetition_max", 0.15)),
        repetition_warmup=int(section.get("repetition_warmup", 16)),
        load_balance_gamma=float(section.get("load_balance_gamma", 0.4)),
        load_balance_max=float(section.get("load_balance_max", 0.15)),
        load_balance_bonus_gamma=float(section.get("load_balance_bonus_gamma", 0.05)),
        load_balance_bonus_max=float(section.get("load_balance_bonus_max", 0.08)),
        history_size=int(section.get("history_size", 200)),
    )


__all__ = [
    "PenaltyBreakdown",
    "PenaltyConfig",
    "RoutingStatistics",
    "complexity_penalty",
    "compute_penalties",
    "load_balance_terms",
    "penalty_config_from",
    "repetition_penalty",
]
