"""The Selector's action space.

The Selector emits a binary mask o_t in {0,1}^n over the n experts. With
n = 4 that is 2^4 = 16 masks, minus the empty one (a turn must consult at
least one expert), so the policy chooses among **15 routes**.

Rather than making the policy emit JSON -- which is slow to decode and easy
to malform -- we assign each route a single capital letter ``A``..``O`` and
constrain generation to exactly one of those tokens (see
:mod:`persuarl.models.decoding`). One forward pass, one token, no parsing
failures.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .constants import EXPERT_KEYS


@dataclass(frozen=True)
class Route:
    """One element of the Selector's action space."""

    label: str
    """Single-character action token, e.g. ``"C"``."""

    mask: tuple[int, ...]
    """Binary activation vector aligned with :data:`~persuarl.constants.EXPERT_KEYS`."""

    @property
    def experts(self) -> tuple[str, ...]:
        """Names of the experts this route activates, in canonical order."""
        return tuple(k for k, bit in zip(EXPERT_KEYS, self.mask) if bit)

    @property
    def size(self) -> int:
        """Number of activated experts -- the ``N`` in the complexity penalty."""
        return sum(self.mask)

    def as_dict(self) -> dict[str, int]:
        return dict(zip(EXPERT_KEYS, self.mask))

    def as_json(self) -> str:
        """Stable key used for route-repetition bookkeeping."""
        return json.dumps(self.as_dict(), separators=(",", ":"))

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.label}({'+'.join(self.experts)})"


def _build_routes(expert_keys: Sequence[str] = EXPERT_KEYS) -> tuple[Route, ...]:
    """Enumerate every non-empty subset of the experts, in bitmask order."""
    routes: list[Route] = []
    for mask in itertools.product([0, 1], repeat=len(expert_keys)):
        if not any(mask):
            continue  # the empty route is not a legal action
        # 'A' + index keeps labels single-character for up to 26 routes; with a
        # fifth expert (31 routes) switch to a two-token scheme in decoding.py.
        routes.append(Route(label=chr(ord("A") + len(routes)), mask=mask))
    return tuple(routes)


ROUTES: tuple[Route, ...] = _build_routes()
NUM_ROUTES: int = len(ROUTES)
ROUTE_LABELS: tuple[str, ...] = tuple(r.label for r in ROUTES)

_BY_LABEL: dict[str, Route] = {r.label: r for r in ROUTES}
_BY_JSON: dict[str, Route] = {r.as_json(): r for r in ROUTES}

assert NUM_ROUTES <= 26, "single-letter action labels only go up to 26 routes"


def route_from_label(label: str) -> Route | None:
    """Look up a route by action token. Returns ``None`` for malformed output.

    Constrained decoding makes malformed output nearly impossible, but GRPO
    resamples aggressively at high temperature and a stray whitespace token
    does show up occasionally -- callers treat ``None`` as a zero-reward rollout.
    """
    return _BY_LABEL.get(label.strip()[:1].upper()) if label and label.strip() else None


def route_from_mask(mask: Mapping[str, int] | Iterable[int]) -> Route | None:
    """Look up a route by its expert mask (dict or bit sequence)."""
    if isinstance(mask, Mapping):
        key = json.dumps({k: int(mask.get(k, 0)) for k in EXPERT_KEYS}, separators=(",", ":"))
    else:
        key = json.dumps(dict(zip(EXPERT_KEYS, (int(b) for b in mask))), separators=(",", ":"))
    return _BY_JSON.get(key)


def format_route_menu() -> str:
    """Render the route table that gets embedded in the Selector prompt."""
    return "\n".join(f"- {r.label}: {r.as_dict()}" for r in ROUTES)


__all__ = [
    "Route",
    "ROUTES",
    "NUM_ROUTES",
    "ROUTE_LABELS",
    "route_from_label",
    "route_from_mask",
    "format_route_menu",
]
