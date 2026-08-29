"""The action space: 15 non-empty routes, single-letter labels, round-trips.

None of these need a GPU or a model download -- they guard the invariants that
would otherwise fail silently deep inside a training run.
"""

from __future__ import annotations

import pytest

from persuarl.constants import EXPERT_KEYS
from persuarl.routes import (
    NUM_ROUTES,
    ROUTE_LABELS,
    ROUTES,
    format_route_menu,
    route_from_label,
    route_from_mask,
)


def test_route_count_is_two_to_the_n_minus_one():
    """Every non-empty subset of the experts, and only those."""
    assert NUM_ROUTES == 2 ** len(EXPERT_KEYS) - 1 == 15


def test_no_empty_route():
    """A turn must consult at least one expert -- the empty mask is not an action."""
    assert all(route.size >= 1 for route in ROUTES)


def test_labels_are_unique_single_characters():
    assert len(set(ROUTE_LABELS)) == NUM_ROUTES
    assert all(len(label) == 1 for label in ROUTE_LABELS)


def test_masks_are_unique():
    assert len({route.mask for route in ROUTES}) == NUM_ROUTES


@pytest.mark.parametrize("route", ROUTES)
def test_label_round_trip(route):
    assert route_from_label(route.label) is route


@pytest.mark.parametrize("route", ROUTES)
def test_mask_round_trip(route):
    assert route_from_mask(route.as_dict()) is route


def test_label_lookup_is_forgiving_about_whitespace_and_case():
    """GRPO samples at high temperature; a stray space should not lose the action."""
    assert route_from_label("  c  ") is route_from_label("C")
    assert route_from_label("") is None
    assert route_from_label("z") is None


def test_experts_follow_canonical_order():
    """The route's expert tuple must match EXPERT_KEYS order, not insertion order."""
    for route in ROUTES:
        assert list(route.experts) == [k for k in EXPERT_KEYS if route.as_dict()[k]]


def test_all_experts_route_exists():
    every = route_from_mask({key: 1 for key in EXPERT_KEYS})
    assert every is not None and every.size == len(EXPERT_KEYS)


def test_menu_lists_every_route():
    menu = format_route_menu()
    assert menu.count("\n") == NUM_ROUTES - 1
    for label in ROUTE_LABELS:
        assert f"- {label}:" in menu
