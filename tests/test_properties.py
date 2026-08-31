"""Behavioural properties over the whole synthetic Cohort.

These are the real validation: not "is the score accurate" (unanswerable without real
outcomes) but "does the scorer behave the way we said it does, on every input".
"""

from dataclasses import replace
from datetime import date
from math import isfinite

import pytest

from candidate_scoring.config import (
    BASE_RATE,
    BRAND_SAFETY_VETO_BELOW,
    FIT_FLOOR,
    PROPENSITY_CURVES,
    PROPENSITY_SIGNALS,
)
from candidate_scoring.domain import Recommendation
from candidate_scoring.pipeline import score_signals, score_snapshot
from candidate_scoring.report.render import render_candidate
from candidate_scoring.signals import derive_signals
from conftest import signals


def test_declared_monotonic_curves_never_reverse():
    for name, curve in PROPENSITY_CURVES.items():
        if not curve.monotonic:
            continue
        lo, hi = curve.points[0][0], curve.points[-1][0]
        ratios = [curve.likelihood_ratio(lo + (hi - lo) * i / 200) for i in range(201)]
        assert all(b >= a - 1e-9 for a, b in zip(ratios, ratios[1:], strict=False)), name


def test_the_one_curve_declared_non_monotonic_actually_turns():
    curve = PROPENSITY_CURVES["posting_cadence"]
    assert not curve.monotonic
    assert curve.likelihood_ratio(14.0) < curve.likelihood_ratio(8.0)


def test_improving_any_propensity_signal_never_lowers_the_score():
    for name in PROPENSITY_SIGNALS:
        curve = PROPENSITY_CURVES[name]
        if not curve.monotonic:
            continue
        lo, hi = curve.points[0][0], curve.points[-1][0]
        worse = score_signals(signals(**{name: lo}), _snapshot()).propensity.probability
        better = score_signals(signals(**{name: hi}), _snapshot()).propensity.probability
        assert better >= worse, name


def test_no_cohort_member_below_the_fit_floor_reaches_onboard(cohort):
    for member in cohort:
        scored = score_signals(member.signals, member.snapshot)
        if scored.fit.score < FIT_FLOOR:
            assert scored.recommendation is not Recommendation.ONBOARD


def test_every_flagged_cohort_member_is_passed(cohort):
    for member in cohort:
        if member.signals["brand_safety"].value < BRAND_SAFETY_VETO_BELOW:
            scored = score_signals(member.signals, member.snapshot)
            assert scored.recommendation is Recommendation.PASS


def test_scoring_is_deterministic(cohort):
    first = [score_signals(m.signals, m.snapshot).propensity.probability for m in cohort]
    second = [score_signals(m.signals, m.snapshot).propensity.probability for m in cohort]
    assert first == second


@pytest.mark.parametrize(
    "snapshot",
    [
        pytest.param(lambda base: replace(base, posts=(), post_count=0), id="no-posts"),
        pytest.param(
            lambda base: replace(
                base,
                posts=tuple(replace(post, likes=0, comments=0) for post in base.posts),
            ),
            id="zero-engagement",
        ),
        pytest.param(
            lambda base: replace(
                base,
                posts=(replace(base.posts[0], likes=1, comments=0),),
                post_count=1,
            ),
            id="one-post",
        ),
        pytest.param(lambda base: replace(base, followers=0), id="zero-followers"),
    ],
)
def test_dormant_or_degenerate_snapshots_remain_scoreable(snapshot, archetype_extractor):
    """An inert audience must not make public data unusable or look sellable."""
    from candidate_scoring.archetypes import ARCHETYPES

    # Named rather than indexed: reordering the Archetypes must not silently
    # change which Candidate this property is asserted against.
    base = next(a for a in ARCHETYPES if a.snapshot.handle == "archetype_promising_unproven")
    degenerate = snapshot(base.snapshot)
    derived = derive_signals(degenerate)
    candidate = score_snapshot(degenerate, archetype_extractor)

    assert all(isfinite(signal.value) for signal in derived.values())
    assert isinstance(candidate.recommendation, Recommendation)
    assert render_candidate(candidate, as_of=date(2026, 8, 31))
    assert candidate.propensity.probability <= BASE_RATE
    reservations = [r for r in candidate.reasons if r.kind == "reservation"]
    assert reservations, "a dormant Candidate must say what is holding it back"


def _snapshot():
    from candidate_scoring.cohort import generate_cohort

    return generate_cohort(1, seed=1)[0].snapshot
