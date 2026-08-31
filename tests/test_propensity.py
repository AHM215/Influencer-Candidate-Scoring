import math

import pytest

from candidate_scoring.config import (
    BASE_RATE,
    ENGAGEMENT_FAMILY,
    LR_MAX,
    LR_MIN,
    POSTERIOR_MAX,
    PROPENSITY_CURVES,
)
from candidate_scoring.scoring.propensity import score_propensity
from conftest import signals


def test_no_signals_returns_the_base_rate():
    assert score_propensity({}).probability == pytest.approx(BASE_RATE)


def test_exactly_one_engagement_signal_ever_applies():
    """ADR-0003: the three engagement Signals are one piece of evidence, not three."""
    result = score_propensity(signals())
    applied = [c for c in result.contributions if c.applied and c.signal in ENGAGEMENT_FAMILY]
    assert len(applied) == 1
    suppressed = [c for c in result.contributions if not c.applied]
    assert {c.signal for c in suppressed} < set(ENGAGEMENT_FAMILY)
    assert all(c.suppressed_by == applied[0].signal for c in suppressed)


def test_an_engagement_farm_is_judged_on_its_weakest_indicator():
    """ADR-0003: inflated likes must not buy credit when comments and variance betray it."""
    farm = signals(engagement_rate=0.30, comment_like_ratio=0.002, engagement_consistency=0.05)
    result = score_propensity(farm)
    applied = next(c for c in result.contributions if c.applied and c.signal in ENGAGEMENT_FAMILY)
    assert applied.likelihood_ratio < 1.0
    assert result.probability < BASE_RATE


def test_corroborated_engagement_is_not_punished_for_the_grouping():
    """A genuine creator strong on all three keeps close to their best member."""
    genuine = signals(engagement_rate=0.06, comment_like_ratio=0.035, engagement_consistency=0.85)
    applied = next(
        c for c in score_propensity(genuine).contributions
        if c.applied and c.signal in ENGAGEMENT_FAMILY
    )
    assert applied.likelihood_ratio > 1.2


def test_a_flawless_candidate_still_stays_well_short_of_certainty():
    """The LR caps bind long before the ceiling does - by design, not by accident."""
    maxed = signals(
        engagement_rate=0.5, comment_like_ratio=0.2, engagement_consistency=1.0,
        posting_cadence=6.0, authenticity_plausibility=1.0,
        commercial_evidence=1.0, selling_content_style=1.0,
    )
    result = score_propensity(maxed)
    assert BASE_RATE < result.probability < 0.80
    assert not result.clipped


def test_the_ceiling_clips_when_the_prior_would_carry_us_past_it():
    strong = signals(commercial_evidence=1.0, selling_content_style=1.0)
    result = score_propensity(strong, base_rate=0.80)
    assert result.probability == pytest.approx(POSTERIOR_MAX)
    assert result.clipped


def test_every_likelihood_ratio_stays_inside_the_declared_band():
    for name, curve in PROPENSITY_CURVES.items():
        lo, hi = curve.points[0][0], curve.points[-1][0]
        for step in range(0, 101):
            value = lo + (hi - lo) * step / 100
            assert LR_MIN <= curve.likelihood_ratio(value) <= LR_MAX, name


def test_a_signal_below_its_curve_moves_the_odds_down_not_up():
    weak = signals(commercial_evidence=0.0)
    strong = signals(commercial_evidence=1.0)
    assert score_propensity(weak).probability < BASE_RATE < score_propensity(strong).probability


def test_contributions_reconstruct_the_posterior():
    """The explanation is the arithmetic, not a story told alongside it."""
    result = score_propensity(signals())
    log_odds = math.log(BASE_RATE / (1 - BASE_RATE))
    log_odds += sum(c.log_lr for c in result.contributions if c.applied)
    assert 1 / (1 + math.exp(-log_odds)) == pytest.approx(result.probability, abs=1e-9)
