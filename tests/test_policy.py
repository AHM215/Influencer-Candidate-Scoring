from candidate_scoring.config import FIT_FLOOR, ONBOARD_LIFT, PASS_LIFT_HIGH_FIT
from candidate_scoring.domain import FitResult, PropensityResult, Recommendation
from candidate_scoring.scoring.policy import recommend


def fit(score: float, vetoed: bool = False) -> FitResult:
    return FitResult(score=score, contributions=(), vetoed=vetoed, veto_reason="flagged")


def propensity(probability: float, base_rate: float = 0.2) -> PropensityResult:
    return PropensityResult(probability=probability, base_rate=base_rate, contributions=())


def test_veto_beats_everything():
    result, reason = recommend(fit(100.0, vetoed=True), propensity(0.85))
    assert result is Recommendation.PASS
    assert "veto" in reason.lower()


def test_high_fit_and_high_propensity_onboards():
    result, _ = recommend(fit(80.0), propensity(0.2 * ONBOARD_LIFT))
    assert result is Recommendation.ONBOARD


def test_below_the_fit_floor_caps_at_hold_however_strong_the_propensity():
    result, reason = recommend(fit(FIT_FLOOR - 0.1), propensity(0.85))
    assert result is Recommendation.HOLD
    assert "Fit Floor" in reason


def test_good_fit_with_merely_absent_evidence_is_held_not_passed():
    """Absent evidence is not negative evidence; pass means do not come back."""
    result, _ = recommend(fit(80.0), propensity(0.18))
    assert result is Recommendation.HOLD


def test_good_fit_with_actively_bad_evidence_is_passed():
    result, _ = recommend(fit(80.0), propensity(0.2 * PASS_LIFT_HIGH_FIT - 0.01))
    assert result is Recommendation.PASS


def test_poor_fit_below_the_base_rate_is_passed():
    result, _ = recommend(fit(30.0), propensity(0.19))
    assert result is Recommendation.PASS


def test_every_reason_names_a_number_the_lead_can_argue_with():
    for f, p in ((80.0, 0.5), (40.0, 0.5), (80.0, 0.05), (80.0, 0.25)):
        _, reason = recommend(fit(f), propensity(p))
        assert any(ch.isdigit() for ch in reason)
