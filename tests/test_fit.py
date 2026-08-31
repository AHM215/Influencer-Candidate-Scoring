from candidate_scoring.config import BRAND_SAFETY_VETO_BELOW, FIT_WEIGHTS
from candidate_scoring.scoring.fit import score_fit
from conftest import signals


def test_perfect_fit_scores_one_hundred():
    result = score_fit(signals(**{name: 1.0 for name in FIT_WEIGHTS}))
    assert result.score == 100.0
    assert not result.vetoed


def test_weights_are_the_only_thing_setting_the_scale():
    """Each Fit Signal contributes exactly its weight and nothing more."""
    for name, weight in FIT_WEIGHTS.items():
        only_this = {n: 0.0 for n in FIT_WEIGHTS}
        only_this[name] = 1.0
        assert score_fit(signals(**only_this)).score == weight


def test_brand_safety_vetoes_an_otherwise_perfect_candidate():
    perfect = {name: 1.0 for name in FIT_WEIGHTS}
    result = score_fit(signals(**perfect, brand_safety=BRAND_SAFETY_VETO_BELOW - 0.01))
    assert result.vetoed
    assert result.score == 100.0, "the veto is a separate decision, not a score penalty"


def test_brand_safety_at_the_threshold_does_not_veto():
    result = score_fit(signals(brand_safety=BRAND_SAFETY_VETO_BELOW))
    assert not result.vetoed


def test_values_outside_zero_one_cannot_inflate_the_score():
    assert score_fit(signals(**{n: 5.0 for n in FIT_WEIGHTS})).score == 100.0
