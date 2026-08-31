from candidate_scoring.cohort import generate_cohort
from candidate_scoring.cohort.generate import OUTCOME_WEIGHTS
from candidate_scoring.config import BASE_RATE, PROPENSITY_SIGNALS


def test_cohort_is_reproducible():
    assert [m.cleared_bar for m in generate_cohort(50, seed=3)] == [
        m.cleared_bar for m in generate_cohort(50, seed=3)
    ]


def test_outcome_rate_lands_near_the_base_rate(cohort):
    """Within three binomial standard errors of the Base Rate the generator targets."""
    n = len(cohort)
    rate = sum(m.cleared_bar for m in cohort) / n
    tolerance = 3 * (BASE_RATE * (1 - BASE_RATE) / n) ** 0.5
    assert abs(rate - BASE_RATE) < tolerance


def test_the_generator_disagrees_with_the_scorer_by_design():
    """ADR-0004: aligning these would destroy the only informative thing the harness has."""
    assert OUTCOME_WEIGHTS["commercial_evidence"] > OUTCOME_WEIGHTS["engagement_rate_norm"]


def test_every_member_carries_all_twelve_signals(cohort):
    for member in cohort:
        for name in PROPENSITY_SIGNALS:
            assert name in member.signals
        assert len(member.signals) == 12


def test_larger_accounts_show_lower_engagement(cohort):
    """The confounder is present, so the harness is not scoring a clean world."""
    ordered = sorted(cohort, key=lambda m: m.snapshot.followers)
    small = ordered[: len(ordered) // 4]
    large = ordered[-len(ordered) // 4 :]
    def mean(group):
        return sum(m.signals["engagement_rate"].value for m in group) / len(group)

    assert mean(small) > mean(large)
