import numpy as np
import pytest

from candidate_scoring.validation.harness import run_harness
from candidate_scoring.validation.metrics import auc, spearman


def test_all_behavioural_checks_pass(cohort):
    report = run_harness(cohort)
    failures = [c.name for c in report.checks if not c.passed]
    assert not failures, f"behavioural checks failed: {failures}"


def test_ablation_covers_every_propensity_signal(cohort):
    from candidate_scoring.config import PROPENSITY_SIGNALS

    report = run_harness(cohort)
    assert {a.signal for a in report.ablations} == set(PROPENSITY_SIGNALS)


def test_calibration_buckets_report_both_predicted_and_observed(cohort):
    report = run_harness(cohort)
    assert report.calibration
    for _, count, predicted, observed in report.calibration:
        assert count > 0
        assert 0.0 <= predicted <= 1.0 and 0.0 <= observed <= 1.0


def test_auc_is_half_for_a_coin_flip():
    rng = np.random.default_rng(0)
    labels = rng.random(500) < 0.3
    assert abs(auc(rng.random(500), labels) - 0.5) < 0.08


def test_auc_is_one_for_a_perfect_ranker():
    labels = np.array([False] * 50 + [True] * 50)
    assert auc(np.arange(100, dtype=float), labels) == 1.0


def test_spearman_catches_a_monotone_but_non_linear_relationship():
    x = np.arange(1, 51, dtype=float)
    assert spearman(x, x**3) == pytest.approx(1.0)
