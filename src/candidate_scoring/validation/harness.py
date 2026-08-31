"""What we can honestly check without real onboarding outcomes.

The behavioural properties and the Archetype regressions are the real harness. The
Cohort metrics at the end measure the pipeline, not the hypothesis: the labels are
invented, so a good number there is not evidence the scoring is right.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..cohort import CohortMember, generate_cohort
from ..config import (
    BRAND_SAFETY_VETO_BELOW,
    FIT_FLOOR,
    PROPENSITY_CURVES,
    PROPENSITY_SIGNALS,
)
from ..domain import Recommendation
from ..pipeline import score_signals
from .metrics import auc, fit_logistic, spearman


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class Ablation:
    signal: str
    rank_correlation: float
    auc_delta: float


@dataclass
class HarnessReport:
    checks: list[Check] = field(default_factory=list)
    ablations: list[Ablation] = field(default_factory=list)
    calibration: list[tuple[str, int, float, float]] = field(default_factory=list)
    cohort_auc: float = float("nan")
    logistic_agreement: list[tuple[str, float, float]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


def run_harness(cohort: list[CohortMember] | None = None) -> HarnessReport:
    cohort = cohort or generate_cohort()
    report = HarnessReport()
    scored = [score_signals(m.signals, m.snapshot) for m in cohort]

    report.checks.extend(_curve_monotonicity())
    report.checks.append(_fit_floor_invariant(scored))
    report.checks.append(_veto_invariant(cohort, scored))
    report.checks.append(_family_suppression(scored))

    probabilities = np.array([s.propensity.probability for s in scored])
    labels = np.array([m.cleared_bar for m in cohort])

    report.cohort_auc = auc(probabilities, labels)
    report.ablations = _ablations(cohort, probabilities, labels)
    report.calibration = _calibration(probabilities, labels)
    report.logistic_agreement = _logistic_cross_check(cohort, labels)
    return report


def _curve_monotonicity() -> list[Check]:
    checks = []
    for name, curve in PROPENSITY_CURVES.items():
        xs = np.linspace(curve.points[0][0], curve.points[-1][0], 200)
        ratios = [curve.likelihood_ratio(float(x)) for x in xs]
        increasing = all(b >= a - 1e-9 for a, b in zip(ratios, ratios[1:], strict=False))
        if curve.monotonic:
            checks.append(
                Check(
                    f"monotonic:{name}",
                    increasing,
                    "more of this Signal never lowers the odds"
                    if increasing
                    else "curve reverses despite being declared monotonic",
                )
            )
        else:
            checks.append(
                Check(
                    f"declared-non-monotonic:{name}",
                    not increasing,
                    "curve turns back down as declared"
                    if not increasing
                    else "declared non-monotonic but never reverses - drop the flag",
                )
            )
    return checks


def _fit_floor_invariant(scored) -> Check:
    offenders = [
        s
        for s in scored
        if s.fit.score < FIT_FLOOR and s.recommendation is Recommendation.ONBOARD
    ]
    return Check(
        "fit-floor-caps-at-hold",
        not offenders,
        f"no Candidate below Fit {FIT_FLOOR:.0f} reaches onboard"
        if not offenders
        else f"{len(offenders)} Candidates broke the Fit Floor",
    )


def _veto_invariant(cohort, scored) -> Check:
    offenders = [
        s
        for m, s in zip(cohort, scored, strict=True)
        if m.signals["brand_safety"].value < BRAND_SAFETY_VETO_BELOW
        and s.recommendation is not Recommendation.PASS
    ]
    return Check(
        "brand-safety-vetoes",
        not offenders,
        "every flagged Candidate is a pass"
        if not offenders
        else f"{len(offenders)} flagged Candidates escaped the veto",
    )


def _family_suppression(scored) -> Check:
    """Exactly one engagement Signal may contribute to any Candidate (ADR-0003)."""
    from ..config import ENGAGEMENT_FAMILY

    bad = [
        s
        for s in scored
        if sum(
            1 for c in s.propensity.contributions if c.applied and c.signal in ENGAGEMENT_FAMILY
        )
        != 1
    ]
    return Check(
        "engagement-family-counted-once",
        not bad,
        "one engagement Signal applied per Candidate"
        if not bad
        else f"{len(bad)} Candidates double-counted engagement",
    )


def _ablations(cohort, baseline: np.ndarray, labels: np.ndarray) -> list[Ablation]:
    """Neutralise each Signal in turn and see how much the ranking moves.

    This is the closest this system gets to answering "which signals actually predict
    success": it shows which Signals move OUR model, not which move reality.
    """
    results = []
    baseline_auc = auc(baseline, labels)
    for signal in PROPENSITY_SIGNALS:
        muted = []
        for member in cohort:
            signals = dict(member.signals)
            signals.pop(signal, None)
            muted.append(score_signals(signals, member.snapshot).propensity.probability)
        muted_array = np.array(muted)
        results.append(
            Ablation(
                signal=signal,
                rank_correlation=spearman(baseline, muted_array),
                auc_delta=auc(muted_array, labels) - baseline_auc,
            )
        )
    return sorted(results, key=lambda a: a.rank_correlation)


def _calibration(probabilities: np.ndarray, labels: np.ndarray):
    edges = [0.0, 0.15, 0.25, 0.40, 0.60, 1.01]
    rows = []
    for low, high in zip(edges, edges[1:], strict=False):
        mask = (probabilities >= low) & (probabilities < high)
        if mask.sum() == 0:
            continue
        rows.append(
            (
                f"{low:.0%}-{high:.0%}",
                int(mask.sum()),
                float(probabilities[mask].mean()),
                float(labels[mask].mean()),
            )
        )
    return rows


def _logistic_cross_check(cohort, labels: np.ndarray):
    """ADR-0002: if a fitted model disagrees violently with our priors, say so.

    Reported as standardised coefficients against the log of each Signal's mid-range
    Likelihood Ratio. Sign disagreement is the finding worth acting on; magnitudes are
    not comparable and are not presented as if they were.
    """
    matrix = np.array(
        [[m.signals[s].value for s in PROPENSITY_SIGNALS] for m in cohort], dtype=float
    )
    _, coefficients = fit_logistic(matrix, labels)
    rows = []
    for signal, coefficient in zip(PROPENSITY_SIGNALS, coefficients, strict=True):
        curve = PROPENSITY_CURVES[signal]
        prior = np.log(curve.points[-1][1] / curve.points[0][1])
        rows.append((signal, float(coefficient), float(prior)))
    return rows
