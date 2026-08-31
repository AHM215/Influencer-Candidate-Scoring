"""Snapshot in, ScoredCandidate out. The one path everything else calls."""

from __future__ import annotations

from .domain import ProfileSnapshot, ScoredCandidate, SignalValue
from .scoring.explain import build_reasons
from .scoring.fit import score_fit
from .scoring.policy import recommend
from .scoring.propensity import score_propensity
from .signals import Extractor, build_signals


def score_signals(
    signals: dict[str, SignalValue], snapshot: ProfileSnapshot
) -> ScoredCandidate:
    fit = score_fit(signals)
    propensity = score_propensity(signals)
    recommendation, reason = recommend(fit, propensity)
    return ScoredCandidate(
        snapshot=snapshot,
        signals=tuple(signals.values()),
        fit=fit,
        propensity=propensity,
        recommendation=recommendation,
        policy_reason=reason,
        reasons=build_reasons(fit, propensity, signals),
    )


def score_snapshot(
    snapshot: ProfileSnapshot, extractor: Extractor | None = None
) -> ScoredCandidate:
    return score_signals(build_signals(snapshot, extractor), snapshot)


def rank(candidates: list[ScoredCandidate]) -> list[ScoredCandidate]:
    """The Shortlist order: onboard first, then by Propensity, then by Fit.

    Sorting by Propensity alone would float a strong seller in the wrong category above
    a sound Candidate, so the Recommendation leads and the scores break ties within it.
    """
    order = {"onboard": 0, "hold": 1, "pass": 2}
    return sorted(
        candidates,
        key=lambda c: (
            order[c.recommendation.value],
            -c.propensity.probability,
            -c.fit.score,
        ),
    )
