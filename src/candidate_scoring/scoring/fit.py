"""Fit: how well a Candidate matches what a Boutique Owner needs to be, judged today.

Deterministic and checkable with no historical outcome data of any kind. That is the
whole point of keeping it separate from Success Propensity (ADR-0001).
"""

from __future__ import annotations

from ..config import BRAND_SAFETY_VETO_BELOW, FIT_WEIGHTS
from ..domain import FitContribution, FitResult, Provenance, SignalValue


def score_fit(signals: dict[str, SignalValue]) -> FitResult:
    contributions = []
    for name, weight in FIT_WEIGHTS.items():
        signal = signals.get(name)
        if signal is None:
            raise KeyError(f"Fit requires signal {name!r}")
        contributions.append(
            FitContribution(
                signal=name,
                weight=weight,
                value=_clamp(signal.value),
                provenance=signal.provenance,
            )
        )

    score = sum(c.points for c in contributions)

    safety = signals.get("brand_safety")
    if safety is not None and safety.value < BRAND_SAFETY_VETO_BELOW:
        return FitResult(
            score=score,
            contributions=tuple(contributions),
            vetoed=True,
            veto_reason=safety.detail or "brand safety concerns on recent content",
        )

    return FitResult(score=score, contributions=tuple(contributions))


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def missing_fit_evidence(signals: dict[str, SignalValue]) -> tuple[str, ...]:
    """Fit Signals resting on mocked values. The Report shows these prominently."""
    return tuple(
        name
        for name in FIT_WEIGHTS
        if name in signals and signals[name].provenance is Provenance.MOCKED
    )
