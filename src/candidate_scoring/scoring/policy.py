"""The Recommendation policy: an explicit grid over Fit and Success Propensity.

Nothing is blended. Each branch below is a sentence a CM lead can disagree with, which
is the reason the two constructs were kept apart in the first place (ADR-0001).
"""

from __future__ import annotations

from ..config import FIT_FLOOR, ONBOARD_LIFT, PASS_LIFT, PASS_LIFT_HIGH_FIT
from ..domain import FitResult, PropensityResult, Recommendation


def recommend(fit: FitResult, propensity: PropensityResult) -> tuple[Recommendation, str]:
    if fit.vetoed:
        return (
            Recommendation.PASS,
            f"Brand safety veto: {fit.veto_reason}. No Propensity justifies overriding this.",
        )

    lift = propensity.lift
    clears_floor = fit.score >= FIT_FLOOR
    pass_below = PASS_LIFT_HIGH_FIT if clears_floor else PASS_LIFT

    if lift < pass_below:
        detail = (
            "and the behavioural evidence is actively bad, not merely absent"
            if clears_floor
            else "and the Candidate does not clear the Fit Floor"
        )
        return (
            Recommendation.PASS,
            f"Propensity {propensity.probability:.0%} is {lift:.1f}x the "
            f"{propensity.base_rate:.0%} Base Rate {detail}.",
        )

    if not clears_floor:
        return (
            Recommendation.HOLD,
            f"Fit {fit.score:.0f} is below the Fit Floor of {FIT_FLOOR:.0f}. Capped at hold: "
            "a strong performer in the wrong category is still not a Boutique Owner.",
        )

    if lift >= ONBOARD_LIFT:
        return (
            Recommendation.ONBOARD,
            f"Fit {fit.score:.0f} clears the floor and Propensity {propensity.probability:.0%} "
            f"is {lift:.1f}x the Base Rate.",
        )

    return (
        Recommendation.HOLD,
        f"Fit {fit.score:.0f} is sound but Propensity {propensity.probability:.0%} is only "
        f"{lift:.1f}x the Base Rate, short of the {ONBOARD_LIFT:.0f}x bar for committing "
        "Onboarding Capacity.",
    )
