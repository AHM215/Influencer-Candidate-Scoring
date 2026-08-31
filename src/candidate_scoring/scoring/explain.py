"""Reason codes: deterministic sentences built from the contributions.

No LLM in the output path. Everything a CM lead reads is generated from numbers the
scorer actually used, so it is testable and cannot drift between runs.
"""

from __future__ import annotations

from dataclasses import replace

from ..domain import (
    Construct,
    FitResult,
    PropensityResult,
    Provenance,
    ReasonCode,
    SignalValue,
)

TEMPLATES: dict[str, str] = {
    "category_alignment": "Content is {q} aligned to beauty, fragrance and fashion ({v:.0%})",
    "gcc_audience_share": "An estimated {v:.0%} of the audience is in the GCC",
    "audience_scale": "Audience scale scores {v:.0%} of what a Boutique needs",
    "language_fit": "Arabic/English language fit at {v:.0%}",
    "brand_safety": "Brand safety check at {v:.0%}",
    "engagement_rate": "Engagement rate of {v:.1%} of followers",
    "comment_like_ratio": "{v:.1%} of engagement is comments rather than likes",
    "engagement_consistency": "Engagement is {q} consistent across recent posts ({v:.0%})",
    "posting_cadence": "Posts {v:.1f} times per week",
    "authenticity_plausibility": "Engagement is {q} plausible for this follower tier ({v:.0%})",
    "commercial_evidence": "{q} evidence of prior affiliate or sponsored selling ({v:.0%})",
    "selling_content_style": "Content style is {q} suited to selling ({v:.0%})",
}

QUALIFIERS: dict[str, tuple[str, str, str]] = {
    "commercial_evidence": ("Little", "Some", "Strong"),
}
"""Signals whose sentence reads better with a different adjective set."""

PROVENANCE_NOTE = {
    Provenance.OBSERVED: "",
    Provenance.INFERRED: " (inferred, not measured)",
    Provenance.MOCKED: " (MOCKED - not real data)",
}


def build_reasons(
    fit: FitResult,
    propensity: PropensityResult,
    signals: dict[str, SignalValue],
    top_n: int = 3,
) -> tuple[ReasonCode, ...]:
    """The strongest drivers, then the weakest points.

    Reservations are the lowest-ranked drivers whether or not they are net-negative. A
    Candidate who is strong on everything still has a weakest link, and a Report that
    lists only what it liked is a sales pitch rather than an assessment.
    """
    codes = [*_propensity_codes(propensity), *_fit_codes(fit)]
    if "brand_safety" in signals:
        codes.append(_safety_code(fit, signals["brand_safety"]))

    ranked = sorted(codes, key=lambda c: c.strength, reverse=True)
    strengths = [c for c in ranked[:top_n] if c.strength > 0]
    chosen = {id(c) for c in strengths}
    reservations = [c for c in reversed(ranked) if id(c) not in chosen][:top_n]

    return tuple(
        [
            *strengths,
            *[replace(c, kind="reservation") for c in reservations],
        ]
    )


def _propensity_codes(propensity: PropensityResult) -> list[ReasonCode]:
    codes = []
    for c in propensity.contributions:
        if not c.applied:
            continue
        positive = c.likelihood_ratio >= 1.0
        effect = (
            f"multiplying the odds by {c.likelihood_ratio:.2f}x"
            if positive
            else f"cutting the odds to {c.likelihood_ratio:.2f}x"
        )
        codes.append(
            ReasonCode(
                construct=Construct.PROPENSITY,
                signal=c.signal,
                positive=positive,
                strength=c.log_lr,
                text=f"{_phrase(c.signal, c.value)}, {effect}"
                + PROVENANCE_NOTE[c.provenance],
                provenance=c.provenance,
            )
        )
    return codes


def _fit_codes(fit: FitResult) -> list[ReasonCode]:
    codes = []
    for c in fit.contributions:
        strong = c.value >= 0.6
        detail = (
            f"contributing {c.points:.0f} of {c.weight:.0f} Fit points"
            if strong
            else f"forgoing {c.forgone:.0f} of {c.weight:.0f} Fit points"
        )
        codes.append(
            ReasonCode(
                construct=Construct.FIT,
                signal=c.signal,
                positive=strong,
                strength=(c.value - 0.5) * c.weight / 35.0,
                text=f"{_phrase(c.signal, c.value)}, {detail}"
                + PROVENANCE_NOTE[c.provenance],
                provenance=c.provenance,
            )
        )
    return codes


def _safety_code(fit: FitResult, signal: SignalValue) -> ReasonCode:
    text = (
        f"Brand safety veto: {fit.veto_reason}"
        if fit.vetoed
        else f"No brand safety concerns found ({signal.value:.0%})"
    )
    return ReasonCode(
        construct=Construct.FIT,
        signal="brand_safety",
        positive=not fit.vetoed,
        strength=-10.0 if fit.vetoed else 0.35,
        text=text + PROVENANCE_NOTE[signal.provenance],
        provenance=signal.provenance,
    )


def _phrase(signal: str, value: float) -> str:
    template = TEMPLATES.get(signal, f"{signal} at {{v:.2f}}")
    return template.format(v=value, q=_qualifier(signal, value))


def _qualifier(signal: str, value: float) -> str:
    """Adjective for the {q} slot. Only 0-1 Signals use it, so value is already a fraction."""
    weak, mid, strong = QUALIFIERS.get(signal, ("poorly", "moderately", "strongly"))
    if value >= 0.75:
        return strong
    if value >= 0.45:
        return mid
    return weak
