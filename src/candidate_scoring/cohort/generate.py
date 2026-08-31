"""Generates synthetic Candidates with synthetic outcomes.

The outcome function here is deliberately NOT the scorer's function (ADR-0004). It
weights commercial evidence and audience fit more heavily and engagement less, over a
confounder where larger accounts show systematically lower engagement. The scorer is
therefore not expected to fit this Cohort well: a mediocre agreement is the designed
result, and disagreement shows which of our stated priors would cost us most if reality
disagreed with them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from ..config import BASE_RATE
from ..domain import Construct, ProfileSnapshot, Provenance, SignalValue

# The generator's beliefs, in log-odds per unit of Signal. Compare with config.py:
# the scorer's strongest lever is engagement; the generator's is commercial evidence.
OUTCOME_WEIGHTS = {
    "commercial_evidence": 1.9,
    "gcc_audience_share": 1.5,
    "category_alignment": 1.2,
    "selling_content_style": 0.8,
    "posting_cadence_norm": 0.6,
    "engagement_rate_norm": 0.5,
}
"""Note the scorer's strongest lever is engagement; the generator's is commercial
evidence. That divergence is deliberate (ADR-0004) and must not be tuned away."""


@dataclass(frozen=True)
class CohortMember:
    signals: dict[str, SignalValue]
    cleared_bar: bool
    snapshot: ProfileSnapshot


def generate_cohort(n: int = 400, seed: int = 20260831) -> list[CohortMember]:
    rng = np.random.default_rng(seed)

    quality = rng.beta(2.0, 3.0, n)
    followers = np.exp(rng.normal(np.log(60_000), 1.1, n)).astype(int)

    # Confounder: engagement falls with size regardless of how good the Candidate is.
    size_drag = (followers / 10_000.0) ** -0.18
    engagement = np.clip(0.09 * size_drag * (0.5 + quality) + rng.normal(0, 0.012, n), 0.001, 0.4)

    # A tenth of the Cohort buys engagement: high rate, thin comments, erratic, implausible.
    farms = rng.random(n) < 0.10
    engagement = np.where(farms, engagement * rng.uniform(2.5, 5.0, n), engagement)

    comment_share = np.clip(
        0.02 + 0.06 * quality + rng.normal(0, 0.012, n) - 0.02 * farms, 0.001, 0.3
    )
    consistency = np.clip(0.3 + 0.6 * quality + rng.normal(0, 0.15, n) - 0.3 * farms, 0.0, 1.0)
    cadence = np.clip(1.0 + 7.0 * quality + rng.normal(0, 1.5, n), 0.2, 14.0)
    authenticity = np.where(farms, rng.uniform(0.05, 0.4, n), rng.uniform(0.75, 1.0, n))

    commercial = np.clip(quality * rng.uniform(0.6, 1.3, n), 0.0, 1.0)
    style = np.clip(0.2 + 0.7 * quality + rng.normal(0, 0.15, n), 0.0, 1.0)

    # Fit is about identity, not quality: drawn independently of the latent.
    category = np.clip(rng.beta(2.2, 2.0, n), 0.0, 1.0)
    gcc = np.clip(rng.beta(2.5, 1.8, n), 0.0, 1.0)
    language = np.clip(rng.beta(3.0, 1.5, n), 0.0, 1.0)
    safety = np.where(rng.random(n) < 0.06, rng.uniform(0.1, 0.49, n), rng.uniform(0.8, 1.0, n))

    # Only genuine engagement converts. Without this the synthetic world rewards bought
    # engagement, which is false by construction - a farm's audience is not real - and the
    # harness's logistic cross-check duly flagged authenticity as disagreeing with our
    # prior. That was a misspecified generator, not a misspecified scorer.
    engagement_norm = np.clip(engagement / 0.08, 0, 1.5) * authenticity
    cadence_norm = np.clip(cadence / 8.0, 0, 1.5)

    logits = (
        OUTCOME_WEIGHTS["commercial_evidence"] * commercial
        + OUTCOME_WEIGHTS["gcc_audience_share"] * gcc
        + OUTCOME_WEIGHTS["category_alignment"] * category
        + OUTCOME_WEIGHTS["selling_content_style"] * style
        + OUTCOME_WEIGHTS["posting_cadence_norm"] * cadence_norm
        + OUTCOME_WEIGHTS["engagement_rate_norm"] * engagement_norm
        + rng.normal(0, 0.6, n)  # everything we cannot see from a public profile
    )
    logits -= _intercept_for(logits, BASE_RATE)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    cleared = rng.random(n) < probabilities

    members = []
    for i in range(n):
        raw = {
            "audience_scale": _scale(followers[i]),
            "engagement_rate": float(engagement[i]),
            "comment_like_ratio": float(comment_share[i]),
            "engagement_consistency": float(consistency[i]),
            "posting_cadence": float(cadence[i]),
            "authenticity_plausibility": float(authenticity[i]),
            "commercial_evidence": float(commercial[i]),
            "selling_content_style": float(style[i]),
            "category_alignment": float(category[i]),
            "gcc_audience_share": float(gcc[i]),
            "language_fit": float(language[i]),
            "brand_safety": float(safety[i]),
        }
        handle = f"cohort_{i:04d}"
        members.append(
            CohortMember(
                signals={
                    name: SignalValue(
                        name=name,
                        construct=_construct(name),
                        value=value,
                        provenance=Provenance.MOCKED,
                        detail="synthetic",
                    )
                    for name, value in raw.items()
                },
                cleared_bar=bool(cleared[i]),
                snapshot=ProfileSnapshot(
                    handle=handle,
                    platform="synthetic",
                    captured_at=date(2026, 8, 31),
                    followers=int(followers[i]),
                    following=0,
                    post_count=0,
                    bio="",
                    verified=False,
                    posts=(),
                    provenance=Provenance.MOCKED,
                    display_name=handle,
                ),
            )
        )
    return members


def _intercept_for(logits: np.ndarray, target: float, tolerance: float = 1e-4) -> float:
    """Shift the logits so the mean success probability lands on the Base Rate."""
    low, high = -20.0, 20.0
    for _ in range(200):
        mid = (low + high) / 2
        mean = float(np.mean(1.0 / (1.0 + np.exp(-(logits - mid)))))
        if abs(mean - target) < tolerance:
            return mid
        if mean > target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def _scale(followers: int) -> float:
    from ..signals.derive import _audience_scale

    return _audience_scale(int(followers))


def _construct(name: str) -> Construct:
    from ..config import SIGNAL_CONSTRUCT

    return SIGNAL_CONSTRUCT[name]
