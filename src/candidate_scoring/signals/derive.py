"""Quantitative Signals derived from a Profile Snapshot. No model, no network."""

from __future__ import annotations

import math
import statistics

from ..config import AUDIENCE_SCALE_POINTS
from ..domain import Construct, ProfileSnapshot, Provenance, SignalValue


def derive_signals(snapshot: ProfileSnapshot) -> dict[str, SignalValue]:
    engagements = [p.likes + p.comments for p in snapshot.posts]
    provenance = snapshot.provenance

    values = {
        "audience_scale": SignalValue(
            "audience_scale",
            Construct.FIT,
            _audience_scale(snapshot.followers),
            provenance,
            f"{snapshot.followers:,} followers",
        ),
        "engagement_rate": SignalValue(
            "engagement_rate",
            Construct.PROPENSITY,
            _engagement_rate(snapshot, engagements),
            provenance,
            f"median engagement across {len(snapshot.posts)} recent posts",
        ),
        "comment_like_ratio": SignalValue(
            "comment_like_ratio",
            Construct.PROPENSITY,
            _comment_share(snapshot),
            provenance,
            "comments as a share of total engagement",
        ),
        "engagement_consistency": SignalValue(
            "engagement_consistency",
            Construct.PROPENSITY,
            _consistency(engagements),
            provenance,
            "1 - coefficient of variation of per-post engagement",
        ),
        "posting_cadence": SignalValue(
            "posting_cadence",
            Construct.PROPENSITY,
            _cadence(snapshot),
            provenance,
            "posts per week across the snapshot window",
        ),
    }

    values["authenticity_plausibility"] = SignalValue(
        "authenticity_plausibility",
        Construct.PROPENSITY,
        _plausibility(snapshot.followers, values["engagement_rate"].value),
        Provenance.INFERRED,
        "engagement rate against what is typical for this follower tier",
    )
    return values


def _audience_scale(followers: int) -> float:
    xs = [math.log10(max(p[0], 1)) for p in AUDIENCE_SCALE_POINTS]
    ys = [p[1] for p in AUDIENCE_SCALE_POINTS]
    x = math.log10(max(followers, 1))
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = next(i for i in range(len(xs) - 1) if xs[i] <= x <= xs[i + 1])
    t = (x - xs[i]) / (xs[i + 1] - xs[i])
    return ys[i] + t * (ys[i + 1] - ys[i])


def _engagement_rate(snapshot: ProfileSnapshot, engagements: list[int]) -> float:
    if not engagements or snapshot.followers <= 0:
        return 0.0
    return statistics.median(engagements) / snapshot.followers


def _comment_share(snapshot: ProfileSnapshot) -> float:
    shares = [
        p.comments / (p.likes + p.comments)
        for p in snapshot.posts
        if (p.likes + p.comments) > 0
    ]
    return statistics.median(shares) if shares else 0.0


def _consistency(engagements: list[int]) -> float:
    if len(engagements) < 2:
        return 0.5
    mean = statistics.fmean(engagements)
    if mean <= 0:
        return 0.0
    cv = statistics.stdev(engagements) / mean
    return max(0.0, 1.0 - cv)


def _cadence(snapshot: ProfileSnapshot) -> float:
    if not snapshot.posts:
        return 0.0
    earliest = min(p.posted_at for p in snapshot.posts)
    days = max((snapshot.captured_at - earliest).days, 1)
    return len(snapshot.posts) / (days / 7.0)


def _plausibility(followers: int, engagement_rate: float) -> float:
    """Penalises engagement that is implausibly HIGH for the follower tier only.

    Implausibly low engagement is already priced by the engagement_rate Signal, so
    penalising it here too would count one weakness twice - the same double-counting
    ADR-0003 avoids inside the engagement family.
    """
    if followers <= 0 or engagement_rate <= 0:
        return 0.5
    expected = 0.10 * (max(followers, 1_000) / 10_000) ** -0.18
    ratio = engagement_rate / expected
    if ratio <= 1.5:
        return 1.0
    return max(0.0, 1.0 - math.log(ratio / 1.5) / math.log(6.0))
