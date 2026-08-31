"""Every tunable number in the system, with the argument for it.

Nothing here was fitted to data. These are stated priors, and the Candidate Report
labels them as such. When real onboarding outcomes exist, the Likelihood Ratios below
are exactly the quantities to re-estimate (ADR-0002).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from .domain import Construct

# --- Success Propensity -------------------------------------------------------------

BASE_RATE = 0.20
"""Assumed share of onboarded Boutique Owners who clear the Performance Bar.

A stated assumption, not a measurement. Influencer commerce is a power law and a fifth
clearing a meaningful bar is defensible without being flattering. This is the one number
a CM lead could supply from memory: if they do, change it here and the whole model moves.
"""

PERFORMANCE_BAR = "attributed sales in the top 20% of first-90-day cohort performance"
"""The definition of success everything downstream inherits. Placeholder until CM sets it."""

LR_MIN, LR_MAX = 0.5, 2.0
"""No single Signal may move the odds by more than 2x either way.

With seven Signals and no cap, a merely good Candidate reaches implausible certainty.
"""

POSTERIOR_MIN, POSTERIOR_MAX = 0.02, 0.85
"""Nothing we can see from a public profile justifies more confidence than this."""

ENGAGEMENT_FAMILY = ("engagement_rate", "comment_like_ratio", "engagement_consistency")
"""Correlated Signals: only the weakest member contributes (ADR-0003).

These three measure substantially the same underlying property, so multiplying their
Likelihood Ratios triple-counts one piece of evidence. The weakest is taken rather than
the strongest because real engagement corroborates itself across all three, while bought
engagement inflates one and leaves the others thin.
"""

ONBOARD_LIFT = 2.0
"""Propensity must be >= 2x the Base Rate to justify Onboarding Capacity."""

PASS_LIFT = 1.0
"""Below the Base Rate, a Candidate who does not clear the Fit Floor is a pass."""

PASS_LIFT_HIGH_FIT = 0.5
"""A Candidate who clears the Fit Floor is passed only on actively bad evidence.

Absent evidence is not negative evidence. A well-fitting creator who simply has not sold
to their audience yet scores near the Base Rate through having nothing to show, and pass
means do not come back - so they are held and re-checked instead. Below half the Base
Rate the evidence is no longer absent but bad, and the pass applies to them too.

This asymmetry was found by the archetype_promising_unproven regression, which a
Fit-blind propensity threshold sent to pass on a knife-edge.
"""

# --- Fit ----------------------------------------------------------------------------

FIT_WEIGHTS: dict[str, float] = {
    "category_alignment": 35.0,
    "gcc_audience_share": 30.0,
    "audience_scale": 20.0,
    "language_fit": 15.0,
}
"""Category and audience dominate: they are what makes someone a Boutique Owner at all.

Scale is real but secondary - a mid-sized creator with the right audience beats a large
one with the wrong audience. Language is a genuine constraint in the GCC but the most
recoverable of the four.
"""

FIT_FLOOR = 50.0
"""Below this the Recommendation caps at hold, whatever the Propensity says."""

BRAND_SAFETY_VETO_BELOW = 0.5
"""Brand safety is a veto, not a weight (ADR-0001).

A weighted brand-safety term lets a large, well-aligned account buy its way past a real
problem. A serious flag is not twenty points off - it is a different answer.
"""

# --- Likelihood curves --------------------------------------------------------------


@dataclass(frozen=True)
class LikelihoodCurve:
    """Maps a Signal value to a Likelihood Ratio by interpolating in log-odds space."""

    signal: str
    points: tuple[tuple[float, float], ...]
    rationale: str
    monotonic: bool = True

    def likelihood_ratio(self, value: float) -> float:
        xs = [p[0] for p in self.points]
        ys = [math.log(p[1]) for p in self.points]
        if value <= xs[0]:
            lr = math.exp(ys[0])
        elif value >= xs[-1]:
            lr = math.exp(ys[-1])
        else:
            i = next(i for i in range(len(xs) - 1) if xs[i] <= value <= xs[i + 1])
            span = xs[i + 1] - xs[i]
            t = 0.0 if span == 0 else (value - xs[i]) / span
            lr = math.exp(ys[i] + t * (ys[i + 1] - ys[i]))
        return min(max(lr, LR_MIN), LR_MAX)


PROPENSITY_CURVES: dict[str, LikelihoodCurve] = {
    c.signal: c
    for c in (
        LikelihoodCurve(
            "engagement_rate",
            ((0.005, 0.5), (0.02, 0.9), (0.04, 1.3), (0.08, 1.8), (0.15, 2.0)),
            "Engagement per follower is the closest free proxy for whether an audience "
            "acts on what the Candidate says. Below ~0.5% an audience is functionally "
            "inert whatever its size.",
        ),
        LikelihoodCurve(
            "comment_like_ratio",
            ((0.002, 0.6), (0.01, 0.95), (0.03, 1.5), (0.06, 1.9)),
            "Comments cost more than likes. A high comment share indicates an audience "
            "that converses rather than scrolls, which is what converts to Attributed Sales.",
        ),
        LikelihoodCurve(
            "engagement_consistency",
            ((0.2, 0.7), (0.5, 1.0), (0.8, 1.4), (1.0, 1.6)),
            "Erratic engagement suggests reach driven by occasional virality rather than "
            "a reliable audience. A Boutique needs the reliable kind.",
        ),
        LikelihoodCurve(
            "posting_cadence",
            ((0.5, 0.5), (2.0, 0.9), (4.0, 1.4), (8.0, 1.6), (14.0, 1.2)),
            "Posts per week. Too few and a Boutique goes stale; past roughly daily the "
            "extra volume stops indicating commitment and starts indicating low-effort "
            "filler, so the curve turns back down.",
            monotonic=False,
        ),
        LikelihoodCurve(
            "authenticity_plausibility",
            ((0.0, 0.5), (0.4, 0.8), (0.7, 1.1), (1.0, 1.3)),
            "Whether engagement is plausible for the follower tier. Bought engagement is "
            "the main way the other Signals lie, so this mostly protects against them; "
            "its upside is capped because looking authentic is only table stakes.",
        ),
        LikelihoodCurve(
            "commercial_evidence",
            ((0.0, 0.6), (0.3, 1.0), (0.7, 1.7), (1.0, 2.0)),
            "Prior affiliate links, discount codes and sponsored posts are the only "
            "Signal that shows the audience has already been asked to buy something and "
            "did not leave. The strongest evidence available without real outcomes.",
        ),
        LikelihoodCurve(
            "selling_content_style",
            ((0.0, 0.7), (0.5, 1.1), (1.0, 1.6)),
            "Tutorials, hauls and reviews put products in front of an audience natively. "
            "Pure aesthetic or personality content monetises through brand deals instead, "
            "which does not transfer to running a Boutique.",
        ),
    )
}

FIT_SIGNALS = tuple(FIT_WEIGHTS) + ("brand_safety",)
PROPENSITY_SIGNALS = tuple(PROPENSITY_CURVES)

SIGNAL_CONSTRUCT: dict[str, Construct] = {
    **{name: Construct.FIT for name in FIT_SIGNALS},
    **{name: Construct.PROPENSITY for name in PROPENSITY_SIGNALS},
}

# --- Fit normalisation --------------------------------------------------------------

AUDIENCE_SCALE_POINTS: tuple[tuple[float, float], ...] = (
    (5_000, 0.0),
    (25_000, 0.35),
    (100_000, 0.7),
    (500_000, 0.95),
    (2_000_000, 1.0),
)
"""Follower count to a 0-1 Fit component, interpolated on a log scale.

Below 5k there is not enough audience to run a Boutique; above ~500k the marginal
follower stops mattering because reach is no longer the constraint.
"""

# --- Environment --------------------------------------------------------------------


def load_env_file(path: str | os.PathLike[str] = ".env") -> None:
    """Reads a .env file into the environment if one is present.

    The README tells an operator to copy .env.example to .env, so the file has to be read
    by something or that instruction is a dead end. A real environment variable always
    wins: an operator who exports a key for one command should not be overridden by a
    stale file. Kept dependency-free - the format we document is KEY=value.
    """
    try:
        content = Path(path).read_text()
    except OSError:
        return
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


# --- LLM extraction -----------------------------------------------------------------

load_env_file()

EXTRACTION_MODEL = os.environ.get("EXTRACTION_MODEL", "openai/gpt-5.2-2025-12-11")
"""Routed through litai. Note that litai 0.0.10 pins an older model list in its type
hints (openai/gpt-5, gpt-5-mini, gpt-4o, o3-mini) and does not name this one; the string
is passed straight through at runtime, so override it here or via EXTRACTION_MODEL if
the provider rejects it."""

EXTRACTION_MAX_TOKENS = 2000
"""litai defaults to 500, which truncates the extraction mid-JSON."""
