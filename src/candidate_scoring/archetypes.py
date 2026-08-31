"""Hand-built reference Candidates whose Recommendation is known in advance.

These are the regression suite for scoring behaviour: if a change to a weight or a
Likelihood Ratio flips one of these, the change needs an argument. Every value here is
authored, so every Archetype's qualitative Signals are MOCKED and the Report says so.

The negative cases live here deliberately: no real named person is published with an
unflattering assessment attached (see README, "On the demo Candidate").
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from .domain import Post, ProfileSnapshot, Provenance, Recommendation
from .signals.qualitative import ExtractionRecord, QualitativeExtraction

CAPTURED = date(2026, 8, 31)


@dataclass(frozen=True)
class Archetype:
    snapshot: ProfileSnapshot
    extraction: ExtractionRecord
    expected: Recommendation
    why: str


def _snapshot(
    handle: str,
    name: str,
    followers: int,
    engagement_rate: float,
    comment_share: float,
    wobble: float,
    per_week: float,
    bio: str,
    n_posts: int = 12,
) -> ProfileSnapshot:
    """Builds posts that reproduce the intended quantitative Signals under derive.py."""
    spacing = 7.0 / per_week
    posts = []
    for i in range(n_posts):
        # Alternating deviation gives an exact, seedless coefficient of variation.
        factor = 1.0 + (wobble if i % 2 == 0 else -wobble)
        engagement = engagement_rate * followers * factor
        comments = int(engagement * comment_share)
        posts.append(
            Post(
                posted_at=CAPTURED - timedelta(days=int(round(spacing * (i + 1)))),
                likes=int(engagement) - comments,
                comments=comments,
                caption=f"{handle} post {i + 1}",
                media_type="video" if i % 3 == 0 else "image",
            )
        )
    return ProfileSnapshot(
        handle=handle,
        platform="instagram",
        captured_at=CAPTURED,
        followers=followers,
        following=int(followers**0.4),
        post_count=n_posts * 30,
        bio=bio,
        verified=followers > 200_000,
        posts=tuple(posts),
        provenance=Provenance.MOCKED,
        display_name=name,
    )


def _extraction(**values: float | str) -> ExtractionRecord:
    defaults = dict(
        category_alignment=0.5,
        gcc_audience_share=0.5,
        language_fit=0.5,
        brand_safety=1.0,
        commercial_evidence=0.5,
        selling_content_style=0.5,
        brand_safety_note="",
        reasoning="Authored archetype value, not a model extraction.",
    )
    return ExtractionRecord(
        extraction=QualitativeExtraction(**{**defaults, **values}),
        provenance=Provenance.MOCKED,
    )


ARCHETYPES: tuple[Archetype, ...] = (
    Archetype(
        snapshot=_snapshot(
            "archetype_ideal_beauty",
            "The ideal Boutique Owner",
            followers=180_000,
            engagement_rate=0.055,
            comment_share=0.035,
            wobble=0.15,
            per_week=4.0,
            bio="Beauty & skincare | Kuwait | code SARA15",
        ),
        extraction=_extraction(
            category_alignment=0.95,
            gcc_audience_share=0.9,
            language_fit=0.95,
            commercial_evidence=0.85,
            selling_content_style=0.9,
        ),
        expected=Recommendation.ONBOARD,
        why="Everything a Boutique Owner should be. If this is not an onboard, the "
        "thresholds are wrong.",
    ),
    Archetype(
        snapshot=_snapshot(
            "archetype_wrong_vertical",
            "Large account, wrong category",
            followers=900_000,
            engagement_rate=0.06,
            comment_share=0.04,
            wobble=0.12,
            per_week=5.0,
            bio="Fitness coach | training programmes | worldwide",
        ),
        extraction=_extraction(
            category_alignment=0.1,
            gcc_audience_share=0.25,
            language_fit=0.4,
            commercial_evidence=0.8,
            selling_content_style=0.8,
        ),
        expected=Recommendation.HOLD,
        why="Excellent seller, wrong audience and category. The Fit Floor must stop "
        "size and engagement from buying an onboard.",
    ),
    Archetype(
        snapshot=_snapshot(
            "archetype_engagement_farm",
            "Bought engagement",
            followers=250_000,
            engagement_rate=0.35,
            comment_share=0.004,
            wobble=0.75,
            per_week=9.0,
            bio="Beauty lover | Gulf | DM for collabs",
        ),
        extraction=_extraction(
            category_alignment=0.8,
            gcc_audience_share=0.8,
            language_fit=0.8,
            commercial_evidence=0.2,
            selling_content_style=0.4,
        ),
        expected=Recommendation.HOLD,
        why="Implausible engagement, almost no comments, wildly erratic. Fit is genuinely "
        "good, so this must not be an onboard on the strength of fake numbers.",
    ),
    Archetype(
        snapshot=_snapshot(
            "archetype_brand_safety_flag",
            "Strong Candidate, safety concern",
            followers=400_000,
            engagement_rate=0.06,
            comment_share=0.03,
            wobble=0.1,
            per_week=4.0,
            bio="Beauty | Gulf | partnerships",
        ),
        extraction=_extraction(
            category_alignment=0.9,
            gcc_audience_share=0.85,
            language_fit=0.9,
            commercial_evidence=0.8,
            selling_content_style=0.85,
            brand_safety=0.2,
            brand_safety_note="Recent content incompatible with a family-facing retail brand.",
        ),
        expected=Recommendation.PASS,
        why="Would otherwise be a clear onboard. The veto has to beat every other Signal.",
    ),
    Archetype(
        snapshot=_snapshot(
            "archetype_promising_unproven",
            "Right person, no selling track record",
            followers=45_000,
            engagement_rate=0.045,
            comment_share=0.025,
            wobble=0.2,
            per_week=1.2,
            bio="Skincare & makeup | Riyadh",
        ),
        extraction=_extraction(
            category_alignment=0.85,
            gcc_audience_share=0.85,
            language_fit=0.9,
            commercial_evidence=0.05,
            selling_content_style=0.5,
        ),
        expected=Recommendation.HOLD,
        why="Exactly the Candidate the hold state exists for: right category and audience, "
        "no evidence yet that they can sell, and posting too rarely.",
    ),
)


class ArchetypeExtractor:
    """Serves the authored extractions. Used by the Archetype regression tests."""

    def __init__(self) -> None:
        self._by_handle = {a.snapshot.handle: a.extraction for a in ARCHETYPES}

    def extract(self, snapshot: ProfileSnapshot) -> ExtractionRecord:
        return self._by_handle[snapshot.handle]
