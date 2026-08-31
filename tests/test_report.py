from datetime import date

from candidate_scoring.archetypes import ARCHETYPES
from candidate_scoring.domain import Recommendation
from candidate_scoring.pipeline import rank, score_snapshot
from candidate_scoring.report.render import (
    render_candidate,
    render_shortlist,
    to_dict,
    write_reports,
)


def _scored(archetype_extractor):
    return [score_snapshot(a.snapshot, archetype_extractor) for a in ARCHETYPES]


def test_an_onboard_report_still_shows_its_weakest_points(archetype_extractor):
    """The Report is an assessment, not a sales pitch - even for a Candidate with no flaws."""
    ideal = next(
        c for c in _scored(archetype_extractor) if c.recommendation is Recommendation.ONBOARD
    )
    reservations = [r for r in ideal.reasons if r.kind == "reservation"]
    assert reservations, "a Candidate strong everywhere still has a weakest link"
    html = render_candidate(ideal)
    assert "Weakest points" in html


def test_every_recommendation_gets_weakest_points(archetype_extractor):
    for candidate in _scored(archetype_extractor):
        assert [r for r in candidate.reasons if r.kind == "reservation"]


def test_mocked_signals_are_labelled_in_the_html(archetype_extractor):
    html = render_candidate(_scored(archetype_extractor)[0])
    assert 'class="tag mocked"' in html
    assert "Do not act on them" in html


def test_candidate_report_makes_snapshot_age_visible(archetype_extractor):
    candidate = _scored(archetype_extractor)[0]

    html = render_candidate(candidate, as_of=date(2026, 9, 3))

    assert f"snapshot {candidate.snapshot.captured_at} (3 days old)" in html


def test_candidate_report_calls_a_same_day_snapshot_current(archetype_extractor):
    candidate = _scored(archetype_extractor)[0]

    html = render_candidate(candidate, as_of=candidate.snapshot.captured_at)

    assert f"snapshot {candidate.snapshot.captured_at} (captured today)" in html


def test_candidate_report_makes_a_months_old_snapshot_conspicuous(archetype_extractor):
    candidate = _scored(archetype_extractor)[0]

    html = render_candidate(candidate, as_of=date(2026, 11, 30))

    assert f"snapshot {candidate.snapshot.captured_at} (91 days old)" in html


def test_write_reports_uses_one_pinned_date_for_every_page(archetype_extractor, tmp_path):
    candidate = _scored(archetype_extractor)[0]

    index = write_reports([candidate], tmp_path, as_of=date(2026, 9, 3))

    assert "generated 2026-09-03" in index.read_text()
    assert "snapshot 2026-08-31 (3 days old)" in (tmp_path / f"{candidate.handle}.html").read_text()


def test_every_report_carries_the_no_real_outcomes_caveat(archetype_extractor):
    for candidate in _scored(archetype_extractor):
        assert "No real onboarding outcomes" in render_candidate(candidate)


def test_shortlist_puts_onboards_above_holds(archetype_extractor):
    ordered = rank(_scored(archetype_extractor))
    seen_hold = False
    for candidate in ordered:
        if candidate.recommendation is Recommendation.HOLD:
            seen_hold = True
        if candidate.recommendation is Recommendation.ONBOARD:
            assert not seen_hold, "an onboard ranked below a hold"


def test_a_strong_seller_in_the_wrong_category_never_tops_the_shortlist(archetype_extractor):
    ordered = rank(_scored(archetype_extractor))
    assert ordered[0].snapshot.handle != "archetype_wrong_vertical"


def test_shortlist_renders_a_row_per_candidate(archetype_extractor):
    html = render_shortlist(rank(_scored(archetype_extractor)))
    for archetype in ARCHETYPES:
        assert archetype.snapshot.handle in html


def test_json_carries_the_suppressed_contributions_too(archetype_extractor):
    """The machine-readable form must not hide what the grouping rule discarded."""
    payload = to_dict(_scored(archetype_extractor)[0])
    suppressed = [
        c for c in payload["propensity"]["contributions"] if not c["applied"]
    ]
    assert suppressed
    assert all(c["suppressed_by"] for c in suppressed)
