"""Archetype regression: if a weight change flips one of these, it needs an argument."""

import pytest

from candidate_scoring.archetypes import ARCHETYPES
from candidate_scoring.domain import Provenance
from candidate_scoring.pipeline import score_snapshot


@pytest.mark.parametrize("archetype", ARCHETYPES, ids=lambda a: a.snapshot.handle)
def test_archetype_lands_on_its_expected_recommendation(archetype, archetype_extractor):
    scored = score_snapshot(archetype.snapshot, archetype_extractor)
    assert scored.recommendation is archetype.expected, archetype.why


@pytest.mark.parametrize("archetype", ARCHETYPES, ids=lambda a: a.snapshot.handle)
def test_archetype_signals_are_never_presented_as_real(archetype, archetype_extractor):
    scored = score_snapshot(archetype.snapshot, archetype_extractor)
    qualitative = [s for s in scored.signals if s.name == "category_alignment"]
    assert all(s.provenance is Provenance.MOCKED for s in qualitative)


def test_the_farm_and_the_ideal_are_separated_by_propensity_not_fit(archetype_extractor):
    """Both fit the category well. Only behaviour tells them apart."""
    by_handle = {a.snapshot.handle: a for a in ARCHETYPES}
    ideal = score_snapshot(by_handle["archetype_ideal_beauty"].snapshot, archetype_extractor)
    farm = score_snapshot(by_handle["archetype_engagement_farm"].snapshot, archetype_extractor)
    assert abs(ideal.fit.score - farm.fit.score) < 15
    assert ideal.propensity.probability > farm.propensity.probability * 3
