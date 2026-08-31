import pytest

from candidate_scoring.archetypes import ARCHETYPES, ArchetypeExtractor
from candidate_scoring.cohort import generate_cohort
from candidate_scoring.domain import Construct, Provenance, SignalValue


@pytest.fixture(scope="session")
def cohort():
    return generate_cohort(200, seed=7)


@pytest.fixture(scope="session")
def archetype_extractor():
    return ArchetypeExtractor()


@pytest.fixture
def archetypes():
    return ARCHETYPES


def signals(**overrides) -> dict[str, SignalValue]:
    """A neutral, middling Candidate; override the Signals a test cares about."""
    from candidate_scoring.config import SIGNAL_CONSTRUCT

    defaults = {
        "category_alignment": 0.5, "gcc_audience_share": 0.5, "language_fit": 0.5,
        "audience_scale": 0.5, "brand_safety": 1.0, "engagement_rate": 0.02,
        "comment_like_ratio": 0.01, "engagement_consistency": 0.5,
        "posting_cadence": 2.0, "authenticity_plausibility": 1.0,
        "commercial_evidence": 0.3, "selling_content_style": 0.5,
    }
    defaults.update(overrides)
    return {
        name: SignalValue(
            name, SIGNAL_CONSTRUCT.get(name, Construct.FIT), value, Provenance.MOCKED
        )
        for name, value in defaults.items()
    }
