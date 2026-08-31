import json

import pytest

from candidate_scoring.domain import Provenance
from candidate_scoring.signals.capture import _snapshot_from_ig
from candidate_scoring.signals.qualitative import (
    FixtureExtractor,
    QualitativeExtraction,
    parse_json,
    schema_instruction,
    to_signals,
)


@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        'Here you go:\n{"a": 1}\nHope that helps!',
        '  \n {"a": 1} \n ',
    ],
)
def test_json_survives_the_shapes_a_chat_model_actually_returns(raw):
    """litai has no structured-output mode, so parsing is our problem."""
    assert parse_json(raw) == {"a": 1}


def test_unparseable_reply_raises_rather_than_guessing():
    with pytest.raises(ValueError):
        parse_json("I cannot help with that request.")


def test_the_prompt_schema_is_generated_from_the_model():
    """Prompt and schema cannot drift apart, because one is built from the other."""
    instruction = schema_instruction()
    for name in QualitativeExtraction.model_fields:
        assert f'"{name}"' in instruction


def test_a_recorded_extraction_is_inferred_not_measured(tmp_path):
    payload = {"source": "model", "category_alignment": 0.9, "gcc_audience_share": 0.8,
               "language_fit": 0.9, "brand_safety": 1.0, "commercial_evidence": 0.7,
               "selling_content_style": 0.8}
    (tmp_path / "someone.json").write_text(json.dumps(payload))
    record = FixtureExtractor(tmp_path).extract(_fake_snapshot("someone"))
    assert record.provenance is Provenance.INFERRED
    assert all(s.provenance is Provenance.INFERRED for s in to_signals(record).values())


def test_a_hand_authored_extraction_is_mocked_not_inferred(tmp_path):
    """Authoring values by hand must never be presentable as a model reading real text."""
    payload = {"source": "authored", "category_alignment": 0.9, "gcc_audience_share": 0.8,
               "language_fit": 0.9, "brand_safety": 1.0, "commercial_evidence": 0.7,
               "selling_content_style": 0.8}
    (tmp_path / "someone.json").write_text(json.dumps(payload))
    record = FixtureExtractor(tmp_path).extract(_fake_snapshot("someone"))
    assert record.provenance is Provenance.MOCKED


def test_missing_extraction_names_the_command_that_fixes_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="score extract"):
        FixtureExtractor(tmp_path).extract(_fake_snapshot("nobody"))


def test_instagram_payload_maps_onto_a_snapshot():
    user = {
        "edge_followed_by": {"count": 120000}, "edge_follow": {"count": 400},
        "biography": "Beauty | Kuwait", "is_verified": True, "full_name": "Someone",
        "external_url": "https://example.com",
        "edge_owner_to_timeline_media": {
            "count": 900,
            "edges": [{"node": {
                "taken_at_timestamp": 1750000000, "is_video": True,
                "edge_liked_by": {"count": 5000}, "edge_media_to_comment": {"count": 120},
                "edge_media_to_caption": {"edges": [{"node": {"text": "  hello  "}}]},
            }}],
        },
    }
    snapshot = _snapshot_from_ig("someone", user)
    assert snapshot.followers == 120000
    assert snapshot.posts[0].caption == "hello"
    assert snapshot.posts[0].media_type == "video"
    assert snapshot.provenance is Provenance.OBSERVED


def _fake_snapshot(handle: str):
    from datetime import date

    from candidate_scoring.domain import ProfileSnapshot

    return ProfileSnapshot(
        handle=handle, platform="instagram", captured_at=date(2026, 8, 31),
        followers=1000, following=100, post_count=10, bio="", verified=False, posts=(),
    )


def test_provenance_never_improves_as_values_flow_downstream():
    """A value derived from a mocked Snapshot is mocked, however it was computed."""
    from datetime import date

    from candidate_scoring.archetypes import ArchetypeExtractor
    from candidate_scoring.domain import Post, ProfileSnapshot
    from candidate_scoring.signals import build_signals

    mocked = ProfileSnapshot(
        handle="archetype_ideal_beauty", platform="instagram", captured_at=date(2026, 8, 31),
        followers=100_000, following=500, post_count=100, bio="", verified=False,
        posts=tuple(
            Post(posted_at=date(2026, 8, i + 1), likes=4000, comments=120, caption="x")
            for i in range(6)
        ),
        provenance=Provenance.MOCKED,
    )
    signals = build_signals(mocked, ArchetypeExtractor())
    assert all(s.provenance is Provenance.MOCKED for s in signals.values())


def test_an_observed_snapshot_keeps_the_finer_grained_tags():
    from datetime import date

    from candidate_scoring.archetypes import ArchetypeExtractor
    from candidate_scoring.domain import Post, ProfileSnapshot
    from candidate_scoring.signals import build_signals

    observed = ProfileSnapshot(
        handle="archetype_ideal_beauty", platform="instagram", captured_at=date(2026, 8, 31),
        followers=100_000, following=500, post_count=100, bio="", verified=False,
        posts=tuple(
            Post(posted_at=date(2026, 8, i + 1), likes=4000, comments=120, caption="x")
            for i in range(6)
        ),
        provenance=Provenance.OBSERVED,
    )
    signals = build_signals(observed, ArchetypeExtractor())
    assert signals["engagement_rate"].provenance is Provenance.OBSERVED
    assert signals["authenticity_plausibility"].provenance is Provenance.INFERRED
