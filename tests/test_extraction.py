import json

import pytest

from candidate_scoring.domain import Provenance
from candidate_scoring.signals.capture import _snapshot_from_ig
from candidate_scoring.signals.qualitative import (
    FixtureExtractor,
    ModelPreflightStatus,
    OpenAIExtractor,
    QualitativeExtraction,
    parse_json,
    preflight_extraction_model,
    schema_instruction,
    to_signals,
)


def _reply(**overrides):
    payload = {
        "category_alignment": 0.9,
        "gcc_audience_share": 0.8,
        "language_fit": 0.9,
        "brand_safety": 1.0,
        "commercial_evidence": 0.7,
        "selling_content_style": 0.8,
    }
    payload.update(overrides)
    return json.dumps(payload)


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


def test_a_model_extraction_records_its_model_and_round_trips_as_inferred(tmp_path):
    extractor = OpenAIExtractor(
        model="test-model", record_to=tmp_path, chat=lambda *_args, **_kwargs: _reply()
    )
    extractor.extract(_fake_snapshot("someone"))

    payload = json.loads((tmp_path / "someone.json").read_text())
    record = FixtureExtractor(tmp_path).extract(_fake_snapshot("someone"))

    assert payload["model"] == "test-model"
    assert payload["source"] == "model"
    assert record.provenance is Provenance.INFERRED


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


def test_an_injected_chat_call_extracts_a_well_formed_reply():
    extractor = OpenAIExtractor(record_to=None, chat=lambda *_args, **_kwargs: _reply())

    record = extractor.extract(_fake_snapshot("someone"))

    assert record.extraction.category_alignment == 0.9
    assert record.provenance is Provenance.INFERRED


@pytest.mark.parametrize(
    "reply",
    [
        f"```json\n{_reply()}\n```",
        f"Here is the extraction:\n{_reply()}\nI hope this helps.",
    ],
)
def test_an_injected_chat_call_recovers_json_wrapped_by_the_model(reply):
    extractor = OpenAIExtractor(record_to=None, chat=lambda *_args, **_kwargs: reply)

    assert extractor.extract(_fake_snapshot("someone")).extraction.brand_safety == 1.0


def test_validation_failure_retries_with_the_validation_error():
    prompts = []
    replies = iter(['{"category_alignment": 1.2}', _reply()])

    def chat(prompt, **_kwargs):
        prompts.append(prompt)
        return next(replies)

    extractor = OpenAIExtractor(record_to=None, chat=chat)

    assert extractor.extract(_fake_snapshot("someone")).extraction.category_alignment == 0.9
    assert len(prompts) == 2
    assert "validation" in prompts[1]
    assert "less than or equal to 1" in prompts[1]


@pytest.mark.parametrize(
    "replies",
    [
        ("not JSON", "still not JSON"),
        ('{"category_alignment": 1.2}', '{"category_alignment": -0.1}'),
    ],
)
def test_two_unrecoverable_replies_raise_without_inventing_a_signal(replies):
    responses = iter(replies)
    extractor = OpenAIExtractor(record_to=None, chat=lambda *_args, **_kwargs: next(responses))

    with pytest.raises(RuntimeError, match="did not return valid extraction JSON"):
        extractor.extract(_fake_snapshot("someone"))


def test_out_of_range_signal_value_is_rejected_not_clamped():
    extractor = OpenAIExtractor(
        record_to=None,
        chat=lambda *_args, **_kwargs: _reply(category_alignment=1.2),
    )

    with pytest.raises(RuntimeError, match="less than or equal to 1"):
        extractor.extract(_fake_snapshot("someone"))


def test_a_real_client_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        OpenAIExtractor(record_to=None)


def test_model_preflight_reports_a_missing_credential_without_calling_chat(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = preflight_extraction_model(chat=lambda *_args, **_kwargs: pytest.fail("called"))

    assert result.status is ModelPreflightStatus.MISSING_CREDENTIAL


def test_model_preflight_reports_when_the_provider_rejects_the_configured_model():
    def chat(*_args, **_kwargs):
        raise RuntimeError("model_not_found")

    result = preflight_extraction_model(model="openai/new-model", api_key="test-key", chat=chat)

    assert result.status is ModelPreflightStatus.MODEL_REJECTED
    assert result.model == "openai/new-model"
    assert result.detail == "model_not_found"


def test_model_preflight_reports_a_usable_configured_model():
    calls = []

    def chat(*args, **kwargs):
        calls.append((args, kwargs))
        return "OK"

    result = preflight_extraction_model(model="openai/new-model", api_key="test-key", chat=chat)

    assert result.status is ModelPreflightStatus.USABLE
    assert result.model == "openai/new-model"
    assert len(calls) == 1
    assert calls[0][1]["max_tokens"] == 1, "the probe checks reachability, not quality"


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


def test_the_recorded_model_identifier_survives_being_read_back(tmp_path):
    """Recording which model produced an extraction is pointless if loading discards it."""
    OpenAIExtractor(
        model="test-model", record_to=tmp_path, chat=lambda *_a, **_k: _reply()
    ).extract(_fake_snapshot("someone"))

    assert FixtureExtractor(tmp_path).extract(_fake_snapshot("someone")).model == "test-model"


def test_a_hand_authored_extraction_names_no_model(tmp_path):
    payload = {"source": "authored", "category_alignment": 0.9, "gcc_audience_share": 0.8,
               "language_fit": 0.9, "brand_safety": 1.0, "commercial_evidence": 0.7,
               "selling_content_style": 0.8}
    (tmp_path / "someone.json").write_text(json.dumps(payload))

    record = FixtureExtractor(tmp_path).extract(_fake_snapshot("someone"))

    assert record.model == "", "no model produced it, so none may be claimed"
