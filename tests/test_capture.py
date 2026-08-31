import json
import urllib.error

import pytest

from candidate_scoring.signals import capture
from candidate_scoring.signals.capture import (
    MAX_CAPTURE_ATTEMPTS,
    RETRY_DELAY_SECONDS,
    InstagramCapturer,
    ProfileUnavailableError,
    RateLimitExhaustedError,
    TransportUnreachableError,
)


class CannedResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_capture_retries_a_rate_limit_then_returns_a_snapshot(monkeypatch):
    monkeypatch.setattr(capture.random, "uniform", lambda *_: 0)
    requests = []
    sleeps = []
    outcomes = [_http_error(429), CannedResponse({"data": {"user": _user()}})]

    def transport(request):
        requests.append(request)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    snapshot = InstagramCapturer(transport=transport, sleep=sleeps.append, clock=lambda: 0).capture(
        "someone"
    )

    assert snapshot.handle == "someone"
    assert len(requests) == 2
    assert sleeps == [RETRY_DELAY_SECONDS]


def test_capture_stops_after_the_configured_rate_limit_attempts(monkeypatch):
    monkeypatch.setattr(capture.random, "uniform", lambda *_: 0)
    requests = []
    sleeps = []

    def transport(request):
        requests.append(request)
        raise _http_error(429)

    capturer = InstagramCapturer(transport=transport, sleep=sleeps.append, clock=lambda: 0)

    with pytest.raises(RateLimitExhaustedError, match="recorded Profile Snapshot"):
        capturer.capture("someone")

    assert len(requests) == MAX_CAPTURE_ATTEMPTS
    assert sleeps == [RETRY_DELAY_SECONDS, RETRY_DELAY_SECONDS * 2]


def test_capture_stops_when_the_rate_limit_delay_would_exceed_its_time_budget(monkeypatch):
    monkeypatch.setattr(capture, "MAX_CAPTURE_ELAPSED_SECONDS", 0.5)
    monkeypatch.setattr(capture.random, "uniform", lambda *_: 0)
    requests = []
    sleeps = []

    def transport(request):
        requests.append(request)
        raise _http_error(429)

    capturer = InstagramCapturer(transport=transport, sleep=sleeps.append, clock=lambda: 0)

    with pytest.raises(RateLimitExhaustedError, match="recorded Profile Snapshot"):
        capturer.capture("someone")

    assert len(requests) == 1
    assert sleeps == []


def test_capture_reports_a_missing_or_private_profile_without_retrying():
    requests = []
    sleeps = []

    def transport(request):
        requests.append(request)
        raise _http_error(404)

    capturer = InstagramCapturer(transport=transport, sleep=sleeps.append, clock=lambda: 0)

    with pytest.raises(ProfileUnavailableError, match="missing or private"):
        capturer.capture("someone")

    assert len(requests) == 1
    assert sleeps == []


def test_capture_reports_a_payload_without_a_user_as_an_unavailable_profile():
    sleeps = []
    capturer = InstagramCapturer(
        transport=lambda _: CannedResponse({"data": {}}), sleep=sleeps.append, clock=lambda: 0
    )

    with pytest.raises(ProfileUnavailableError, match="No public profile data"):
        capturer.capture("someone")

    assert sleeps == []


def test_capture_reports_an_unreachable_transport_without_retrying():
    requests = []
    sleeps = []

    def transport(request):
        requests.append(request)
        raise urllib.error.URLError("offline")

    capturer = InstagramCapturer(transport=transport, sleep=sleeps.append, clock=lambda: 0)

    with pytest.raises(TransportUnreachableError, match="offline"):
        capturer.capture("someone")

    assert len(requests) == 1
    assert sleeps == []


def _http_error(code):
    return urllib.error.HTTPError("https://example.com", code, "error", None, None)


def _user():
    return {"edge_owner_to_timeline_media": {"edges": []}}
