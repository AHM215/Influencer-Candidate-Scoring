"""Capturing a Profile Snapshot from a live public profile.

This is the adapter boundary. The Instagram implementation reads one public endpoint and
is deliberately not in the scoring path: it is fragile, unauthenticated, and rate
limited, so Snapshots are captured once, written to data/fixtures/ and replayed from
there. A paid profile API would slot in here as another Capturer with no change
anywhere else - and would additionally supply the audience demographics that
gcc_audience_share currently has to infer.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol

from ..domain import Post, ProfileSnapshot, Provenance

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "data" / "fixtures"
IG_ENDPOINT = "https://i.instagram.com/api/v1/users/web_profile_info/?username={handle}"
IG_APP_ID = "936619743392459"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
MAX_CAPTURE_ATTEMPTS = 3
"""Few enough to stay polite. A source that refuses three spaced requests is not having a
momentary blip, and continuing to ask is how a caller gets blocked outright."""

MAX_CAPTURE_ELAPSED_SECONDS = 30.0
"""An operator is waiting at a terminal. Bound the wall clock as well as the attempts, so a
slow refusal cannot outlast their patience however the backoff lands."""

RETRY_DELAY_SECONDS = 1.0
RETRY_JITTER_SECONDS = 0.25
"""Jitter keeps repeated runs from retrying in lockstep against the same source."""


class Capturer(Protocol):
    def capture(self, handle: str) -> ProfileSnapshot: ...


class RateLimitExhaustedError(RuntimeError):
    """Instagram continued to rate limit capture within its polite retry bounds."""


class ProfileUnavailableError(RuntimeError):
    """The requested profile is missing, private, or unavailable to public capture."""


class TransportUnreachableError(RuntimeError):
    """Instagram could not be reached over the network."""


class CaptureRefusedError(RuntimeError):
    """Instagram refused the request for a reason retrying will not fix.

    An unauthenticated caller is commonly met with 401 or 403 once it is blocked, which
    is a refusal rather than a transient rate limit: retrying burns the budget without
    improving the odds. The operator still needs the recorded-Snapshot pointer.
    """


def _open_instagram(request: urllib.request.Request) -> Any:
    return urllib.request.urlopen(request, timeout=20)


class InstagramCapturer:
    """Reads the public web profile endpoint. Public data only; no login, no credentials."""

    def __init__(
        self,
        transport: Callable[[urllib.request.Request], Any] = _open_instagram,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.transport = transport
        self.sleep = sleep
        self.clock = clock

    def capture(self, handle: str) -> ProfileSnapshot:
        request = urllib.request.Request(
            IG_ENDPOINT.format(handle=handle),
            headers={"User-Agent": USER_AGENT, "X-IG-App-ID": IG_APP_ID},
        )
        started_at = self.clock()
        attempts = 0

        while attempts < MAX_CAPTURE_ATTEMPTS:
            if attempts and self.clock() - started_at >= MAX_CAPTURE_ELAPSED_SECONDS:
                raise _rate_limit_error(handle, attempts)
            attempts += 1
            try:
                with self.transport(request) as response:
                    payload = json.loads(response.read())
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    raise ProfileUnavailableError(
                        f"Instagram returned 404 for @{handle}; the profile may be "
                        "missing or private."
                    ) from exc
                if exc.code != 429:
                    raise CaptureRefusedError(
                        f"Instagram returned {exc.code} for @{handle} and will not be "
                        "retried. Use a recorded Profile Snapshot instead."
                    ) from exc
                if attempts == MAX_CAPTURE_ATTEMPTS:
                    raise _rate_limit_error(handle, attempts) from exc

                delay = RETRY_DELAY_SECONDS * 2 ** (attempts - 1)
                delay += random.uniform(0, RETRY_JITTER_SECONDS)
                if self.clock() - started_at + delay > MAX_CAPTURE_ELAPSED_SECONDS:
                    raise _rate_limit_error(handle, attempts) from exc
                self.sleep(delay)
                continue
            except urllib.error.URLError as exc:
                raise TransportUnreachableError(f"Could not reach Instagram: {exc.reason}") from exc

            user = (payload.get("data") or {}).get("user")
            if not user:
                raise ProfileUnavailableError(f"No public profile data returned for @{handle}")
            return _snapshot_from_ig(handle, user)

        raise _rate_limit_error(handle, attempts)


def _rate_limit_error(handle: str, attempts: int) -> RateLimitExhaustedError:
    return RateLimitExhaustedError(
        f"Instagram kept rate limiting @{handle} after {attempts} attempts. "
        "Use a recorded Profile Snapshot instead."
    )


def _snapshot_from_ig(handle: str, user: dict) -> ProfileSnapshot:
    edges = (user.get("edge_owner_to_timeline_media") or {}).get("edges", [])
    posts = []
    for edge in edges:
        node = edge.get("node", {})
        caption_edges = (node.get("edge_media_to_caption") or {}).get("edges", [])
        caption = caption_edges[0]["node"]["text"] if caption_edges else ""
        posts.append(
            Post(
                posted_at=datetime.fromtimestamp(
                    node.get("taken_at_timestamp", 0), tz=UTC
                ).date(),
                likes=(node.get("edge_liked_by") or {}).get("count", 0),
                comments=(node.get("edge_media_to_comment") or {}).get("count", 0),
                caption=caption.strip(),
                media_type="video" if node.get("is_video") else "image",
            )
        )
    return ProfileSnapshot(
        handle=handle,
        platform="instagram",
        captured_at=date.today(),
        followers=(user.get("edge_followed_by") or {}).get("count", 0),
        following=(user.get("edge_follow") or {}).get("count", 0),
        post_count=(user.get("edge_owner_to_timeline_media") or {}).get("count", 0),
        bio=(user.get("biography") or "").strip(),
        verified=bool(user.get("is_verified")),
        posts=tuple(posts),
        external_url=user.get("external_url"),
        provenance=Provenance.OBSERVED,
        display_name=(user.get("full_name") or "").strip(),
    )


def save_snapshot(snapshot: ProfileSnapshot, directory: Path = FIXTURE_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{snapshot.handle}.json"
    payload = asdict(snapshot)
    payload["captured_at"] = snapshot.captured_at.isoformat()
    payload["provenance"] = snapshot.provenance.value
    payload["posts"] = [
        {**asdict(p), "posted_at": p.posted_at.isoformat()} for p in snapshot.posts
    ]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


def load_snapshot(path: Path) -> ProfileSnapshot:
    payload = json.loads(Path(path).read_text())
    posts = tuple(
        Post(
            posted_at=date.fromisoformat(p["posted_at"]),
            likes=p["likes"],
            comments=p["comments"],
            caption=p["caption"],
            media_type=p.get("media_type", "image"),
        )
        for p in payload.pop("posts", [])
    )
    payload["captured_at"] = date.fromisoformat(payload["captured_at"])
    payload["provenance"] = Provenance(payload.get("provenance", "observed"))
    return ProfileSnapshot(posts=posts, **payload)
