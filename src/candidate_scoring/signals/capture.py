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
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol

from ..domain import Post, ProfileSnapshot, Provenance

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "data" / "fixtures"
IG_ENDPOINT = "https://i.instagram.com/api/v1/users/web_profile_info/?username={handle}"
IG_APP_ID = "936619743392459"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


class Capturer(Protocol):
    def capture(self, handle: str) -> ProfileSnapshot: ...


class InstagramCapturer:
    """Reads the public web profile endpoint. Public data only; no login, no credentials."""

    def capture(self, handle: str) -> ProfileSnapshot:
        request = urllib.request.Request(
            IG_ENDPOINT.format(handle=handle),
            headers={"User-Agent": USER_AGENT, "X-IG-App-ID": IG_APP_ID},
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Instagram returned {exc.code} for @{handle}. Unauthenticated capture is "
                "rate limited and often blocked; use a recorded Snapshot instead."
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach Instagram: {exc.reason}") from exc

        user = (payload.get("data") or {}).get("user")
        if not user:
            raise RuntimeError(f"No public profile data returned for @{handle}")
        return _snapshot_from_ig(handle, user)


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
