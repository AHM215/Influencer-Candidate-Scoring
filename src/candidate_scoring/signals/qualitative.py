"""Qualitative Signals read out of bio and captions by an LLM.

The model reads text and returns structured values; it never sees a score and never
produces one. Scoring stays deterministic (ADR-0002), and the recorded fixtures mean the
whole system - tests included - runs offline with no API key.

The provider is OpenAI via litai. litai's chat() returns a plain string with no
structured-output mode, so the schema is enforced on our side: the prompt asks for JSON,
the response is parsed and validated against QualitativeExtraction, and one retry feeds
the validation error back to the model. A provider with native structured output would
let that layer go away.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError

from ..config import EXTRACTION_MAX_TOKENS, EXTRACTION_MODEL
from ..domain import Construct, ProfileSnapshot, Provenance, SignalValue

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "data" / "fixtures" / "extractions"

SYSTEM = """You read a social media profile and report observable properties of it.

You are assisting an assessment of whether a creator would suit running a beauty and \
fashion storefront for a GCC e-commerce business. You do not score the creator, rank \
them, or recommend anything - you report what their content shows, and a separate \
deterministic model does the scoring.

Every numeric field is a 0-1 fraction. Be calibrated rather than generous: 0.5 means \
genuinely middling, and reserve values above 0.9 for cases with strong evidence in the \
text. When the profile gives you little to go on, say so in the reasoning and stay near \
the middle rather than guessing high or low.

Reply with a single JSON object and nothing else. No prose, no code fences."""


class QualitativeExtraction(BaseModel):
    """What the model returns. Field descriptions are the extraction spec."""

    category_alignment: float = Field(
        ge=0, le=1, description="How far the content sits in beauty, fragrance or fashion."
    )
    gcc_audience_share: float = Field(
        ge=0,
        le=1,
        description=(
            "Estimated share of the audience in the GCC, judged from caption language, "
            "dialect, locations, brands and cultural references. An estimate, not a "
            "measurement: audience geography is not visible on a public profile."
        ),
    )
    language_fit: float = Field(
        ge=0,
        le=1,
        description="Fit of the Arabic/English mix for a GCC beauty storefront audience.",
    )
    brand_safety: float = Field(
        ge=0,
        le=1,
        description=(
            "1.0 is unproblematic for a mainstream family-facing retail brand. Below 0.5 "
            "triggers an automatic rejection, so reserve it for real concerns."
        ),
    )
    commercial_evidence: float = Field(
        ge=0,
        le=1,
        description=(
            "Evidence the creator has already sold to this audience: affiliate links, "
            "discount codes, sponsored posts, brand partnerships, a shop link in bio."
        ),
    )
    selling_content_style: float = Field(
        ge=0,
        le=1,
        description=(
            "How far the content format suits selling products natively - tutorials, "
            "hauls, reviews and get-ready-with-me score high; pure aesthetic or "
            "personality content scores low."
        ),
    )
    brand_safety_note: str = Field(
        default="", description="One sentence on any brand safety concern, or empty if none."
    )
    reasoning: str = Field(
        default="", description="Two or three sentences on what drove these values."
    )


@dataclass(frozen=True)
class ExtractionRecord:
    """An extraction plus where it came from.

    A value the model read out of real captions is INFERRED. A value a human wrote by
    hand - every Archetype, and any Candidate whose extraction was never recorded - is
    MOCKED, and the Report says so. Conflating the two would be the exact overstated
    claim this system is meant to avoid.
    """

    extraction: QualitativeExtraction
    provenance: Provenance
    model: str = ""
    """Which model produced this, so an operator can tell which extractions predate a
    model change. Empty for a hand-authored extraction, which no model produced."""


class ModelPreflightStatus(StrEnum):
    MISSING_CREDENTIAL = "missing_credential"
    MODEL_REJECTED = "model_rejected"
    USABLE = "usable"


@dataclass(frozen=True)
class ModelPreflight:
    status: ModelPreflightStatus
    model: str
    detail: str = ""


def preflight_extraction_model(
    model: str = EXTRACTION_MODEL,
    api_key: str | None = None,
    chat: Callable[..., str] | None = None,
) -> ModelPreflight:
    """Checks the configured model without risking an extraction on a fallback model."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        return ModelPreflight(ModelPreflightStatus.MISSING_CREDENTIAL, model)

    if chat is None:
        from litai import LLM

        chat = LLM(model=model, fallback_models=[], max_retries=1, api_key=key).chat

    try:
        chat("Reply with OK.", max_tokens=1)
    except Exception as exc:
        return ModelPreflight(ModelPreflightStatus.MODEL_REJECTED, model, str(exc))
    return ModelPreflight(ModelPreflightStatus.USABLE, model)


class Extractor(Protocol):
    def extract(self, snapshot: ProfileSnapshot) -> ExtractionRecord: ...


class FixtureExtractor:
    """Replays a recorded extraction. The default everywhere: tests, CI, demos."""

    def __init__(self, fixture_dir: Path = FIXTURE_DIR) -> None:
        self.fixture_dir = fixture_dir

    def extract(self, snapshot: ProfileSnapshot) -> ExtractionRecord:
        path = self.fixture_dir / f"{snapshot.handle}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"No extraction for {snapshot.handle!r} at {path}. "
                "Record one with `score extract <handle>` (needs OPENAI_API_KEY)."
            )
        payload = json.loads(path.read_text())
        source = payload.pop("source", "model")
        model = payload.pop("model", "")
        return ExtractionRecord(
            extraction=QualitativeExtraction.model_validate(payload),
            provenance=Provenance.MOCKED if source == "authored" else Provenance.INFERRED,
            model=model,
        )


class OpenAIExtractor:
    """Calls OpenAI through litai. Used to record fixtures, not in the scoring path."""

    def __init__(
        self,
        model: str = EXTRACTION_MODEL,
        record_to: Path | None = FIXTURE_DIR,
        api_key: str | None = None,
        chat: Callable[..., str] | None = None,
    ) -> None:
        if chat is None:
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY is not set; see .env.example")
            from litai import LLM

            chat = LLM(model=model, api_key=key).chat
        self.chat = chat
        self.model = model
        self.record_to = record_to

    def extract(self, snapshot: ProfileSnapshot) -> ExtractionRecord:
        prompt = f"{render_profile(snapshot)}\n\n{schema_instruction()}"
        extraction = self._ask(prompt)
        if self.record_to is not None:
            self.record_to.mkdir(parents=True, exist_ok=True)
            (self.record_to / f"{snapshot.handle}.json").write_text(
                json.dumps(
                    {"source": "model", "model": self.model, **extraction.model_dump()},
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n"
            )
        return ExtractionRecord(
            extraction=extraction, provenance=Provenance.INFERRED, model=self.model
        )

    def _ask(self, prompt: str, attempts: int = 2) -> QualitativeExtraction:
        last_error = ""
        for attempt in range(attempts):
            body = prompt
            if attempt:
                body += (
                    f"\n\nYour last reply failed validation: {last_error}"
                    "\nReturn corrected JSON only."
                )
            try:
                raw = self.chat(
                    body, system_prompt=SYSTEM, max_tokens=EXTRACTION_MAX_TOKENS
                )
            except Exception as exc:  # litai wraps provider errors; surface the text
                raise RuntimeError(f"Extraction call failed on {self.model}: {exc}") from exc
            try:
                return QualitativeExtraction.model_validate(parse_json(raw))
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)[:400]
        raise RuntimeError(
            f"{self.model} did not return valid extraction JSON after {attempts} attempts: "
            f"{last_error}"
        )


def parse_json(raw: str) -> dict:
    """Pulls the JSON object out of a text reply, tolerating fences and stray prose."""
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"No JSON object in reply: {raw[:200]!r}")
    return json.loads(text[start : end + 1])


def schema_instruction() -> str:
    """The field list, generated from the model so prompt and schema cannot drift."""
    lines = ["Return a JSON object with exactly these keys:"]
    for name, field in QualitativeExtraction.model_fields.items():
        kind = "0-1 number" if field.annotation is float else "string"
        lines.append(f'  "{name}": {kind} - {field.description}')
    return "\n".join(lines)


def render_profile(snapshot: ProfileSnapshot) -> str:
    """The profile as text. Deterministic, so a re-record is comparable to the last one."""
    lines = [
        f"Platform: {snapshot.platform}",
        f"Handle: @{snapshot.handle}",
        f"Followers: {snapshot.followers:,}  Following: {snapshot.following:,}"
        f"  Posts: {snapshot.post_count:,}",
        f"Verified: {snapshot.verified}",
        f"Bio: {snapshot.bio}",
        f"Link in bio: {snapshot.external_url or 'none'}",
        "",
        "Recent posts:",
    ]
    for post in snapshot.posts:
        lines.append(
            f"- [{post.posted_at.isoformat()}] {post.media_type}, "
            f"{post.likes:,} likes, {post.comments:,} comments\n"
            f"  {post.caption}"
        )
    return "\n".join(lines)


def to_signals(record: ExtractionRecord) -> dict[str, SignalValue]:
    """Nothing here is measured: it is read out of text, or authored by hand."""
    extraction = record.extraction
    spec = {
        "category_alignment": Construct.FIT,
        "gcc_audience_share": Construct.FIT,
        "language_fit": Construct.FIT,
        "brand_safety": Construct.FIT,
        "commercial_evidence": Construct.PROPENSITY,
        "selling_content_style": Construct.PROPENSITY,
    }
    return {
        name: SignalValue(
            name=name,
            construct=construct,
            value=getattr(extraction, name),
            provenance=record.provenance,
            detail=extraction.brand_safety_note if name == "brand_safety" else "",
        )
        for name, construct in spec.items()
    }
