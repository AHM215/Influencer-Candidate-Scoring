"""Domain types. Terms here are defined in CONTEXT.md; keep the two in step."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum


class Provenance(str, Enum):
    """Where a Signal's value came from. Surfaced on every line of the Candidate Report."""

    OBSERVED = "observed"
    INFERRED = "inferred"
    MOCKED = "mocked"


class Construct(str, Enum):
    """The two things this system scores. They are never blended (ADR-0001)."""

    FIT = "fit"
    PROPENSITY = "propensity"


class Recommendation(str, Enum):
    ONBOARD = "onboard"
    HOLD = "hold"
    PASS = "pass"


@dataclass(frozen=True)
class Post:
    posted_at: date
    likes: int
    comments: int
    caption: str
    media_type: str = "image"


@dataclass(frozen=True)
class ProfileSnapshot:
    """Raw public data for one Candidate at one point in time. Never scored directly."""

    handle: str
    platform: str
    captured_at: date
    followers: int
    following: int
    post_count: int
    bio: str
    verified: bool
    posts: tuple[Post, ...]
    external_url: str | None = None
    provenance: Provenance = Provenance.OBSERVED
    display_name: str = ""

    @property
    def label(self) -> str:
        return self.display_name or self.handle


@dataclass(frozen=True)
class SignalValue:
    name: str
    construct: Construct
    value: float
    provenance: Provenance
    detail: str = ""

    @property
    def is_evidence(self) -> bool:
        """Mocked values still score, but the Report must never present them as measured."""
        return self.provenance is not Provenance.MOCKED


@dataclass(frozen=True)
class FitContribution:
    signal: str
    weight: float
    value: float
    provenance: Provenance

    @property
    def points(self) -> float:
        return self.weight * self.value

    @property
    def forgone(self) -> float:
        return self.weight * (1.0 - self.value)


@dataclass(frozen=True)
class FitResult:
    score: float
    contributions: tuple[FitContribution, ...]
    vetoed: bool = False
    veto_reason: str = ""


@dataclass(frozen=True)
class PropensityContribution:
    signal: str
    value: float
    likelihood_ratio: float
    provenance: Provenance
    applied: bool
    suppressed_by: str = ""

    @property
    def log_lr(self) -> float:
        from math import log

        return log(self.likelihood_ratio)


@dataclass(frozen=True)
class PropensityResult:
    probability: float
    base_rate: float
    contributions: tuple[PropensityContribution, ...]
    clipped: bool = False

    @property
    def lift(self) -> float:
        """How many times the Base Rate this Candidate sits at. The policy reads this."""
        return self.probability / self.base_rate


@dataclass(frozen=True)
class ReasonCode:
    """One line of the Candidate Report.

    `strength` is a signed, roughly comparable score across both constructs, so strengths
    and reservations can be ranked against each other. `kind` is what the Report calls it:
    a Candidate strong everywhere still has a weakest point, and hiding it would make the
    Report a sales pitch.
    """

    construct: Construct
    signal: str
    positive: bool
    strength: float
    text: str
    provenance: Provenance
    kind: str = "strength"


@dataclass(frozen=True)
class ScoredCandidate:
    snapshot: ProfileSnapshot
    signals: tuple[SignalValue, ...]
    fit: FitResult
    propensity: PropensityResult
    recommendation: Recommendation
    policy_reason: str
    reasons: tuple[ReasonCode, ...] = ()

    @property
    def handle(self) -> str:
        return self.snapshot.handle

    def signal(self, name: str) -> SignalValue:
        for s in self.signals:
            if s.name == name:
                return s
        raise KeyError(name)
