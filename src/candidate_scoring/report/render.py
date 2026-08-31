"""HTML rendering. Deterministic: every string comes from the scored objects."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..config import FIT_FLOOR
from ..domain import ScoredCandidate, SignalValue

TEMPLATES = Path(__file__).parent / "templates"
PERCENT_SIGNALS = {
    "engagement_rate",
    "comment_like_ratio",
    "category_alignment",
    "gcc_audience_share",
    "language_fit",
    "brand_safety",
    "commercial_evidence",
    "selling_content_style",
    "engagement_consistency",
    "authenticity_plausibility",
    "audience_scale",
}


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATES),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _css() -> str:
    return (TEMPLATES / "base.css").read_text()


def _format(signal: SignalValue) -> str:
    if signal.name == "posting_cadence":
        return f"{signal.value:.1f}/wk"
    if signal.name in PERCENT_SIGNALS:
        return f"{signal.value:.1%}"
    return f"{signal.value:.2f}"


def _lead_reason(candidate: ScoredCandidate) -> str:
    """The single line a CM lead sees in the Shortlist: why this Candidate, or why not."""
    wanted = "strength" if candidate.recommendation.value == "onboard" else "reservation"
    decisive = [r for r in candidate.reasons if r.kind == wanted]
    return decisive[0].text if decisive else candidate.policy_reason


def render_candidate(candidate: ScoredCandidate) -> str:
    template = _environment().get_template("candidate.html")
    ordered = sorted(candidate.signals, key=lambda s: (s.construct.value, s.name))
    return template.render(
        c=candidate, signals=ordered, css=_css(), fmt=_format, fit_floor=FIT_FLOOR
    )


def render_shortlist(candidates: list[ScoredCandidate]) -> str:
    template = _environment().get_template("shortlist.html")
    return template.render(
        candidates=candidates,
        css=_css(),
        lead=_lead_reason,
        generated=date.today().isoformat(),
    )


def to_dict(candidate: ScoredCandidate) -> dict:
    """Machine-readable form. Everything the HTML shows, nothing it does not."""
    return {
        "handle": candidate.handle,
        "display_name": candidate.snapshot.label,
        "platform": candidate.snapshot.platform,
        "captured_at": candidate.snapshot.captured_at.isoformat(),
        "followers": candidate.snapshot.followers,
        "recommendation": candidate.recommendation.value,
        "policy_reason": candidate.policy_reason,
        "fit": {
            "score": round(candidate.fit.score, 1),
            "vetoed": candidate.fit.vetoed,
            "veto_reason": candidate.fit.veto_reason,
            "contributions": [
                {
                    "signal": c.signal,
                    "weight": c.weight,
                    "value": round(c.value, 4),
                    "points": round(c.points, 2),
                    "provenance": c.provenance.value,
                }
                for c in candidate.fit.contributions
            ],
        },
        "propensity": {
            "probability": round(candidate.propensity.probability, 4),
            "base_rate": candidate.propensity.base_rate,
            "lift": round(candidate.propensity.lift, 2),
            "clipped": candidate.propensity.clipped,
            "contributions": [
                {
                    "signal": c.signal,
                    "value": round(c.value, 4),
                    "likelihood_ratio": round(c.likelihood_ratio, 3),
                    "applied": c.applied,
                    "suppressed_by": c.suppressed_by,
                    "provenance": c.provenance.value,
                }
                for c in candidate.propensity.contributions
            ],
        },
        "reasons": [
            {"construct": r.construct.value, "signal": r.signal, "positive": r.positive,
             "text": r.text, "provenance": r.provenance.value}
            for r in candidate.reasons
        ],
    }


def write_reports(candidates: list[ScoredCandidate], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        (out_dir / f"{candidate.handle}.html").write_text(render_candidate(candidate))
        (out_dir / f"{candidate.handle}.json").write_text(
            json.dumps(to_dict(candidate), indent=2) + "\n"
        )
    index = out_dir / "index.html"
    index.write_text(render_shortlist(candidates))
    return index
