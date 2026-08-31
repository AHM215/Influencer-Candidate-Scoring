"""Command line entry points. Files in, files out - no service, no database."""

from __future__ import annotations

from pathlib import Path

import typer

from .archetypes import ARCHETYPES, ArchetypeExtractor
from .domain import ProfileSnapshot
from .pipeline import rank, score_snapshot
from .report import write_reports
from .signals import FixtureExtractor
from .signals.capture import FIXTURE_DIR, InstagramCapturer, load_snapshot, save_snapshot

app = typer.Typer(add_completion=False, help="Score and rank prospective Boutique owners.")
OUT = Path("out")


class _Router:
    """Serves recorded extractions, falling back to the authored Archetype values."""

    def __init__(self) -> None:
        self.fixtures = FixtureExtractor()
        self.archetypes = ArchetypeExtractor()

    def extract(self, snapshot: ProfileSnapshot):
        if snapshot.handle.startswith("archetype_"):
            return self.archetypes.extract(snapshot)
        return self.fixtures.extract(snapshot)


def _snapshots(handles: list[str]) -> list[ProfileSnapshot]:
    found = []
    for handle in handles:
        path = Path(handle) if handle.endswith(".json") else FIXTURE_DIR / f"{handle}.json"
        if not path.exists():
            raise typer.BadParameter(f"No saved Snapshot at {path}")
        found.append(load_snapshot(path))
    return found


@app.command()
def candidate(
    handle: str = typer.Argument(..., help="Handle with a saved Snapshot, or a path to one."),
    out: Path = typer.Option(OUT, help="Directory for the report."),
) -> None:
    """Score one Candidate and write its Candidate Report."""
    scored = score_snapshot(_snapshots([handle])[0], _Router())
    index = write_reports([scored], out)
    typer.echo(
        f"{scored.snapshot.label}: {scored.recommendation.value.upper()} "
        f"(Fit {scored.fit.score:.0f}, Propensity {scored.propensity.probability:.0%})"
    )
    typer.echo(f"  {scored.policy_reason}")
    typer.echo(f"\nReport: {index.parent / (scored.handle + '.html')}")


@app.command()
def shortlist(
    handles: list[str] = typer.Argument(None, help="Handles to rank. Defaults to everything."),
    out: Path = typer.Option(OUT, help="Directory for the reports."),
    archetypes: bool = typer.Option(True, help="Include the synthetic Archetypes."),
) -> None:
    """Rank Candidates and write the Shortlist plus one Report each."""
    router = _Router()
    snapshots = _snapshots(list(handles)) if handles else _saved_snapshots()
    if archetypes and not handles:
        snapshots += [a.snapshot for a in ARCHETYPES]
    if not snapshots:
        raise typer.BadParameter("No Snapshots found. Capture one first.")

    scored = rank([score_snapshot(s, router) for s in snapshots])
    index = write_reports(scored, out)
    width = max(len(c.snapshot.label) for c in scored)
    for i, c in enumerate(scored, 1):
        typer.echo(
            f"{i:2d}. {c.snapshot.label:<{width}}  fit {c.fit.score:5.1f}  "
            f"propensity {c.propensity.probability:4.0%}  -> {c.recommendation.value.upper()}"
        )
    typer.echo(f"\nShortlist: {index}")


@app.command()
def capture(
    handle: str = typer.Argument(..., help="Public Instagram handle."),
) -> None:
    """Capture a live public Profile Snapshot and save it as a fixture."""
    snapshot = InstagramCapturer().capture(handle)
    path = save_snapshot(snapshot)
    typer.echo(
        f"Captured @{handle}: {snapshot.followers:,} followers, "
        f"{len(snapshot.posts)} recent posts -> {path}"
    )
    typer.echo("Next: `score extract " + handle + "` to record the qualitative Signals.")


@app.command()
def extract(
    handle: str = typer.Argument(..., help="Handle with a saved Snapshot."),
) -> None:
    """Record the qualitative extraction for a Candidate. Needs OPENAI_API_KEY."""
    from .signals.qualitative import OpenAIExtractor

    snapshot = _snapshots([handle])[0]
    record = OpenAIExtractor().extract(snapshot)
    typer.echo(f"Recorded extraction for @{handle} ({record.provenance.value}):")
    typer.echo("  " + record.extraction.reasoning)


@app.command()
def validate() -> None:
    """Run the validation harness: behavioural checks, ablations, Cohort diagnostics."""
    from .validation.harness import run_harness

    report = run_harness()
    typer.echo("Behavioural checks")
    for check in report.checks:
        typer.echo(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")

    typer.echo("\nSignal ablation - how far the ranking moves when each Signal is muted")
    typer.echo("  (shows which Signals move OUR model, not which move reality)")
    for a in report.ablations:
        typer.echo(
            f"  {a.signal:<28} rank correlation {a.rank_correlation:.3f}   "
            f"cohort AUC {a.auc_delta:+.3f}"
        )

    typer.echo("\nSynthetic Cohort diagnostics - MEASURES THE PIPELINE, NOT THE HYPOTHESIS")
    typer.echo("  The labels below are invented. A good number here is not evidence that")
    typer.echo("  the scoring is right; see ADR-0004 for why they deliberately disagree.")
    typer.echo(f"  AUC against synthetic outcomes: {report.cohort_auc:.3f}")
    typer.echo("  Calibration:")
    for bucket, count, predicted, observed in report.calibration:
        typer.echo(
            f"    {bucket:>10}  n={count:<4} predicted {predicted:5.1%}  observed {observed:5.1%}"
        )
    typer.echo("  Fitted logistic cross-check (standardised coefficient vs our prior direction):")
    for signal, coefficient, prior in report.logistic_agreement:
        agree = "agrees" if (coefficient >= 0) == (prior >= 0) else "DISAGREES"
        typer.echo(f"    {signal:<28} {coefficient:+.3f}  ({agree} with our prior)")

    raise typer.Exit(0 if report.passed else 1)


def _saved_snapshots() -> list[ProfileSnapshot]:
    if not FIXTURE_DIR.exists():
        return []
    return [load_snapshot(p) for p in sorted(FIXTURE_DIR.glob("*.json"))]


if __name__ == "__main__":
    app()
