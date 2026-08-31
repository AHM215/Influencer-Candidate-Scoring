"""Signal extraction: Profile Snapshot in, the twelve Signals out."""

from __future__ import annotations

from dataclasses import replace

from ..domain import ProfileSnapshot, Provenance, SignalValue
from .derive import derive_signals
from .qualitative import ExtractionRecord, Extractor, FixtureExtractor, to_signals

__all__ = [
    "build_signals",
    "derive_signals",
    "ExtractionRecord",
    "Extractor",
    "FixtureExtractor",
    "to_signals",
]

# Weakest evidence wins: a value can never be better-sourced than what it was derived from.
_SEVERITY = {Provenance.OBSERVED: 0, Provenance.INFERRED: 1, Provenance.MOCKED: 2}


def build_signals(
    snapshot: ProfileSnapshot, extractor: Extractor | None = None
) -> dict[str, SignalValue]:
    extractor = extractor or FixtureExtractor()
    signals = derive_signals(snapshot)
    signals.update(to_signals(extractor.extract(snapshot)))
    return {name: _inherit(signal, snapshot.provenance) for name, signal in signals.items()}


def _inherit(signal: SignalValue, source: Provenance) -> SignalValue:
    """Degrade a Signal to its Snapshot's provenance when the Snapshot is weaker.

    A plausibility score inferred from invented post counts is invented, not inferred, and
    a model reading invented captions is not reading evidence. Without this, provenance
    silently improves as values flow downstream - exactly the overstated claim the
    Candidate Report exists to prevent.
    """
    if _SEVERITY[source] > _SEVERITY[signal.provenance]:
        return replace(signal, provenance=source)
    return signal
