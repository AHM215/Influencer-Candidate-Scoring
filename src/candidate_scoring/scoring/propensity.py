"""Success Propensity: a Base Rate moved by argued Likelihood Ratios (ADR-0002).

No model is fitted here. Each Signal multiplies the prior odds by an amount stated in
config.py, correlated Signals are grouped so one piece of evidence is counted once
(ADR-0003), and the result is clipped to a range a public profile can actually support.
"""

from __future__ import annotations

import math

from ..config import (
    BASE_RATE,
    ENGAGEMENT_FAMILY,
    POSTERIOR_MAX,
    POSTERIOR_MIN,
    PROPENSITY_CURVES,
)
from ..domain import PropensityContribution, PropensityResult, SignalValue


def score_propensity(
    signals: dict[str, SignalValue], base_rate: float = BASE_RATE
) -> PropensityResult:
    ratios: dict[str, tuple[float, SignalValue]] = {}
    for name, curve in PROPENSITY_CURVES.items():
        signal = signals.get(name)
        if signal is None:
            continue
        ratios[name] = (curve.likelihood_ratio(signal.value), signal)

    winner = _weakest_in_family(ratios)

    contributions = []
    for name, (lr, signal) in ratios.items():
        in_family = name in ENGAGEMENT_FAMILY
        applied = (not in_family) or name == winner
        contributions.append(
            PropensityContribution(
                signal=name,
                value=signal.value,
                likelihood_ratio=lr,
                provenance=signal.provenance,
                applied=applied,
                suppressed_by=winner if (in_family and not applied) else "",
            )
        )

    log_odds = math.log(base_rate / (1.0 - base_rate))
    log_odds += sum(c.log_lr for c in contributions if c.applied)
    raw = 1.0 / (1.0 + math.exp(-log_odds))
    probability = min(max(raw, POSTERIOR_MIN), POSTERIOR_MAX)

    return PropensityResult(
        probability=probability,
        base_rate=base_rate,
        contributions=tuple(contributions),
        clipped=probability != raw,
    )


def _weakest_in_family(ratios: dict[str, tuple[float, SignalValue]]) -> str:
    """The least favourable engagement Signal: the family is as strong as its weakest member.

    Real engagement corroborates itself - a genuine audience likes, comments and shows up
    consistently. Bought engagement does not: it shows one indicator inflated and the
    others thin, and picking the largest or most flattering member would hand exactly that
    Candidate the credit. Disagreement inside the family is itself the evidence, so the
    pessimistic reading is the honest one.
    """
    present = [(n, lr) for n, (lr, _) in ratios.items() if n in ENGAGEMENT_FAMILY]
    if not present:
        return ""
    return min(present, key=lambda pair: pair[1])[0]
