"""Model support envelope: abstain outside the range the model was validated on.

Tree ensembles extrapolate by returning the nearest leaf. The output looks like
a confident probability and carries no signal that the input was nothing like
the training data. This module makes that condition explicit and refuses the
entry instead.

The envelope is immutable by design. Live observations that fall outside it
raise an alert and start a challenger; they never widen the envelope, because a
model that expands its own validity ends up trading conditions nobody checked.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


class InferenceAction(Enum):
    """The only decision this gate governs is whether a new entry may proceed."""

    ALLOW = "allow_new_entry"
    ABSTAIN = "abstain_from_new_entry"


@dataclass(frozen=True)
class FeatureBounds:
    """Validated range for one feature, inclusive at both ends."""

    feature_id: str
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if self.maximum < self.minimum:
            raise ValueError(f"invalid bounds for {self.feature_id!r}")

    def contains(self, value: float) -> bool:
        return self.minimum <= value <= self.maximum


@dataclass(frozen=True)
class OodEnvelope:
    """Immutable record of where the model was actually validated."""

    version: str
    bounds: dict[str, FeatureBounds]
    valid_regimes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("envelope requires a version")
        if not self.bounds:
            raise ValueError("envelope requires at least one feature")
        if not self.valid_regimes:
            raise ValueError("envelope requires at least one validated regime")


@dataclass(frozen=True)
class InferenceContext:
    """Point-in-time inputs offered to the model."""

    features: dict[str, float]
    regime: str
    quality_ok: bool = True


@dataclass(frozen=True)
class OodDecision:
    """Reason-coded support decision."""

    action: InferenceAction
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class DriftReport:
    """What live observations said about the envelope, without changing it."""

    version: str
    observations: int
    abstentions: int
    rate: float
    alert: bool
    action: str


def fit_envelope(
    frame: pd.DataFrame, *, version: str, regimes: Sequence[str]
) -> OodEnvelope:
    """Build an envelope from the rows the model was validated on.

    Raises:
        ValueError: No regimes were given, or a feature has no observations to
            bound.
    """

    if not regimes:
        raise ValueError("an envelope must declare the regimes it was validated on")

    bounds: dict[str, FeatureBounds] = {}
    empty: list[str] = []
    for column in frame.columns:
        series = pd.to_numeric(frame[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if series.notna().sum() == 0:
            empty.append(str(column))
            continue
        bounds[str(column)] = FeatureBounds(
            feature_id=str(column),
            minimum=float(series.min()),
            maximum=float(series.max()),
        )

    if empty:
        raise ValueError(
            "cannot bound features with no observations: " + ", ".join(sorted(empty))
        )

    return OodEnvelope(version=version, bounds=bounds, valid_regimes=tuple(regimes))


def evaluate_ood(envelope: OodEnvelope, context: InferenceContext) -> OodDecision:
    """Decide whether the model is inside its validated support.

    Every failing condition is reported, not just the first, so a diagnosis
    does not need repeated round trips.
    """

    reasons: list[str] = []

    if not context.quality_ok:
        reasons.append("FEATURE_QUALITY_INVALID")
    if context.regime not in envelope.valid_regimes:
        reasons.append("REGIME_OOD")

    for feature_id, bound in envelope.bounds.items():
        value = context.features.get(feature_id)
        if value is None or not np.isfinite(value):
            reasons.append(f"FEATURE_MISSING:{feature_id}")
        elif not bound.contains(float(value)):
            reasons.append(f"FEATURE_OOD:{feature_id}")

    action = InferenceAction.ABSTAIN if reasons else InferenceAction.ALLOW
    return OodDecision(action=action, reasons=tuple(reasons))


def detect_drift(
    envelope: OodEnvelope,
    contexts: Iterable[InferenceContext],
    *,
    alert_rate: float,
) -> DriftReport:
    """Measure how often live inputs fell outside support.

    The envelope is never modified. An alert asks for a challenger model fitted
    on the newer data, which then has to pass its own gates.

    Raises:
        ValueError: No observations were supplied, or the rate is not a
            proportion.
    """

    listed = list(contexts)
    if not listed:
        raise ValueError("drift detection requires observations")
    if not 0.0 <= alert_rate <= 1.0:
        raise ValueError("alert_rate must be a proportion between zero and one")

    abstentions = sum(
        evaluate_ood(envelope, context).action is InferenceAction.ABSTAIN
        for context in listed
    )
    rate = abstentions / len(listed)
    alert = rate >= alert_rate
    return DriftReport(
        version=envelope.version,
        observations=len(listed),
        abstentions=abstentions,
        rate=rate,
        alert=alert,
        action="ALERT_AND_CREATE_CHALLENGER" if alert else "NO_CHANGE",
    )
