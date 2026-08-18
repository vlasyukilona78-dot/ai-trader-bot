"""Execution cost as a range, and the rule that decides on the middle of it.

A single slippage constant turns the central question — is there edge after
costs? — into an assumption. This module replaces it with three bounds and one
rule taken from how a serious replay simulator gates its strategies:

* **Optimistic** — everything fills near the quoted price.
* **Neutral** — the honest middle. This is the bound that decides.
* **Pessimistic** — the stress case.

A strategy that is profitable only in the optimistic bound is rejected. That is
the single most common way a backtest reports edge that does not survive
contact with a real book.

Costs scale with what the bar can tell us: a wide range means a wide spread,
and thin volume means a thin book. That is a proxy, not a measurement — every
coefficient here is declared ``UNVALIDATED`` until it is fitted against real
fills, and :meth:`CostModel.live_ready` stays false until then.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ai.evidence import Assumption, EvidenceClass, live_gate


class ExecutionBound(Enum):
    """How favourably fills are assumed to land."""

    OPTIMISTIC = "optimistic"
    NEUTRAL = "neutral"
    PESSIMISTIC = "pessimistic"

    @property
    def slippage_multiplier(self) -> float:
        return _MULTIPLIER[self]


_MULTIPLIER: dict[ExecutionBound, float] = {
    ExecutionBound.OPTIMISTIC: 0.5,
    ExecutionBound.NEUTRAL: 1.0,
    ExecutionBound.PESSIMISTIC: 2.5,
}

#: Volume ratio below which a bar is treated as this thin, to keep the
#: illiquidity term finite when a bar reports no volume at all.
_MIN_VOLUME_RATIO = 0.05


@dataclass(frozen=True)
class BarContext:
    """What the bar says about execution conditions.

    Attributes:
        range_bps: ``(high - low) / close`` in basis points; a spread proxy.
        volume_ratio: Bar volume over its recent average; a depth proxy.
    """

    range_bps: float
    volume_ratio: float


@dataclass(frozen=True)
class CostModel:
    """Round-trip cost in basis points, as a function of bar conditions."""

    fee_bps_per_side: float
    base_slippage_bps: float
    volatility_coefficient: float
    illiquidity_coefficient: float

    def slippage_bps(self, bound: ExecutionBound, bar: BarContext) -> float:
        """One-sided slippage under the given bound."""

        volatility_term = self.volatility_coefficient * max(bar.range_bps, 0.0) / 100.0
        depth = max(bar.volume_ratio, _MIN_VOLUME_RATIO)
        illiquidity_term = self.illiquidity_coefficient / depth - self.illiquidity_coefficient

        raw = self.base_slippage_bps + volatility_term + illiquidity_term
        return max(raw, 0.0) * bound.slippage_multiplier

    def cost_bps(self, bound: ExecutionBound, bar: BarContext) -> float:
        """Full round-trip cost: fees and slippage on both legs."""

        return 2.0 * (self.fee_bps_per_side + self.slippage_bps(bound, bar))

    def assumptions(self) -> tuple[Assumption, ...]:
        """Provenance of every coefficient this model uses."""

        return (
            Assumption(
                name="fee_bps_per_side",
                value=self.fee_bps_per_side,
                evidence=EvidenceClass.UNVALIDATED,
                source="placeholder; not reconciled against the venue fee schedule",
            ),
            Assumption(
                name="base_slippage_bps",
                value=self.base_slippage_bps,
                evidence=EvidenceClass.UNVALIDATED,
                source="placeholder; no measured fill distribution exists",
            ),
            Assumption(
                name="volatility_coefficient",
                value=self.volatility_coefficient,
                evidence=EvidenceClass.UNVALIDATED,
                source="bar range used as a spread proxy; never fitted to quotes",
            ),
            Assumption(
                name="illiquidity_coefficient",
                value=self.illiquidity_coefficient,
                evidence=EvidenceClass.UNVALIDATED,
                source="volume ratio used as a depth proxy; never fitted to book data",
            ),
        )

    def live_ready(self) -> bool:
        """Whether every coefficient has been measured against real fills."""

        return live_gate(self.assumptions()).passed


@dataclass(frozen=True)
class BoundedPnl:
    """One result evaluated under all three bounds."""

    optimistic: float
    neutral: float
    pessimistic: float


class ProfitabilityGate(Enum):
    """Verdict derived from the three bounds."""

    PASS = "pass"
    REJECT_OPTIMISTIC_ONLY = "reject_optimistic_only"
    REJECT_NON_POSITIVE = "reject_non_positive"


def evaluate_gate(pnl: BoundedPnl) -> ProfitabilityGate:
    """Decide on the neutral bound, never the optimistic one.

    Raises:
        ValueError: The bounds are inverted, which means the caller computed
            them incorrectly rather than found an unusual result.
    """

    if not (pnl.optimistic >= pnl.neutral >= pnl.pessimistic):
        raise ValueError(
            "bounds must be ordered optimistic >= neutral >= pessimistic; "
            f"got {pnl.optimistic}, {pnl.neutral}, {pnl.pessimistic}"
        )
    if pnl.neutral > 0:
        return ProfitabilityGate.PASS
    if pnl.optimistic > 0:
        return ProfitabilityGate.REJECT_OPTIMISTIC_ONLY
    return ProfitabilityGate.REJECT_NON_POSITIVE
