"""Provenance labels for the numbers a strategy's economics depend on.

A cost constant that was guessed and one that was measured against real fills
render identically in a report. Attaching the provenance to the value keeps the
difference visible, and lets a gate refuse to treat a placeholder as evidence.

Nothing in this project has been measured against real fills yet, so every
shipped cost assumption is declared ``UNVALIDATED``. That is the honest label,
and the test suite enforces it until real measurements replace them.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class EvidenceClass(Enum):
    """How a number came to be believed."""

    #: Assumed, copied from a default, or otherwise not derived from anything.
    UNVALIDATED = "unvalidated"
    #: Derived from research assumptions or transformations of other estimates.
    RESEARCH_DERIVED = "research_derived"
    #: Measured against independent real observations outside the fixtures.
    EMPIRICALLY_VALIDATED = "empirically_validated"

    @property
    def strength(self) -> int:
        return _STRENGTH[self]

    def permits_live(self) -> bool:
        """Whether this class can support a real-money decision.

        Passing is necessary, never sufficient: the caller still has to check
        that the referenced measurement is current and covers the case at hand.
        """

        return self is EvidenceClass.EMPIRICALLY_VALIDATED


_STRENGTH: dict[EvidenceClass, int] = {
    EvidenceClass.UNVALIDATED: 0,
    EvidenceClass.RESEARCH_DERIVED: 1,
    EvidenceClass.EMPIRICALLY_VALIDATED: 2,
}


@dataclass(frozen=True)
class Assumption:
    """One number, with how it was established and where it came from."""

    name: str
    value: float
    evidence: EvidenceClass
    source: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("assumption requires a name")
        if not self.source.strip():
            raise ValueError(
                f"assumption {self.name!r} requires a source note describing where the value came from"
            )

    def __repr__(self) -> str:
        return (
            f"Assumption({self.name}={self.value!r}, "
            f"{self.evidence.name}, source={self.source!r})"
        )


@dataclass(frozen=True)
class EvidenceReport:
    """Result of checking a set of assumptions against the live gate."""

    passed: bool
    blocking: tuple[str, ...]


def live_gate(assumptions: Iterable[Assumption]) -> EvidenceReport:
    """Check that every assumption is measured well enough for real money.

    An empty set does not pass: nothing declared means nothing checked, which
    is not evidence of safety.
    """

    listed = list(assumptions)
    if not listed:
        return EvidenceReport(passed=False, blocking=())
    blocking = tuple(a.name for a in listed if not a.evidence.permits_live())
    return EvidenceReport(passed=not blocking, blocking=blocking)


#: The cost inputs this project currently ships. All placeholders.
COST_ASSUMPTIONS: dict[str, Assumption] = {
    "fee_bps_per_side": Assumption(
        name="fee_bps_per_side",
        value=5.5,
        evidence=EvidenceClass.UNVALIDATED,
        source="placeholder; not reconciled against venue fee schedule or real fills",
    ),
    "stop_slippage_bps": Assumption(
        name="stop_slippage_bps",
        value=15.0,
        evidence=EvidenceClass.UNVALIDATED,
        source="placeholder; no measured stop-fill distribution exists",
    ),
    "gap_buffer_bps": Assumption(
        name="gap_buffer_bps",
        value=10.0,
        evidence=EvidenceClass.UNVALIDATED,
        source="placeholder; no measured gap-past-stop distribution exists",
    ),
    "backtest_slippage_bps": Assumption(
        name="backtest_slippage_bps",
        value=2.0,
        evidence=EvidenceClass.UNVALIDATED,
        source="placeholder default in backtesting/backtest.py; flat across all conditions",
    ),
}
