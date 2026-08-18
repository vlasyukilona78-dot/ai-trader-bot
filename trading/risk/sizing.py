"""Position sizing from a fully priced loss, capped by every binding limit.

The size that satisfies a risk budget depends on what one unit can actually
lose. The stop distance is only part of that: the stop fills through a spread,
the price can gap past it, and both legs pay fees. Sizing on the distance alone
silently spends more of the budget than intended, and does so hardest in the
fast markets a fade strategy trades in.

``size_position`` prices those components, takes the minimum across every cap,
quantizes down to the venue step, and rechecks the budget afterwards rather
than trusting that rounding could only have helped.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

_BPS = 10_000.0


class SizingLimit(Enum):
    """An independently computed ceiling on position size."""

    TRADE_LOSS = "trade_loss"
    LIQUIDITY = "liquidity"
    EXPOSURE = "exposure"
    MARGIN = "margin"
    VENUE_MAXIMUM = "venue_maximum"


@dataclass(frozen=True)
class LossComponents:
    """Everything one unit of the position can lose before the stop is done.

    Attributes:
        entry_price: Expected entry price.
        stop_loss: Protective stop price, on either side of entry.
        stop_slippage_bps: Expected adverse fill distance when the stop
            triggers, in basis points of entry price.
        gap_buffer_bps: Allowance for price gapping past the stop.
        fee_bps_per_side: Trading fee per side, charged on entry and exit.
    """

    entry_price: float
    stop_loss: float
    stop_slippage_bps: float = 0.0
    gap_buffer_bps: float = 0.0
    fee_bps_per_side: float = 0.0

    def loss_per_unit(self) -> float:
        """Total loss for one unit, in quote currency."""

        if self.entry_price <= 0 or self.stop_loss <= 0:
            return 0.0
        structural = abs(self.entry_price - self.stop_loss)
        slippage = self.entry_price * max(self.stop_slippage_bps, 0.0) / _BPS
        gap = self.entry_price * max(self.gap_buffer_bps, 0.0) / _BPS
        fees = self.entry_price * max(self.fee_bps_per_side, 0.0) * 2.0 / _BPS
        return structural + slippage + gap + fees


@dataclass(frozen=True)
class SizingResult:
    """Final size, plus what constrained it."""

    quantity: float
    raw_quantity: float
    loss_per_unit: float
    projected_loss: float
    limiting_factors: tuple[SizingLimit, ...]
    recheck_passed: bool


_NO_POSITION = SizingResult(
    quantity=0.0,
    raw_quantity=0.0,
    loss_per_unit=0.0,
    projected_loss=0.0,
    limiting_factors=(),
    recheck_passed=False,
)


def size_position(
    *,
    risk_amount: float,
    components: LossComponents,
    qty_step: float,
    caps: dict[SizingLimit, float],
    tolerance: float = 1e-9,
) -> SizingResult:
    """Size a position against a fully priced loss and every supplied cap.

    Args:
        risk_amount: Loss budget for this trade, in quote currency.
        components: Priced loss for one unit.
        qty_step: Venue quantity step; ``0`` disables quantization.
        caps: Additional independent ceilings, in base quantity.
        tolerance: Slack allowed when rechecking the budget after rounding.

    Returns:
        A result whose ``quantity`` is zero when no valid size exists.
    """

    loss_per_unit = components.loss_per_unit()
    if loss_per_unit <= 0 or risk_amount <= 0:
        return _NO_POSITION

    ceilings: dict[SizingLimit, float] = {SizingLimit.TRADE_LOSS: risk_amount / loss_per_unit}
    for limit, value in caps.items():
        ceilings[limit] = max(float(value), 0.0)

    raw_quantity = min(ceilings.values())
    if raw_quantity <= 0:
        return SizingResult(
            quantity=0.0,
            raw_quantity=0.0,
            loss_per_unit=loss_per_unit,
            projected_loss=0.0,
            limiting_factors=tuple(
                limit for limit, value in ceilings.items() if value <= 0
            ),
            recheck_passed=False,
        )

    limiting = tuple(
        limit
        for limit, value in ceilings.items()
        if abs(value - raw_quantity) <= tolerance * max(1.0, raw_quantity)
    )

    quantity = raw_quantity
    if qty_step > 0:
        quantity = int(raw_quantity / qty_step) * qty_step

    # Rounding down cannot raise the loss, but a step wider than the budget
    # rounds to zero. Recheck rather than assume.
    projected_loss = quantity * loss_per_unit
    recheck_passed = quantity > 0 and projected_loss <= risk_amount + tolerance
    if not recheck_passed:
        quantity = 0.0
        projected_loss = 0.0

    return SizingResult(
        quantity=quantity,
        raw_quantity=raw_quantity,
        loss_per_unit=loss_per_unit,
        projected_loss=projected_loss,
        limiting_factors=limiting,
        recheck_passed=recheck_passed,
    )


def position_size_for_stop(
    *, equity_usdt: float, risk_pct: float, entry_price: float, stop_loss: float
) -> float:
    """Size from the stop distance alone.

    Retained for callers that have no cost estimates available. Prefer
    :func:`size_position`, which prices the rest of the loss.
    """

    if equity_usdt <= 0 or entry_price <= 0 or stop_loss <= 0:
        return 0.0
    stop_distance = abs(entry_price - stop_loss)
    if stop_distance <= 0:
        return 0.0
    risk_amount = equity_usdt * max(risk_pct, 0.0)
    if risk_amount <= 0:
        return 0.0
    return float(risk_amount / stop_distance)
