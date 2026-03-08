from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class EquityPoint:
    ts: float
    equity: float


class PnLTracker:
    def __init__(self):
        self.realized_pnl = 0.0
        self.equity_curve: list[EquityPoint] = []

    def record_realized(self, pnl_usdt: float, current_equity: float):
        self.realized_pnl += float(pnl_usdt)
        self.equity_curve.append(EquityPoint(ts=datetime.now(timezone.utc).timestamp(), equity=float(current_equity)))
