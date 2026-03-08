from __future__ import annotations

import os

if os.getenv("ALLOW_LEGACY_RUNTIME", "false").strip().lower() not in ("1", "true", "yes"):
    raise RuntimeError("Legacy runtime is quarantined. Use V2 entrypoint app/main.py and trading/* modules.")

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta


@dataclass
class RiskConfig:
    account_equity_usdt: float = 1000.0
    risk_per_trade: float = 0.01
    max_concurrent_positions: int = 3
    daily_stop_loss_usdt: float = 100.0
    max_consecutive_losses: int = 3
    circuit_breaker_minutes: int = 30
    min_qty: float = 0.001
    max_qty: float = 100.0


class RiskEngine:
    def __init__(self, config: RiskConfig):
        self.config = config
        self.open_positions: dict[str, dict] = {}
        self.daily_realized_pnl = 0.0
        self.consecutive_losses = 0
        self.circuit_breaker_until: datetime | None = None
        self.current_day = datetime.now(timezone.utc).date()

    def _roll_day(self):
        today = datetime.now(timezone.utc).date()
        if today != self.current_day:
            self.current_day = today
            self.daily_realized_pnl = 0.0
            self.consecutive_losses = 0
            self.circuit_breaker_until = None

    def can_open_trade(self) -> tuple[bool, str]:
        self._roll_day()
        now = datetime.now(timezone.utc)

        if self.circuit_breaker_until is not None and now < self.circuit_breaker_until:
            return False, f"circuit_breaker_until:{self.circuit_breaker_until.isoformat()}"

        if len(self.open_positions) >= self.config.max_concurrent_positions:
            return False, "max_concurrent_positions_reached"

        if self.daily_realized_pnl <= -abs(self.config.daily_stop_loss_usdt):
            return False, "daily_stop_loss_reached"

        if self.consecutive_losses >= self.config.max_consecutive_losses:
            self.circuit_breaker_until = now + timedelta(minutes=self.config.circuit_breaker_minutes)
            return False, "max_consecutive_losses_reached"

        return True, "ok"

    def recommend_qty(self, entry: float, sl: float) -> float:
        if entry <= 0 or sl <= 0:
            return 0.0

        stop_distance = abs(entry - sl)
        if stop_distance <= 0:
            return 0.0

        risk_usdt = max(self.config.account_equity_usdt * self.config.risk_per_trade, 0.0)
        if risk_usdt <= 0:
            return 0.0

        qty = risk_usdt / stop_distance
        qty = max(self.config.min_qty, min(qty, self.config.max_qty))
        return float(qty)

    def on_trade_open(self, signal_id: str, symbol: str, qty: float, entry: float, sl: float):
        self.open_positions[signal_id] = {
            "symbol": symbol,
            "qty": float(qty),
            "entry": float(entry),
            "sl": float(sl),
            "opened_at": datetime.now(timezone.utc).timestamp(),
        }

    def on_trade_closed(self, profit_usdt: float):
        self._roll_day()
        pnl = float(profit_usdt)
        self.daily_realized_pnl += pnl
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def close_position(self, signal_id: str):
        self.open_positions.pop(signal_id, None)

    def snapshot(self) -> dict:
        self._roll_day()
        return {
            "open_positions": len(self.open_positions),
            "daily_realized_pnl": self.daily_realized_pnl,
            "consecutive_losses": self.consecutive_losses,
            "circuit_breaker_until": self.circuit_breaker_until.isoformat() if self.circuit_breaker_until else None,
        }

