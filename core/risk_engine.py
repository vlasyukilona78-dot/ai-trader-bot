from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass
class RiskConfig:
    account_equity_usdt: float = 1000.0
    max_risk_per_trade: float = 0.01
    max_open_positions: int = 3
    max_total_exposure_pct: float = 0.50
    daily_loss_limit_pct: float = 0.05
    max_consecutive_losses: int = 4
    cooldown_minutes: int = 30
    min_qty: float = 0.001
    max_qty: float = 100.0
    slippage_bps: float = 2.0


@dataclass
class SizingResult:
    approved: bool
    reason: str
    qty: float = 0.0
    expected_fill: float = 0.0


class RiskEngine:
    def __init__(self, config: RiskConfig):
        self.cfg = config
        self.open_positions: dict[str, dict] = {}
        self.daily_realized_pnl = 0.0
        self.consecutive_losses = 0
        self.cooldown_until: datetime | None = None
        self._day = datetime.now(timezone.utc).date()

    def _roll_day(self):
        today = datetime.now(timezone.utc).date()
        if today != self._day:
            self._day = today
            self.daily_realized_pnl = 0.0
            self.consecutive_losses = 0
            self.cooldown_until = None

    def can_open(self, open_count: int | None = None, open_exposure_usdt: float = 0.0) -> tuple[bool, str]:
        self._roll_day()
        now = datetime.now(timezone.utc)

        if self.cooldown_until is not None and now < self.cooldown_until:
            return False, f"cooldown_until:{self.cooldown_until.isoformat()}"

        count = len(self.open_positions) if open_count is None else int(open_count)
        if count >= self.cfg.max_open_positions:
            return False, "max_open_positions"

        equity = max(self.cfg.account_equity_usdt, 1e-9)
        if open_exposure_usdt / equity >= self.cfg.max_total_exposure_pct:
            return False, "max_total_exposure"

        if self.daily_realized_pnl <= -equity * self.cfg.daily_loss_limit_pct:
            return False, "daily_loss_limit"

        if self.consecutive_losses >= self.cfg.max_consecutive_losses:
            self.cooldown_until = now + timedelta(minutes=self.cfg.cooldown_minutes)
            return False, "consecutive_losses"

        return True, "ok"

    def estimate_fill_price(self, entry: float, side: str) -> float:
        slip = self.cfg.slippage_bps / 10000.0
        if side == "SHORT":
            return float(entry * (1 - slip))
        return float(entry * (1 + slip))

    def position_size(self, entry: float, sl: float) -> float:
        if entry <= 0 or sl <= 0:
            return 0.0
        risk_per_unit = abs(entry - sl)
        if risk_per_unit <= 0:
            return 0.0

        risk_budget = self.cfg.account_equity_usdt * self.cfg.max_risk_per_trade
        qty = risk_budget / risk_per_unit
        qty = max(self.cfg.min_qty, min(qty, self.cfg.max_qty))
        return float(qty)

    def evaluate_order(self, signal_id: str, side: str, entry: float, sl: float, open_exposure_usdt: float = 0.0) -> SizingResult:
        allowed, reason = self.can_open(open_exposure_usdt=open_exposure_usdt)
        if not allowed:
            return SizingResult(approved=False, reason=reason)

        qty = self.position_size(entry=entry, sl=sl)
        if qty <= 0:
            return SizingResult(approved=False, reason="invalid_qty")

        fill = self.estimate_fill_price(entry=entry, side=side)
        self.open_positions[signal_id] = {
            "side": side,
            "qty": qty,
            "entry": fill,
            "sl": sl,
            "opened_at": datetime.now(timezone.utc).timestamp(),
        }
        return SizingResult(approved=True, reason="ok", qty=qty, expected_fill=fill)

    def close_position(self, signal_id: str, pnl_usdt: float):
        self._roll_day()
        self.open_positions.pop(signal_id, None)
        pnl = float(pnl_usdt)
        self.daily_realized_pnl += pnl
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def snapshot(self) -> dict:
        self._roll_day()
        return {
            "open_positions": len(self.open_positions),
            "daily_realized_pnl": self.daily_realized_pnl,
            "consecutive_losses": self.consecutive_losses,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
        }
