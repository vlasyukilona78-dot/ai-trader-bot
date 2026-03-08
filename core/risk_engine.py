from __future__ import annotations

import os

if os.getenv("ALLOW_LEGACY_RUNTIME", "false").strip().lower() not in ("1", "true", "yes"):
    raise RuntimeError("Legacy runtime is quarantined. Use V2 entrypoint app/main.py and trading/* modules.")

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass
class RiskConfig:
    account_equity_usdt: float = 1000.0
    max_risk_per_trade: float = 0.01
    max_open_positions: int = 3
    max_total_exposure_pct: float = 0.50
    max_single_symbol_exposure_pct: float = 0.25
    max_notional_per_trade_pct: float = 0.25
    max_leverage: float = 3.0
    min_stop_distance_pct: float = 0.0015
    min_liquidation_buffer_pct: float = 0.003
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

    @staticmethod
    def _normalize_side(side: str) -> str:
        s = str(side).upper().strip()
        if s in ("LONG", "BUY"):
            return "LONG"
        if s in ("SHORT", "SELL"):
            return "SHORT"
        return s

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _roll_day(self):
        today = datetime.now(timezone.utc).date()
        if today != self._day:
            self._day = today
            self.daily_realized_pnl = 0.0
            self.consecutive_losses = 0
            self.cooldown_until = None

    def current_open_exposure(self) -> float:
        exposure = 0.0
        for pos in self.open_positions.values():
            exposure += self._to_float(pos.get("entry", 0.0)) * self._to_float(pos.get("qty", 0.0))
        return float(max(0.0, exposure))

    def symbol_exposure(self, symbol: str) -> float:
        target = str(symbol).upper().replace("/", "")
        exposure = 0.0
        for pos in self.open_positions.values():
            pos_symbol = str(pos.get("symbol", "")).upper().replace("/", "")
            if pos_symbol != target:
                continue
            exposure += self._to_float(pos.get("entry", 0.0)) * self._to_float(pos.get("qty", 0.0))
        return float(max(0.0, exposure))

    def has_open_symbol(self, symbol: str) -> bool:
        target = str(symbol).upper().replace("/", "")
        for pos in self.open_positions.values():
            if self._to_float(pos.get("qty", 0.0)) <= 0.0:
                continue
            pos_symbol = str(pos.get("symbol", "")).upper().replace("/", "")
            if pos_symbol == target:
                return True
        return False

    def can_open(self, open_count: int | None = None, open_exposure_usdt: float | None = None) -> tuple[bool, str]:
        self._roll_day()
        now = datetime.now(timezone.utc)

        if self.cooldown_until is not None and now < self.cooldown_until:
            return False, f"cooldown_until:{self.cooldown_until.isoformat()}"

        count = len(self.open_positions) if open_count is None else int(open_count)
        if count >= self.cfg.max_open_positions:
            return False, "max_open_positions"

        equity = max(self.cfg.account_equity_usdt, 1e-9)
        exposure = self.current_open_exposure() if open_exposure_usdt is None else float(open_exposure_usdt)
        if exposure / equity >= self.cfg.max_total_exposure_pct:
            return False, "max_total_exposure"

        if self.daily_realized_pnl <= -equity * self.cfg.daily_loss_limit_pct:
            return False, "daily_loss_limit"

        if self.consecutive_losses >= self.cfg.max_consecutive_losses:
            self.cooldown_until = now + timedelta(minutes=self.cfg.cooldown_minutes)
            return False, "consecutive_losses"

        return True, "ok"

    def estimate_fill_price(self, entry: float, side: str) -> float:
        slip = self.cfg.slippage_bps / 10000.0
        if self._normalize_side(side) == "SHORT":
            return float(entry * (1 - slip))
        return float(entry * (1 + slip))

    def evaluate_order(
        self,
        signal_id: str,
        side: str,
        entry: float,
        sl: float,
        open_exposure_usdt: float | None = None,
        symbol: str | None = None,
    ) -> SizingResult:
        side_n = self._normalize_side(side)
        entry = self._to_float(entry)
        sl = self._to_float(sl)
        if side_n not in ("LONG", "SHORT"):
            return SizingResult(approved=False, reason="invalid_side")
        if entry <= 0 or sl <= 0:
            return SizingResult(approved=False, reason="invalid_price")

        if side_n == "SHORT" and sl <= entry:
            return SizingResult(approved=False, reason="invalid_sl_for_short")
        if side_n == "LONG" and sl >= entry:
            return SizingResult(approved=False, reason="invalid_sl_for_long")

        if signal_id in self.open_positions:
            return SizingResult(approved=False, reason="duplicate_signal_id")

        if symbol and self.has_open_symbol(symbol):
            return SizingResult(approved=False, reason="symbol_already_open")

        allowed, reason = self.can_open(open_exposure_usdt=open_exposure_usdt)
        if not allowed:
            return SizingResult(approved=False, reason=reason)

        distance = abs(entry - sl)
        if distance <= 0:
            return SizingResult(approved=False, reason="zero_sl_distance")

        stop_distance_pct = distance / max(entry, 1e-9)
        if stop_distance_pct < self.cfg.min_stop_distance_pct:
            return SizingResult(approved=False, reason="stop_too_close")
        if stop_distance_pct < self.cfg.min_liquidation_buffer_pct:
            return SizingResult(approved=False, reason="liq_buffer_too_small")

        equity = max(self.cfg.account_equity_usdt, 1e-9)
        risk_budget = equity * self.cfg.max_risk_per_trade
        raw_qty = risk_budget / distance
        if raw_qty <= 0:
            return SizingResult(approved=False, reason="invalid_qty")

        fill = self.estimate_fill_price(entry=entry, side=side_n)
        if fill <= 0:
            return SizingResult(approved=False, reason="invalid_fill")

        max_notional = equity * max(0.0, self.cfg.max_notional_per_trade_pct)
        qty_by_notional = raw_qty if max_notional <= 0 else max_notional / fill

        max_lev = max(0.0, self.cfg.max_leverage)
        qty_by_leverage = raw_qty if max_lev <= 0 else (equity * max_lev) / fill

        qty = min(raw_qty, self.cfg.max_qty, qty_by_notional, qty_by_leverage)
        if qty < self.cfg.min_qty:
            return SizingResult(approved=False, reason="qty_below_min_after_caps")

        current_exposure = self.current_open_exposure() if open_exposure_usdt is None else float(open_exposure_usdt)
        projected_total = current_exposure + qty * fill
        if projected_total / equity > self.cfg.max_total_exposure_pct:
            return SizingResult(approved=False, reason="max_total_exposure_after_trade")

        if symbol:
            projected_symbol = self.symbol_exposure(symbol) + qty * fill
            if projected_symbol / equity > self.cfg.max_single_symbol_exposure_pct:
                return SizingResult(approved=False, reason="max_symbol_exposure")

        return SizingResult(approved=True, reason="ok", qty=float(qty), expected_fill=float(fill))

    def register_open_position(
        self,
        signal_id: str,
        *,
        symbol: str,
        side: str,
        qty: float,
        entry: float,
        sl: float,
        source: str = "local",
    ):
        self.open_positions[signal_id] = {
            "symbol": str(symbol),
            "side": self._normalize_side(side),
            "qty": float(qty),
            "entry": float(entry),
            "sl": float(sl),
            "opened_at": datetime.now(timezone.utc).timestamp(),
            "source": source,
        }

    def sync_exchange_positions(self, positions: list[dict]):
        now_ts = datetime.now(timezone.utc).timestamp()
        exchange_map: dict[str, dict] = {}
        for item in positions:
            if not isinstance(item, dict):
                continue
            qty = self._to_float(item.get("size", 0.0))
            if qty <= 0:
                continue

            symbol = str(item.get("symbol", ""))
            side = self._normalize_side(item.get("side"))
            if not symbol or side not in ("LONG", "SHORT"):
                continue

            position_idx = int(self._to_float(item.get("positionIdx", 0)))
            key = f"exchange:{symbol}:{side}:{position_idx}"
            exchange_map[key] = {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "entry": self._to_float(item.get("entry_price", item.get("entryPrice", 0.0))),
                "sl": self._to_float(item.get("stopLoss", 0.0)),
                "opened_at": now_ts,
                "source": "exchange",
                "positionIdx": position_idx,
            }

        merged = dict(exchange_map)
        for key, pos in self.open_positions.items():
            if pos.get("source") != "local":
                continue
            age_sec = now_ts - self._to_float(pos.get("opened_at"), now_ts)
            if age_sec > 120:
                continue
            symbol = str(pos.get("symbol", ""))
            side = self._normalize_side(pos.get("side"))
            conflict = any(
                str(x.get("symbol", "")).upper().replace("/", "") == symbol.upper().replace("/", "")
                and self._normalize_side(x.get("side")) == side
                for x in exchange_map.values()
            )
            if not conflict:
                merged[key] = pos

        self.open_positions = merged

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
            "open_exposure": round(self.current_open_exposure(), 4),
            "daily_realized_pnl": self.daily_realized_pnl,
            "consecutive_losses": self.consecutive_losses,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
        }

