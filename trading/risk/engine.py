from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from trading.exchange.schemas import AccountSnapshot, InstrumentRules, PositionSide, PositionSnapshot
from trading.risk.limits import RiskLimits
from trading.risk.liquidation import liquidation_buffer_ok
from trading.portfolio.positions import split_effective_positions
from trading.risk.sizing import position_size_for_stop
from trading.signals.signal_types import IntentAction, StrategyIntent
from trading.state.persistence import PersistedRiskRow, RuntimeStore


@dataclass
class RiskDecision:
    approved: bool
    reason: str
    quantity: float = 0.0
    notional: float = 0.0
    implied_leverage: float = 0.0


@dataclass
class RiskState:
    realized_pnl_today: float = 0.0
    consecutive_losses: int = 0
    cooldown_until_ts: float = 0.0
    session_day: str = ""
    last_updated_ts: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())


class RiskEngine:
    """Single source of truth for trade approval and sizing."""

    def __init__(self, limits: RiskLimits, persistence: RuntimeStore | None = None):
        self.limits = limits
        self.persistence = persistence
        self.state = RiskState(session_day=self._session_day())
        self._load_state()

    @staticmethod
    def _now_ts() -> float:
        return datetime.now(timezone.utc).timestamp()

    @staticmethod
    def _session_day(ts: float | None = None) -> str:
        source = datetime.fromtimestamp(ts or datetime.now(timezone.utc).timestamp(), tz=timezone.utc)
        return source.date().isoformat()

    @staticmethod
    def _position_notional(pos: PositionSnapshot) -> float:
        px = pos.entry_price if pos.entry_price > 0 else 0.0
        return float(max(pos.qty, 0.0) * px)

    def _load_state(self):
        if self.persistence is None:
            return
        row = self.persistence.load_risk_row(self.state.session_day)
        if row is None:
            self._persist_state()
            return
        self.state.realized_pnl_today = float(row.realized_pnl)
        self.state.consecutive_losses = int(row.consecutive_losses)
        self.state.cooldown_until_ts = float(row.cooldown_until_ts)
        self.state.last_updated_ts = float(row.updated_at)

    def _persist_state(self):
        if self.persistence is None:
            return
        self.persistence.save_risk_row(
            PersistedRiskRow(
                session_day=self.state.session_day,
                realized_pnl=float(self.state.realized_pnl_today),
                consecutive_losses=int(self.state.consecutive_losses),
                cooldown_until_ts=float(self.state.cooldown_until_ts),
                updated_at=float(self.state.last_updated_ts),
            )
        )

    def _roll_session_if_needed(self):
        now_ts = self._now_ts()
        day = self._session_day(now_ts)
        if day == self.state.session_day:
            return
        self.state.session_day = day
        self.state.realized_pnl_today = 0.0
        self.state.consecutive_losses = 0
        self.state.cooldown_until_ts = 0.0
        self.state.last_updated_ts = now_ts
        self._persist_state()

    def health_snapshot(self) -> dict:
        self._roll_session_if_needed()
        now_ts = self._now_ts()
        return {
            "session_day": self.state.session_day,
            "realized_pnl_today": float(self.state.realized_pnl_today),
            "consecutive_losses": int(self.state.consecutive_losses),
            "cooldown_active": bool(now_ts < self.state.cooldown_until_ts),
            "cooldown_until_ts": float(self.state.cooldown_until_ts),
        }

    def record_trade_result(self, pnl_usdt: float, stopped_out: bool = False):
        self._roll_session_if_needed()
        self.state.last_updated_ts = self._now_ts()
        self.state.realized_pnl_today += float(pnl_usdt)
        if pnl_usdt < 0:
            self.state.consecutive_losses += 1
            if stopped_out and self.limits.cooldown_after_stop_sec > 0:
                self.state.cooldown_until_ts = self.state.last_updated_ts + float(self.limits.cooldown_after_stop_sec)
        else:
            self.state.consecutive_losses = 0
        self._persist_state()

    def _global_guards(self, account: AccountSnapshot) -> tuple[bool, str]:
        self._roll_session_if_needed()
        now_ts = self._now_ts()
        if now_ts < self.state.cooldown_until_ts:
            return False, "cooldown_active"

        equity = max(account.equity_usdt, 0.0)
        if equity <= 0:
            return False, "non_positive_equity"

        if self.state.realized_pnl_today <= -(equity * self.limits.max_daily_loss_pct):
            return False, "daily_loss_limit"

        if self.state.consecutive_losses >= self.limits.halt_after_consecutive_losses:
            return False, "consecutive_loss_halt"

        return True, "ok"

    def evaluate(
        self,
        *,
        intent: StrategyIntent,
        account: AccountSnapshot,
        existing_positions: list[PositionSnapshot],
        mark_price: float,
        rules: InstrumentRules,
    ) -> RiskDecision:
        if intent.action in (IntentAction.HOLD, IntentAction.EXIT_LONG, IntentAction.EXIT_SHORT):
            return RiskDecision(approved=True, reason="exit_or_hold")

        if intent.action not in (IntentAction.LONG_ENTRY, IntentAction.SHORT_ENTRY):
            return RiskDecision(approved=False, reason="unsupported_intent")

        effective_positions, _ = split_effective_positions(existing_positions)

        ok, reason = self._global_guards(account)
        if not ok:
            return RiskDecision(approved=False, reason=reason)

        if self.limits.require_stop_loss and (intent.stop_loss is None or intent.stop_loss <= 0):
            return RiskDecision(approved=False, reason="stop_loss_required")

        if mark_price <= 0:
            return RiskDecision(approved=False, reason="invalid_mark_price")

        if intent.action == IntentAction.LONG_ENTRY and intent.stop_loss is not None and intent.stop_loss >= mark_price:
            return RiskDecision(approved=False, reason="invalid_long_stop")

        if intent.action == IntentAction.SHORT_ENTRY and intent.stop_loss is not None and intent.stop_loss <= mark_price:
            return RiskDecision(approved=False, reason="invalid_short_stop")

        if not self.limits.pyramiding_enabled:
            if len(effective_positions) >= self.limits.max_concurrent_positions:
                return RiskDecision(approved=False, reason="max_concurrent_positions")

        normalized_symbol = intent.symbol.replace("/", "").upper()
        if not self.limits.pyramiding_enabled:
            if any(pos.symbol.replace("/", "").upper() == normalized_symbol for pos in effective_positions):
                return RiskDecision(approved=False, reason="symbol_already_open")

        equity = max(account.equity_usdt, 1e-9)
        total_open_notional = sum(self._position_notional(pos) for pos in effective_positions)
        if total_open_notional / equity >= self.limits.max_total_notional_pct:
            return RiskDecision(approved=False, reason="max_total_notional")

        symbol_notional = sum(
            self._position_notional(pos)
            for pos in effective_positions
            if pos.symbol.replace("/", "").upper() == normalized_symbol
        )
        if symbol_notional / equity >= self.limits.max_symbol_exposure_pct:
            return RiskDecision(approved=False, reason="max_symbol_exposure")

        stop_loss = float(intent.stop_loss or 0.0)
        raw_qty = position_size_for_stop(
            equity_usdt=equity,
            risk_pct=self.limits.max_risk_per_trade_pct,
            entry_price=mark_price,
            stop_loss=stop_loss,
        )
        if raw_qty <= 0:
            return RiskDecision(approved=False, reason="non_positive_size")

        remaining_total_notional = max(0.0, equity * self.limits.max_total_notional_pct - total_open_notional)
        if remaining_total_notional <= 0:
            return RiskDecision(approved=False, reason="projected_total_notional")

        remaining_symbol_notional = max(0.0, equity * self.limits.max_symbol_exposure_pct - symbol_notional)
        if remaining_symbol_notional <= 0:
            return RiskDecision(approved=False, reason="projected_symbol_exposure")

        remaining_leverage_notional = max(0.0, equity * self.limits.max_leverage - total_open_notional)
        if remaining_leverage_notional <= 0:
            return RiskDecision(approved=False, reason="max_leverage")

        remaining_available_balance = max(0.0, account.available_balance_usdt)
        max_notional_allowed = min(
            remaining_total_notional,
            remaining_symbol_notional,
            remaining_leverage_notional,
            remaining_available_balance if remaining_available_balance > 0 else remaining_total_notional,
        )
        if max_notional_allowed <= 0:
            return RiskDecision(approved=False, reason="insufficient_notional_budget")

        qty_cap_by_notional = max_notional_allowed / mark_price
        qty = min(raw_qty, qty_cap_by_notional)
        if rules.max_qty > 0:
            qty = min(qty, rules.max_qty)
        qty = int(qty / rules.qty_step) * rules.qty_step if rules.qty_step > 0 else qty
        if qty < rules.min_qty:
            return RiskDecision(approved=False, reason="below_min_qty_after_limits")

        notional = qty * mark_price
        if notional < rules.min_notional:
            return RiskDecision(approved=False, reason="below_min_notional_after_limits")

        projected_notional = total_open_notional + notional
        implied_leverage = projected_notional / equity
        if implied_leverage > self.limits.max_leverage:
            return RiskDecision(approved=False, reason="max_leverage")

        if projected_notional / equity > self.limits.max_total_notional_pct:
            return RiskDecision(approved=False, reason="projected_total_notional")

        projected_symbol = symbol_notional + notional
        if projected_symbol / equity > self.limits.max_symbol_exposure_pct:
            return RiskDecision(approved=False, reason="projected_symbol_exposure")

        liq_hint = intent.metadata.get("liq_price_hint") if isinstance(intent.metadata, dict) else None
        try:
            liq_price = float(liq_hint) if liq_hint is not None else 0.0
        except (TypeError, ValueError):
            liq_price = 0.0

        side = PositionSide.LONG if intent.action == IntentAction.LONG_ENTRY else PositionSide.SHORT
        if not liquidation_buffer_ok(
            side=side,
            entry_price=mark_price,
            liq_price=liq_price,
            min_buffer_pct=self.limits.min_liquidation_buffer_pct,
        ):
            return RiskDecision(approved=False, reason="liquidation_too_close")

        return RiskDecision(
            approved=True,
            reason="approved",
            quantity=float(qty),
            notional=float(notional),
            implied_leverage=float(implied_leverage),
        )


