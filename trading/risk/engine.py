from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN

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
    risk_amount_usdt: float = 0.0
    effective_stop_loss: float = 0.0
    execution_cost_buffer_bps_used: float = 0.0
    quality_penalty_bps_used: float = 0.0


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

    @staticmethod
    def _safe_max_qty(rules: InstrumentRules) -> float:
        max_qty = float(rules.max_qty or 0.0)
        if max_qty <= 0:
            return 0.0
        qty_step = max(float(rules.qty_step or 0.0), 0.0)
        step_buffer = qty_step if qty_step > 0 else 0.0
        pct_buffer = max_qty * 0.002
        candidate = max_qty - max(step_buffer, pct_buffer)
        return candidate if candidate > 0 else max_qty

    @staticmethod
    def _round_qty_down(qty: float, qty_step: float) -> float:
        if qty_step <= 0:
            return max(float(qty), 0.0)
        try:
            qty_d = Decimal(str(max(qty, 0.0)))
            step_d = Decimal(str(qty_step))
        except (InvalidOperation, ValueError):
            return 0.0
        if step_d <= 0:
            return 0.0
        steps = (qty_d / step_d).to_integral_value(rounding=ROUND_DOWN)
        rounded = steps * step_d
        return float(max(rounded, Decimal("0")))

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

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _extract_layer_details(intent: StrategyIntent, layer_name: str) -> Mapping[str, object]:
        metadata = intent.metadata if isinstance(intent.metadata, Mapping) else {}
        trace = metadata.get("layer_trace", {})
        if not isinstance(trace, Mapping):
            return {}
        layers = trace.get("layers", {})
        if not isinstance(layers, Mapping):
            return {}
        layer_entry = layers.get(layer_name, {})
        if not isinstance(layer_entry, Mapping):
            return {}
        details = layer_entry.get("details", {})
        return details if isinstance(details, Mapping) else {}

    def _resolve_execution_cost_buffer_bps(
        self,
        *,
        intent: StrategyIntent,
        mark_price: float,
        rules: InstrumentRules,
    ) -> tuple[float, float]:
        base_bps = max(float(getattr(self.limits, "execution_cost_buffer_bps", 0.0) or 0.0), 0.0)
        quality_penalty_bps = 0.0

        layer1 = self._extract_layer_details(intent, "layer1_pump_detection")
        layer2 = self._extract_layer_details(intent, "layer2_weakness_confirmation")
        layer3 = self._extract_layer_details(intent, "layer3_entry_location")
        layer4 = self._extract_layer_details(intent, "layer4_fake_filter")
        metadata = intent.metadata if isinstance(intent.metadata, Mapping) else {}

        confidence = self._safe_float(getattr(intent, "confidence", 0.0), 0.0)
        volume_spike = self._safe_float(layer1.get("volume_spike"), self._safe_float(metadata.get("volume_spike"), 0.0))
        clean_pump_pct = self._safe_float(layer1.get("clean_pump_pct"), self._safe_float(metadata.get("clean_pump_pct"), 0.0))
        layer2_strength = self._safe_float(
            layer2.get("weakness_strength"),
            self._safe_float(metadata.get("weakness_strength"), 0.0),
        )
        layer3_strength = self._safe_float(
            layer3.get("entry_location_strength"),
            self._safe_float(metadata.get("entry_location_strength"), 0.0),
        )
        degraded_mode = bool(
            self._safe_float(layer4.get("degraded_mode"), self._safe_float(metadata.get("degraded_mode"), 0.0))
        )

        spread_bps = self._safe_float(metadata.get("spread_bps"), 0.0)
        turnover24h = max(
            self._safe_float(metadata.get("turnover24h_usdt"), 0.0),
            self._safe_float(metadata.get("quote_turnover_24h"), 0.0),
            self._safe_float(metadata.get("turnover24h"), 0.0),
            self._safe_float(metadata.get("daily_turnover_usdt"), 0.0),
        )

        if confidence > 0.0:
            if confidence < 0.64:
                quality_penalty_bps += 5.0
            elif confidence < 0.76:
                quality_penalty_bps += 2.5

        if layer3_strength > 0.0:
            if layer3_strength < 0.70:
                quality_penalty_bps += 6.0
            elif layer3_strength < 0.80:
                quality_penalty_bps += 3.0

        if layer2_strength > 0.0:
            if layer2_strength < 0.66:
                quality_penalty_bps += 4.0
            elif layer2_strength < 0.76:
                quality_penalty_bps += 2.0

        if volume_spike > 0.0:
            if volume_spike < 0.55:
                quality_penalty_bps += 5.0
            elif volume_spike < 0.90:
                quality_penalty_bps += 2.5

        if clean_pump_pct > 0.0 and clean_pump_pct < 0.05:
            quality_penalty_bps += 2.0

        if degraded_mode:
            quality_penalty_bps += 4.0

        if turnover24h > 0.0:
            if turnover24h < 300_000.0:
                quality_penalty_bps += 8.0
            elif turnover24h < 1_000_000.0:
                quality_penalty_bps += 4.0

        if spread_bps > 0.0:
            if spread_bps >= 15.0:
                quality_penalty_bps += 5.0
            elif spread_bps >= 8.0:
                quality_penalty_bps += 2.5

        if mark_price > 0.0 and rules.tick_size > 0.0:
            tick_bps = (rules.tick_size / max(mark_price, 1e-9)) * 10000.0
            if tick_bps >= 8.0:
                quality_penalty_bps += 2.5
            elif tick_bps >= 4.0:
                quality_penalty_bps += 1.0

        total_bps = min(base_bps + quality_penalty_bps, max(base_bps + 20.0, 35.0))
        return float(total_bps), float(quality_penalty_bps)

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
        execution_cost_buffer_bps_used, quality_penalty_bps_used = self._resolve_execution_cost_buffer_bps(
            intent=intent,
            mark_price=mark_price,
            rules=rules,
        )
        execution_cost_pct = execution_cost_buffer_bps_used / 10000.0
        effective_stop_loss = stop_loss
        if execution_cost_pct > 0.0:
            execution_cost_distance = mark_price * execution_cost_pct
            if intent.action == IntentAction.LONG_ENTRY:
                effective_stop_loss = max(stop_loss - execution_cost_distance, 1e-9)
            else:
                effective_stop_loss = stop_loss + execution_cost_distance
        risk_amount_usdt = equity * max(self.limits.max_risk_per_trade_pct, 0.0)
        raw_qty = position_size_for_stop(
            equity_usdt=equity,
            risk_pct=self.limits.max_risk_per_trade_pct,
            entry_price=mark_price,
            stop_loss=effective_stop_loss,
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
            qty = min(qty, self._safe_max_qty(rules))
        qty = self._round_qty_down(qty, rules.qty_step)
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
            risk_amount_usdt=float(risk_amount_usdt),
            effective_stop_loss=float(effective_stop_loss),
            execution_cost_buffer_bps_used=float(execution_cost_buffer_bps_used),
            quality_penalty_bps_used=float(quality_penalty_bps_used),
        )


