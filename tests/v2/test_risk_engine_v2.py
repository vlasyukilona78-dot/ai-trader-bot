from __future__ import annotations

import unittest

from trading.exchange.schemas import AccountSnapshot, InstrumentRules
from trading.risk.engine import RiskEngine
from trading.risk.limits import RiskLimits
from trading.signals.signal_types import IntentAction, StrategyIntent


class RiskEngineV2Tests(unittest.TestCase):
    def setUp(self):
        self.limits = RiskLimits(
            max_risk_per_trade_pct=0.01,
            max_daily_loss_pct=0.05,
            max_leverage=2.0,
            max_concurrent_positions=2,
            max_symbol_exposure_pct=0.4,
            max_total_notional_pct=0.8,
            min_liquidation_buffer_pct=0.01,
            require_stop_loss=True,
            pyramiding_enabled=False,
        )
        self.engine = RiskEngine(self.limits)
        self.account = AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0)
        self.rules = InstrumentRules(symbol="BTCUSDT", tick_size=0.1, qty_step=0.001, min_qty=0.001, min_notional=5.0)

    @staticmethod
    def _roomy_limits(*, execution_cost_buffer_bps: float = 6.0) -> RiskLimits:
        return RiskLimits(
            max_risk_per_trade_pct=0.01,
            max_daily_loss_pct=0.05,
            max_leverage=10.0,
            max_concurrent_positions=2,
            max_symbol_exposure_pct=2.0,
            max_total_notional_pct=10.0,
            min_liquidation_buffer_pct=0.01,
            execution_cost_buffer_bps=execution_cost_buffer_bps,
            require_stop_loss=True,
            pyramiding_enabled=False,
        )

    def test_size_calculation_positive(self):
        intent = StrategyIntent(symbol="BTCUSDT", action=IntentAction.LONG_ENTRY, reason="t", stop_loss=99.0)
        decision = self.engine.evaluate(intent=intent, account=self.account, existing_positions=[], mark_price=100.0, rules=self.rules)
        self.assertTrue(decision.approved)
        self.assertGreater(decision.quantity, 0.0)

    def test_reject_without_stop(self):
        intent = StrategyIntent(symbol="BTCUSDT", action=IntentAction.LONG_ENTRY, reason="t", stop_loss=None)
        decision = self.engine.evaluate(intent=intent, account=self.account, existing_positions=[], mark_price=100.0, rules=self.rules)
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "stop_loss_required")

    def test_applies_leverage_limit(self):
        tight = RiskLimits(max_risk_per_trade_pct=0.05, max_leverage=0.2, max_total_notional_pct=1.0, max_symbol_exposure_pct=1.0)
        engine = RiskEngine(tight)
        intent = StrategyIntent(symbol="BTCUSDT", action=IntentAction.LONG_ENTRY, reason="t", stop_loss=99.5)
        decision = engine.evaluate(intent=intent, account=self.account, existing_positions=[], mark_price=100.0, rules=self.rules)
        self.assertTrue(decision.approved)
        self.assertLessEqual(decision.implied_leverage, 0.2 + 1e-9)

    def test_reject_liquidation_too_close(self):
        intent = StrategyIntent(
            symbol="BTCUSDT",
            action=IntentAction.LONG_ENTRY,
            reason="t",
            stop_loss=99.0,
            metadata={"liq_price_hint": 99.5},
        )
        decision = self.engine.evaluate(intent=intent, account=self.account, existing_positions=[], mark_price=100.0, rules=self.rules)
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "liquidation_too_close")

    def test_caps_quantity_with_small_safety_margin_below_exchange_max_qty(self):
        rules = InstrumentRules(
            symbol="BTCUSDT",
            tick_size=0.1,
            qty_step=1.0,
            min_qty=1.0,
            min_notional=5.0,
            max_qty=1000.0,
        )
        limits = RiskLimits(
            max_risk_per_trade_pct=0.5,
            max_daily_loss_pct=0.05,
            max_leverage=50.0,
            max_concurrent_positions=2,
            max_symbol_exposure_pct=1.0,
            max_total_notional_pct=1.0,
            min_liquidation_buffer_pct=0.0,
            require_stop_loss=True,
            pyramiding_enabled=False,
        )
        engine = RiskEngine(limits)
        intent = StrategyIntent(symbol="BTCUSDT", action=IntentAction.LONG_ENTRY, reason="t", stop_loss=99.99)
        large_account = AccountSnapshot(equity_usdt=1_000_000.0, available_balance_usdt=1_000_000.0)
        decision = engine.evaluate(intent=intent, account=large_account, existing_positions=[], mark_price=100.0, rules=rules)
        self.assertTrue(decision.approved)
        self.assertLess(decision.quantity, 1000.0)
        self.assertEqual(decision.quantity, 998.0)

    def test_execution_cost_buffer_reduces_position_size(self):
        account = AccountSnapshot(equity_usdt=10_000.0, available_balance_usdt=10_000.0)
        rules = InstrumentRules(symbol="BTCUSDT", tick_size=0.1, qty_step=0.001, min_qty=0.001, min_notional=5.0)
        intent = StrategyIntent(symbol="BTCUSDT", action=IntentAction.LONG_ENTRY, reason="t", stop_loss=98.0)
        baseline_engine = RiskEngine(self._roomy_limits(execution_cost_buffer_bps=6.0))
        baseline = baseline_engine.evaluate(
            intent=intent,
            account=account,
            existing_positions=[],
            mark_price=100.0,
            rules=rules,
        )
        widened_limits = self._roomy_limits(execution_cost_buffer_bps=50.0)
        widened_engine = RiskEngine(widened_limits)
        widened = widened_engine.evaluate(
            intent=intent,
            account=account,
            existing_positions=[],
            mark_price=100.0,
            rules=rules,
        )

        self.assertTrue(baseline.approved)
        self.assertTrue(widened.approved)
        self.assertGreater(widened.execution_cost_buffer_bps_used, baseline.execution_cost_buffer_bps_used)
        self.assertLess(widened.effective_stop_loss, baseline.effective_stop_loss)
        self.assertLess(widened.quantity, baseline.quantity)

    def test_quality_penalty_buffer_reduces_position_size_for_weaker_setup(self):
        engine = RiskEngine(self._roomy_limits(execution_cost_buffer_bps=6.0))
        account = AccountSnapshot(equity_usdt=10_000.0, available_balance_usdt=10_000.0)
        rules = InstrumentRules(symbol="BTCUSDT", tick_size=0.1, qty_step=0.001, min_qty=0.001, min_notional=5.0)
        baseline_intent = StrategyIntent(
            symbol="BTCUSDT",
            action=IntentAction.SHORT_ENTRY,
            reason="t",
            stop_loss=101.0,
            confidence=0.88,
            metadata={
                "layer_trace": {
                    "layers": {
                        "layer1_pump_detection": {"details": {"volume_spike": 1.8, "clean_pump_pct": 0.08}},
                        "layer2_weakness_confirmation": {"details": {"weakness_strength": 0.92}},
                        "layer3_entry_location": {"details": {"entry_location_strength": 0.91}},
                        "layer4_fake_filter": {"details": {"degraded_mode": 0.0}},
                    }
                }
            },
        )
        weak_intent = StrategyIntent(
            symbol="BTCUSDT",
            action=IntentAction.SHORT_ENTRY,
            reason="t",
            stop_loss=101.0,
            confidence=0.58,
            metadata={
                "layer_trace": {
                    "layers": {
                        "layer1_pump_detection": {"details": {"volume_spike": 0.52, "clean_pump_pct": 0.041}},
                        "layer2_weakness_confirmation": {"details": {"weakness_strength": 0.63}},
                        "layer3_entry_location": {"details": {"entry_location_strength": 0.67}},
                        "layer4_fake_filter": {"details": {"degraded_mode": 1.0}},
                    }
                }
            },
        )

        baseline = engine.evaluate(
            intent=baseline_intent,
            account=account,
            existing_positions=[],
            mark_price=100.0,
            rules=rules,
        )
        weak = engine.evaluate(
            intent=weak_intent,
            account=account,
            existing_positions=[],
            mark_price=100.0,
            rules=rules,
        )

        self.assertTrue(baseline.approved)
        self.assertTrue(weak.approved)
        self.assertGreater(weak.execution_cost_buffer_bps_used, baseline.execution_cost_buffer_bps_used)
        self.assertGreater(weak.quality_penalty_bps_used, 0.0)
        self.assertLess(weak.quantity, baseline.quantity)
        self.assertGreater(weak.effective_stop_loss, baseline.effective_stop_loss)
        self.assertGreater(weak.effective_stop_loss, baseline_intent.stop_loss)

    def test_turnover_and_spread_penalties_raise_buffer_even_without_layer_trace(self):
        intent = StrategyIntent(
            symbol="BTCUSDT",
            action=IntentAction.SHORT_ENTRY,
            reason="t",
            stop_loss=101.0,
            confidence=0.82,
            metadata={
                "turnover24h_usdt": 250000.0,
                "spread_bps": 16.0,
            },
        )

        decision = self.engine.evaluate(
            intent=intent,
            account=self.account,
            existing_positions=[],
            mark_price=100.0,
            rules=self.rules,
        )

        self.assertTrue(decision.approved)
        self.assertGreater(decision.execution_cost_buffer_bps_used, self.limits.execution_cost_buffer_bps)
        self.assertGreater(decision.quality_penalty_bps_used, 0.0)

    def test_rounds_qty_down_with_decimal_safe_step_alignment(self):
        limits = self._roomy_limits(execution_cost_buffer_bps=0.0)
        limits = RiskLimits(
            max_risk_per_trade_pct=0.0003,
            max_daily_loss_pct=limits.max_daily_loss_pct,
            max_leverage=limits.max_leverage,
            max_concurrent_positions=limits.max_concurrent_positions,
            max_symbol_exposure_pct=limits.max_symbol_exposure_pct,
            max_total_notional_pct=limits.max_total_notional_pct,
            min_liquidation_buffer_pct=limits.min_liquidation_buffer_pct,
            execution_cost_buffer_bps=0.0,
            require_stop_loss=True,
            pyramiding_enabled=False,
        )
        engine = RiskEngine(limits)
        account = AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0)
        rules = InstrumentRules(symbol="BTCUSDT", tick_size=0.001, qty_step=0.1, min_qty=0.1, min_notional=5.0)
        intent = StrategyIntent(symbol="BTCUSDT", action=IntentAction.LONG_ENTRY, reason="t", stop_loss=99.0)

        decision = engine.evaluate(
            intent=intent,
            account=account,
            existing_positions=[],
            mark_price=100.0,
            rules=rules,
        )

        self.assertTrue(decision.approved)
        self.assertEqual(decision.quantity, 0.3)


if __name__ == "__main__":
    unittest.main()

