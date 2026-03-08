from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trading.exchange.schemas import AccountSnapshot, InstrumentRules, PositionSide, PositionSnapshot
from trading.risk.engine import RiskEngine
from trading.risk.limits import RiskLimits
from trading.signals.signal_types import IntentAction, StrategyIntent
from trading.state.persistence import RuntimeStore


class RiskPersistenceV2Tests(unittest.TestCase):
    def setUp(self):
        self.account = AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0)
        self.rules = InstrumentRules(symbol="BTCUSDT", tick_size=0.1, qty_step=0.001, min_qty=0.001, min_notional=5.0)

    def test_persisted_session_loss_and_consecutive_halt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RuntimeStore(str(Path(tmpdir) / "runtime.db"))
            limits = RiskLimits(
                max_daily_loss_pct=0.5,
                halt_after_consecutive_losses=2,
                max_total_notional_pct=1.0,
                max_symbol_exposure_pct=1.0,
                max_leverage=5.0,
            )
            engine = RiskEngine(limits, persistence=store)
            engine.record_trade_result(-10.0)
            engine.record_trade_result(-5.0)

            engine2 = RiskEngine(limits, persistence=store)
            intent = StrategyIntent(symbol="BTCUSDT", action=IntentAction.LONG_ENTRY, reason="x", stop_loss=99.0)
            decision = engine2.evaluate(intent=intent, account=self.account, existing_positions=[], mark_price=100.0, rules=self.rules)
            self.assertFalse(decision.approved)
            self.assertEqual(decision.reason, "consecutive_loss_halt")
            store.close()

    def test_cooldown_after_stop_loss_exit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RuntimeStore(str(Path(tmpdir) / "runtime.db"))
            limits = RiskLimits(
                cooldown_after_stop_sec=3600,
                max_total_notional_pct=1.0,
                max_symbol_exposure_pct=1.0,
                max_leverage=5.0,
            )
            engine = RiskEngine(limits, persistence=store)
            engine.record_trade_result(-2.0, stopped_out=True)
            intent = StrategyIntent(symbol="ETHUSDT", action=IntentAction.SHORT_ENTRY, reason="x", stop_loss=101.0)
            decision = engine.evaluate(intent=intent, account=self.account, existing_positions=[], mark_price=100.0, rules=self.rules)
            self.assertFalse(decision.approved)
            self.assertEqual(decision.reason, "cooldown_active")
            store.close()

    def test_cross_symbol_exposure_limit(self):
        limits = RiskLimits(
            max_symbol_exposure_pct=0.3,
            max_total_notional_pct=1.0,
            max_leverage=5.0,
            pyramiding_enabled=True,
        )
        engine = RiskEngine(limits)
        existing = [
            PositionSnapshot(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                qty=3.5,
                entry_price=100.0,
                liq_price=0.0,
                leverage=1.0,
                position_idx=0,
            )
        ]
        intent = StrategyIntent(symbol="BTCUSDT", action=IntentAction.LONG_ENTRY, reason="x", stop_loss=99.0)
        decision = engine.evaluate(intent=intent, account=self.account, existing_positions=existing, mark_price=100.0, rules=self.rules)
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "max_symbol_exposure")


if __name__ == "__main__":
    unittest.main()

