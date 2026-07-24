from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from core.signal_generator import SignalResult
from trading.exchange.schemas import AccountSnapshot
from trading.market_data.reconciliation import ExchangeSnapshot
from trading.signals.layered_strategy import LayeredPumpStrategy
from trading.signals.signal_types import IntentAction
from trading.signals.strategy_interface import StrategyContext
from trading.state.models import TradeState


def _ohlcv(n: int = 85) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    closes = [100.0 + i * 0.01 for i in range(n)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.001 for c in closes],
            "low": [c * 0.999 for c in closes],
            "close": closes,
            "volume": [10.0] * n,
            "rsi": [50.0] * n,
        },
        index=idx,
    )


def _snapshot() -> ExchangeSnapshot:
    return ExchangeSnapshot(
        symbol="BTCUSDT",
        account=AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0),
        positions=[],
        open_orders=[],
    )


def _context(synced_state: TradeState) -> StrategyContext:
    return StrategyContext(
        symbol="BTCUSDT",
        market_ohlcv=_ohlcv(),
        mark_price=100.0,
        exchange=_snapshot(),
        synced_state=synced_state,
        sentiment_index=50.0,
        sentiment_source="fallback_neutral_50",
    )


def _signal(side: str) -> SignalResult:
    return SignalResult(
        signal_id="TEST-1",
        symbol="BTCUSDT",
        side=side,
        entry=100.0,
        sl=99.0 if side == "LONG" else 101.0,
        tp=102.0 if side == "LONG" else 98.0,
        confidence=0.8,
    )


class LayeredPumpStrategyV2Tests(unittest.TestCase):
    def _strategy_with_signal(self, signal: SignalResult | None):
        strategy = LayeredPumpStrategy()
        strategy._generator.generate = lambda ctx: signal
        strategy._generator.last_diagnostics = {"failed_layer": None if signal else "layer1_pump_detection", "layers": {}}
        return strategy

    def test_insufficient_history_holds(self):
        strategy = LayeredPumpStrategy()
        short_ctx = _context(TradeState.FLAT)
        short_ctx.market_ohlcv = short_ctx.market_ohlcv.iloc[:10]
        intent = strategy.generate(short_ctx)
        self.assertEqual(intent.action, IntentAction.HOLD)
        self.assertEqual(intent.reason, "insufficient_history")

    def test_no_signal_holds_with_failed_layer_reason(self):
        strategy = self._strategy_with_signal(None)
        intent = strategy.generate(_context(TradeState.FLAT))
        self.assertEqual(intent.action, IntentAction.HOLD)
        self.assertIn("layer1_pump_detection", intent.reason)

    def test_long_entry_when_flat(self):
        strategy = self._strategy_with_signal(_signal("LONG"))
        intent = strategy.generate(_context(TradeState.FLAT))
        self.assertEqual(intent.action, IntentAction.LONG_ENTRY)
        self.assertEqual(intent.stop_loss, 99.0)
        self.assertEqual(intent.take_profit, 102.0)

    def test_short_entry_when_flat(self):
        strategy = self._strategy_with_signal(_signal("SHORT"))
        intent = strategy.generate(_context(TradeState.FLAT))
        self.assertEqual(intent.action, IntentAction.SHORT_ENTRY)

    def test_opposite_signal_closes_long(self):
        strategy = self._strategy_with_signal(_signal("SHORT"))
        intent = strategy.generate(_context(TradeState.LONG))
        self.assertEqual(intent.action, IntentAction.EXIT_LONG)
        self.assertEqual(intent.reason, "opposite_signal_close_long")

    def test_opposite_signal_closes_short(self):
        strategy = self._strategy_with_signal(_signal("LONG"))
        intent = strategy.generate(_context(TradeState.SHORT))
        self.assertEqual(intent.action, IntentAction.EXIT_SHORT)
        self.assertEqual(intent.reason, "opposite_signal_close_short")

    def test_same_side_signal_while_not_flat_holds(self):
        strategy = self._strategy_with_signal(_signal("LONG"))
        intent = strategy.generate(_context(TradeState.LONG))
        self.assertEqual(intent.action, IntentAction.HOLD)
        self.assertEqual(intent.reason, "state_not_flat")

    def test_confirmation_pending_surfaces_as_hold(self):
        # Integration check: a real SignalGenerator with confirmation enabled
        # (the default) must arm rather than fire immediately, and the strategy
        # must surface that as a HOLD with the confirmation-pending reason.
        strategy = LayeredPumpStrategy()
        with patch.object(
            strategy._generator,
            "_evaluate_gates",
            return_value=("LONG", {"volume_spike": 3.0}, {}, {}, {"sentiment": 50.0, "crowd_extreme": 0.0}),
        ):
            intent = strategy.generate(_context(TradeState.FLAT))
        self.assertEqual(intent.action, IntentAction.HOLD)
        self.assertIn("layer_confirmation_pending", intent.reason)


if __name__ == "__main__":
    unittest.main()
