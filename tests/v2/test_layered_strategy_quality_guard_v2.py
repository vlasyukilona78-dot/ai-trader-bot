from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from core.signal_generator import SignalConfig
from trading.exchange.schemas import AccountSnapshot
from trading.market_data.reconciliation import ExchangeSnapshot
from trading.signals.layered_strategy import LayeredPumpStrategy
from trading.signals.signal_types import IntentAction
from trading.signals.strategy_interface import StrategyContext
from trading.state.models import TradeState


class LayeredStrategyQualityGuardV2Tests(unittest.TestCase):
    def _build_df(self, n: int = 120) -> pd.DataFrame:
        idx = pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC")
        close = np.linspace(100.0, 110.0, n)
        df = pd.DataFrame(
            {
                "open": close - 0.1,
                "high": close + 0.2,
                "low": close - 0.3,
                "close": close,
                "volume": np.linspace(10.0, 15.0, n),
            },
            index=idx,
        )
        df["rsi"] = 55.0
        df["volume_spike"] = 1.2
        df["bb_upper"] = df["close"] + 1.0
        df["bb_lower"] = df["close"] - 1.0
        df["kc_upper"] = df["close"] + 1.0
        df["kc_lower"] = df["close"] - 1.0
        df["obv"] = np.linspace(100.0, 200.0, n)
        df["cvd"] = np.linspace(80.0, 150.0, n)
        df["vwap"] = df["close"] - 0.2
        df["atr"] = 0.4
        df["ema20"] = df["close"] - 0.1
        df["ema50"] = df["close"] - 0.2
        df["hist"] = 0.0
        return df

    def test_strategy_holds_when_recent_frame_has_severe_gaps(self):
        df = self._build_df()
        broken = df.drop(df.index[[95, 96, 105, 106]])
        strategy = LayeredPumpStrategy(SignalConfig())
        context = StrategyContext(
            symbol="BTCUSDT",
            market_ohlcv=broken,
            mark_price=float(broken["close"].iloc[-1]),
            exchange=ExchangeSnapshot(
                symbol="BTCUSDT",
                account=AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0),
                positions=[],
                open_orders=[],
            ),
            synced_state=TradeState.FLAT,
            timeframe="1",
        )
        intent = strategy.generate(context)
        self.assertEqual(intent.action, IntentAction.HOLD)
        self.assertTrue(str(intent.reason).startswith("data_quality_guard_"))
        self.assertIn("frame_quality", intent.metadata)

    def test_strategy_holds_when_recent_frame_has_zero_volume_cluster(self):
        df = self._build_df()
        df.loc[df.index[-12:], "volume"] = 0.0
        strategy = LayeredPumpStrategy(SignalConfig())
        context = StrategyContext(
            symbol="BTCUSDT",
            market_ohlcv=df,
            mark_price=float(df["close"].iloc[-1]),
            exchange=ExchangeSnapshot(
                symbol="BTCUSDT",
                account=AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0),
                positions=[],
                open_orders=[],
            ),
            synced_state=TradeState.FLAT,
            timeframe="1",
        )
        intent = strategy.generate(context)
        self.assertEqual(intent.action, IntentAction.HOLD)
        self.assertEqual(str(intent.reason), "data_quality_guard_recent_zero_volume_cluster")
        self.assertIn("frame_quality", intent.metadata)


if __name__ == "__main__":
    unittest.main()
