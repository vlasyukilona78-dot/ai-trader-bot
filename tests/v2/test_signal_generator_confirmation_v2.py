from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from core.market_regime import MarketRegime
from core.signal_generator import SignalConfig, SignalContext, SignalGenerator


def _make_df(closes: list[float]) -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.001 for c in closes],
            "low": [c * 0.999 for c in closes],
            "close": closes,
            "volume": [100.0] * n,
        },
        index=idx,
    )


def _ctx(symbol: str, df: pd.DataFrame) -> SignalContext:
    return SignalContext(
        symbol=symbol,
        df=df,
        volume_profile=None,
        regime=MarketRegime.PANIC,
        sentiment_index=50.0,
        sentiment_source="fallback_neutral_50",
        funding_rate=None,
        long_short_ratio=None,
    )


_GATE_LAYERS = (
    "LONG",
    {"volume_spike": 3.0},
    {"price_down": 1.0},
    {"entry_ok": 1.0},
    {"sentiment": 50.0, "crowd_extreme": 0.0},
)


class SignalGeneratorConfirmationV2Tests(unittest.TestCase):
    def _gen(self, **overrides) -> SignalGenerator:
        defaults = {
            "confirmation_enabled": True,
            "confirmation_max_wait_bars": 3,
            "confirmation_invalidate_pct": 0.0015,
        }
        defaults.update(overrides)
        return SignalGenerator(SignalConfig(**defaults))

    def test_arm_then_confirm_next_bar(self):
        gen = self._gen()
        full = [100.0] * 40 + [100.0, 101.0]

        with patch.object(gen, "_evaluate_gates", return_value=_GATE_LAYERS):
            bar1 = _ctx("BTCUSDT", _make_df(full[:41]))
            result1 = gen.generate(bar1)
            self.assertIsNone(result1)
            self.assertIn("BTCUSDT", gen._pending)
            self.assertEqual(gen._pending["BTCUSDT"].armed_close, 100.0)
            self.assertEqual(gen.last_diagnostics["failed_layer"], "layer_confirmation_pending")

            bar2 = _ctx("BTCUSDT", _make_df(full[:42]))
            result2 = gen.generate(bar2)

        self.assertIsNotNone(result2)
        self.assertEqual(result2.side, "LONG")
        self.assertEqual(result2.entry, 101.0)
        self.assertNotIn("BTCUSDT", gen._pending)
        self.assertTrue(gen.last_diagnostics["layers"]["layer_confirmation"]["passed"])

    def test_same_bar_repeat_call_is_a_noop(self):
        gen = self._gen()
        full = [100.0] * 41

        with patch.object(gen, "_evaluate_gates", return_value=_GATE_LAYERS):
            df = _make_df(full)
            gen.generate(_ctx("BTCUSDT", df))
            pending_before = gen._pending["BTCUSDT"]
            result = gen.generate(_ctx("BTCUSDT", df))

        self.assertIsNone(result)
        pending_after = gen._pending["BTCUSDT"]
        self.assertEqual(pending_before.bars_waited, pending_after.bars_waited)
        self.assertEqual(pending_before.last_seen_bar_ts, pending_after.last_seen_bar_ts)

    def test_invalidation_drops_candidate(self):
        gen = self._gen()
        full = [100.0] * 40 + [100.0, 99.0]  # 99.0 < 100 * (1 - 0.0015) = 99.85

        with patch.object(gen, "_evaluate_gates", return_value=_GATE_LAYERS):
            gen.generate(_ctx("BTCUSDT", _make_df(full[:41])))
            result = gen.generate(_ctx("BTCUSDT", _make_df(full[:42])))

        self.assertIsNone(result)
        self.assertNotIn("BTCUSDT", gen._pending)
        self.assertEqual(gen.last_diagnostics["failed_layer"], "layer_confirmation_invalidated")

    def test_expiry_after_max_wait_bars(self):
        gen = self._gen(confirmation_max_wait_bars=2)
        full = [100.0] * 40 + [100.0, 100.0, 100.0]  # never confirms, never invalidates

        with patch.object(gen, "_evaluate_gates", return_value=_GATE_LAYERS):
            gen.generate(_ctx("BTCUSDT", _make_df(full[:41])))  # arm
            r1 = gen.generate(_ctx("BTCUSDT", _make_df(full[:42])))  # bars_waited -> 1
            self.assertIsNone(r1)
            self.assertIn("BTCUSDT", gen._pending)
            r2 = gen.generate(_ctx("BTCUSDT", _make_df(full[:43])))  # bars_waited -> 2, expires

        self.assertIsNone(r2)
        self.assertNotIn("BTCUSDT", gen._pending)
        self.assertEqual(gen.last_diagnostics["failed_layer"], "layer_confirmation_expired")

    def test_pending_state_is_isolated_per_symbol(self):
        gen = self._gen()
        full = [100.0] * 41

        with patch.object(gen, "_evaluate_gates", return_value=_GATE_LAYERS):
            gen.generate(_ctx("BTCUSDT", _make_df(full)))
            gen.generate(_ctx("ETHUSDT", _make_df(full)))

        self.assertIn("BTCUSDT", gen._pending)
        self.assertIn("ETHUSDT", gen._pending)
        self.assertIsNot(gen._pending["BTCUSDT"], gen._pending["ETHUSDT"])

    def test_confirmation_disabled_reproduces_immediate_fire(self):
        gen = self._gen(confirmation_enabled=False)
        df = _make_df([100.0] * 41)

        with patch.object(gen, "_evaluate_gates", return_value=_GATE_LAYERS):
            result = gen.generate(_ctx("BTCUSDT", df))

        self.assertIsNotNone(result)
        self.assertEqual(result.entry, 100.0)
        self.assertEqual(gen._pending, {})


if __name__ == "__main__":
    unittest.main()
