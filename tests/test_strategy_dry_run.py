import unittest

try:
    import numpy as np
    import pandas as pd

    from core.market_regime import MarketRegime
    from core.signal_generator import SignalConfig, SignalContext, SignalGenerator
    from core.volume_profile import VolumeProfileLevels

    HAS_DEPS = True
except Exception:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "numpy/pandas not installed")
class SignalGeneratorTests(unittest.TestCase):
    @staticmethod
    def _build_df() -> pd.DataFrame:
        n = 120
        idx = pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC")

        close = np.linspace(95.0, 112.0, n)
        close[-2] = 111.0
        close[-1] = 108.0

        df = pd.DataFrame(
            {
                "open": close + 0.3,
                "high": close + 1.2,
                "low": close - 1.1,
                "close": close,
                "volume": [10.0] * (n - 1) + [80.0],
            },
            index=idx,
        )

        # Minimal indicator fields used by layers.
        df["rsi"] = 60.0
        df.iloc[-1, df.columns.get_loc("rsi")] = 85.0
        df["volume_spike"] = 2.0
        df.iloc[-1, df.columns.get_loc("volume_spike")] = 8.0
        df["bb_upper"] = 106.0
        df["bb_lower"] = 94.0
        df["kc_upper"] = 106.0
        df["kc_lower"] = 94.0
        df["obv"] = np.linspace(100.0, 1200.0, n)
        df.iloc[-1, df.columns.get_loc("obv")] = 900.0
        df["cvd"] = np.linspace(80.0, 1100.0, n)
        df.iloc[-1, df.columns.get_loc("cvd")] = 850.0
        df["vwap"] = 104.0
        df["atr"] = 1.8

        # MSB helper: force bearish EMA20 cross on the last bar.
        df["ema20"] = 108.5
        return df

    def test_generate_short_signal_on_pump(self):
        df = self._build_df()

        signal_gen = SignalGenerator(SignalConfig())
        ctx = SignalContext(
            symbol="BTC/USDT",
            df=df,
            volume_profile=VolumeProfileLevels(poc=102.0, vah=109.0, val=98.0),
            regime=MarketRegime.PUMP,
            sentiment_index=78.0,
            sentiment_source="provided",
            funding_rate=0.001,
            long_short_ratio=1.2,
        )
        signal = signal_gen.generate(ctx)

        self.assertIsNotNone(signal)
        self.assertEqual(signal.side, "SHORT")
        self.assertGreater(signal.sl, signal.entry)
        self.assertLess(signal.tp, signal.entry)

        trace = signal.details.get("layer_trace", {})
        self.assertEqual(trace.get("failed_layer"), None)
        self.assertTrue(trace.get("layers", {}).get("layer1_pump_detection", {}).get("passed", False))
        self.assertTrue(trace.get("layers", {}).get("layer2_weakness_confirmation", {}).get("passed", False))
        self.assertTrue(trace.get("layers", {}).get("layer3_entry_location", {}).get("passed", False))
        self.assertTrue(trace.get("layers", {}).get("layer4_fake_filter", {}).get("passed", False))

    def test_sentiment_fallback_degraded_mode_is_explicit(self):
        df = self._build_df()

        signal_gen = SignalGenerator(SignalConfig())
        ctx = SignalContext(
            symbol="BTC/USDT",
            df=df,
            volume_profile=VolumeProfileLevels(poc=102.0, vah=109.0, val=98.0),
            regime=MarketRegime.PUMP,
            sentiment_index=None,
            sentiment_source="unavailable",
            funding_rate=None,
            long_short_ratio=None,
        )
        signal = signal_gen.generate(ctx)

        self.assertIsNotNone(signal)
        trace = signal.details.get("layer_trace", {})
        layer4 = trace.get("layers", {}).get("layer4_fake_filter", {}).get("details", {})
        self.assertEqual(layer4.get("sentiment_source"), "unavailable")
        self.assertEqual(layer4.get("degraded_mode"), 1.0)
        self.assertEqual(layer4.get("sentiment_fallback_used"), 1.0)


if __name__ == "__main__":
    unittest.main()
