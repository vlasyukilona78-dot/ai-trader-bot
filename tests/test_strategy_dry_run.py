import unittest

try:
    import numpy as np
    import pandas as pd

    from core.signal_generator import SignalConfig, SignalContext, SignalGenerator
    from core.market_regime import MarketRegime
    from core.volume_profile import VolumeProfileLevels

    HAS_DEPS = True
except Exception:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "numpy/pandas not installed")
class SignalGeneratorTests(unittest.TestCase):
    def test_generate_short_signal_on_pump(self):
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

        signal_gen = SignalGenerator(SignalConfig())
        ctx = SignalContext(
            symbol="BTC/USDT",
            df=df,
            volume_profile=VolumeProfileLevels(poc=102.0, vah=109.0, val=98.0),
            regime=MarketRegime.PUMP,
            sentiment_index=78.0,
            funding_rate=0.001,
            long_short_ratio=1.2,
        )
        signal = signal_gen.generate(ctx)

        self.assertIsNotNone(signal)
        self.assertEqual(signal.side, "SHORT")
        self.assertGreater(signal.sl, signal.entry)
        self.assertLess(signal.tp, signal.entry)


if __name__ == "__main__":
    unittest.main()
