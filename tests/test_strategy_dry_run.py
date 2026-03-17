import json
import os
import subprocess
import tempfile
import unittest

try:
    import numpy as np
    import pandas as pd
    from app.main import _strategy_audit_log_payload

    from core.market_regime import MarketRegime
    from core.signal_generator import SignalConfig, SignalContext, SignalGenerator
    from core.volume_profile import VolumeProfileLevels
    from trading.exchange.schemas import AccountSnapshot
    from trading.market_data.reconciliation import ExchangeSnapshot
    from trading.signals.layered_strategy import LayeredPumpStrategy
    from trading.signals.runtime_source_adapter import build_runtime_signal_inputs
    from trading.signals.strategy_audit import StrategyAuditCollector
    from trading.signals.calibration_control import (
        CalibrationGuardrails,
        aggregate_observation,
        assess_window_quality,
        compare_observation_windows,
        recommend_calibration_step,
    )
    from trading.signals.strategy_interface import StrategyContext
    from trading.state.models import TradeState

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
        self.assertTrue(trace.get("layers", {}).get("layer5_tp_sl", {}).get("passed", False))

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

    def test_layer1_pump_detection_semantics_pass(self):
        df = self._build_df()
        signal_gen = SignalGenerator(SignalConfig())

        side, layer1 = signal_gen._layer1_pump_detection(df)

        self.assertEqual(side, "SHORT")
        self.assertEqual(layer1.get("passed"), 1.0)
        self.assertEqual(layer1.get("failed_reason"), "none")
        self.assertIn("rsi_high", layer1)
        self.assertIn("volume_spike", layer1)
        self.assertIn("above_bollinger_upper", layer1)
        self.assertIn("above_keltner_upper", layer1)
        self.assertIn("pump_context_strength", layer1)
        self.assertEqual(layer1.get("volume_spike_high"), 1.0)
        self.assertEqual(layer1.get("missing_conditions"), "")
        self.assertEqual(layer1.get("soft_pass_candidate"), 0.0)
        self.assertEqual(layer1.get("pump_bar_offset"), 0)
        self.assertEqual(layer1.get("rsi_high_threshold_used"), float(signal_gen.config.rsi_high))

    def test_layer1_hard_pass_clears_missing_conditions_when_two_of_three_conditions_suffice(self):
        df = self._build_df()
        df.iloc[-1, df.columns.get_loc("rsi")] = 72.0
        df.iloc[-1, df.columns.get_loc("volume_spike")] = 1.2
        df.iloc[-1, df.columns.get_loc("close")] = float(df.iloc[-1]["kc_upper"]) + 1.0

        signal_gen = SignalGenerator(SignalConfig())

        side, layer1 = signal_gen._layer1_pump_detection(df)

        self.assertEqual(side, "SHORT")
        self.assertEqual(layer1.get("passed"), 1.0)
        self.assertEqual(layer1.get("failed_reason"), "none")
        self.assertEqual(layer1.get("missing_conditions"), "")
        self.assertEqual(layer1.get("rsi_high"), 1.0)
        self.assertEqual(layer1.get("volume_spike_high"), 0.0)
        self.assertEqual(layer1.get("upper_band_breakout"), 1.0)
        self.assertEqual(layer1.get("volume_spike_threshold_used"), float(signal_gen.config.volume_spike_threshold))
        self.assertEqual(
            layer1.get("layer1_subconditions_state", {}),
            {
                "rsi_high": True,
                "volume_spike_high": False,
                "upper_band_breakout": True,
                "above_bollinger_upper": True,
                "above_keltner_upper": True,
            },
        )

    def test_layer1_pump_detection_semantics_fail(self):
        df = self._build_df()
        df.iloc[-1, df.columns.get_loc("rsi")] = 55.0
        df.iloc[-1, df.columns.get_loc("volume_spike")] = 1.1
        df.iloc[-1, df.columns.get_loc("close")] = 103.0

        signal_gen = SignalGenerator(SignalConfig())
        side, layer1 = signal_gen._layer1_pump_detection(df)

        self.assertIsNone(side)
        self.assertEqual(layer1.get("passed"), 0.0)
        self.assertTrue(str(layer1.get("failed_reason", "")).startswith("missing:"))
        missing = str(layer1.get("missing_conditions", ""))
        self.assertIn("rsi_high", missing)
        self.assertIn("volume_spike", missing)
        self.assertIn("upper_band_breakout", missing)
        self.assertLessEqual(float(layer1.get("pump_context_strength", 0.0)), 0.34)

    def test_layer1_pump_detection_soft_pass_candidate_when_only_band_break_is_missing(self):
        df = self._build_df()
        df.iloc[-1, df.columns.get_loc("bb_upper")] = 120.0
        df.iloc[-1, df.columns.get_loc("kc_upper")] = 120.0

        signal_gen = SignalGenerator(SignalConfig())
        side, layer1 = signal_gen._layer1_pump_detection(df)

        self.assertIsNone(side)
        self.assertEqual(layer1.get("passed"), 0.0)
        self.assertEqual(layer1.get("failed_reason"), "missing:upper_band_breakout")
        self.assertEqual(layer1.get("soft_pass_candidate"), 1.0)
        self.assertEqual(layer1.get("volume_spike_high"), 1.0)
        self.assertEqual(layer1.get("pump_bar_offset"), 0)
        self.assertEqual(
            layer1.get("layer1_subconditions_state", {}),
            {
                "rsi_high": True,
                "volume_spike_high": True,
                "upper_band_breakout": False,
                "above_bollinger_upper": False,
                "above_keltner_upper": False,
            },
        )

    def test_layer1_pump_detection_recent_pump_window_passes(self):
        df = self._build_df()
        df.iloc[-1, df.columns.get_loc("rsi")] = 55.0
        df.iloc[-1, df.columns.get_loc("volume_spike")] = 1.1
        df.iloc[-1, df.columns.get_loc("close")] = 103.0

        signal_gen = SignalGenerator(SignalConfig(layer1_pump_lookback_bars=2))
        side, layer1 = signal_gen._layer1_pump_detection(df)

        self.assertEqual(side, "SHORT")
        self.assertEqual(layer1.get("passed"), 1.0)
        self.assertEqual(layer1.get("failed_reason"), "none")
        self.assertEqual(layer1.get("pump_bar_offset"), 1)
        self.assertEqual(layer1.get("soft_pass_candidate"), 0.0)

    def test_layer1_pump_detection_recent_soft_pass_window_passes_when_enabled(self):
        df = self._build_df()
        for offset in range(1, 13):
            df.iloc[-offset, df.columns.get_loc("rsi")] = 55.0
            df.iloc[-offset, df.columns.get_loc("volume_spike")] = 1.1
            df.iloc[-offset, df.columns.get_loc("bb_upper")] = 120.0
            df.iloc[-offset, df.columns.get_loc("kc_upper")] = 120.0
        df.iloc[-8, df.columns.get_loc("rsi")] = 85.0
        df.iloc[-8, df.columns.get_loc("volume_spike")] = 8.0
        df.iloc[-8, df.columns.get_loc("bb_upper")] = 120.0
        df.iloc[-8, df.columns.get_loc("kc_upper")] = 120.0

        strict_gen = SignalGenerator(SignalConfig(layer1_pump_lookback_bars=12))
        runtime_gen = SignalGenerator(SignalConfig(layer1_pump_lookback_bars=12, layer1_soft_pass_enabled=True))

        strict_side, strict_layer1 = strict_gen._layer1_pump_detection(df)
        runtime_side, runtime_layer1 = runtime_gen._layer1_pump_detection(df)

        self.assertIsNone(strict_side)
        self.assertEqual(strict_layer1.get("soft_pass_used"), 0.0)
        self.assertEqual(runtime_side, "SHORT")
        self.assertEqual(runtime_layer1.get("passed"), 1.0)
        self.assertEqual(runtime_layer1.get("soft_pass_candidate"), 1.0)
        self.assertEqual(runtime_layer1.get("soft_pass_used"), 1.0)
        self.assertEqual(runtime_layer1.get("soft_pass_reason"), "upper_band_breakout")
        self.assertEqual(runtime_layer1.get("failed_reason"), "none")
        self.assertEqual(runtime_layer1.get("missing_conditions"), "")
        self.assertEqual(runtime_layer1.get("pump_bar_offset"), 7)

    def test_layer1_window_context_soft_pass_passes_when_pump_is_split_across_bars(self):
        df = self._build_df()
        for offset in range(1, 13):
            df.iloc[-offset, df.columns.get_loc("rsi")] = 55.0
            df.iloc[-offset, df.columns.get_loc("volume_spike")] = 1.1
            df.iloc[-offset, df.columns.get_loc("bb_upper")] = 120.0
            df.iloc[-offset, df.columns.get_loc("kc_upper")] = 120.0
            df.iloc[-offset, df.columns.get_loc("close")] = 103.0

        df.iloc[-10, df.columns.get_loc("rsi")] = 85.0
        df.iloc[-6, df.columns.get_loc("volume_spike")] = 8.0
        df.iloc[-3, df.columns.get_loc("close")] = 121.0

        strict_gen = SignalGenerator(SignalConfig(layer1_pump_lookback_bars=12))
        runtime_gen = SignalGenerator(SignalConfig(layer1_pump_lookback_bars=12, layer1_soft_pass_enabled=True))

        strict_side, strict_layer1 = strict_gen._layer1_pump_detection(df)
        runtime_side, runtime_layer1 = runtime_gen._layer1_pump_detection(df)

        self.assertIsNone(strict_side)
        self.assertEqual(strict_layer1.get("soft_pass_used"), 0.0)
        self.assertEqual(runtime_side, "SHORT")
        self.assertEqual(runtime_layer1.get("soft_pass_used"), 1.0)
        self.assertEqual(runtime_layer1.get("soft_pass_reason"), "window_pump_context")
        self.assertEqual(runtime_layer1.get("failed_reason"), "none")
        self.assertEqual(runtime_layer1.get("missing_conditions"), "")
        self.assertEqual(
            runtime_layer1.get("layer1_subconditions_state", {}),
            {
                "rsi_high": True,
                "volume_spike_high": True,
                "upper_band_breakout": True,
                "above_bollinger_upper": True,
                "above_keltner_upper": True,
            },
        )

    def test_layer1_near_upper_band_soft_pass_passes_when_enabled(self):
        df = self._build_df()
        df.iloc[-1, df.columns.get_loc("rsi")] = 52.0
        df.iloc[-1, df.columns.get_loc("volume_spike")] = 1.45
        df.iloc[-1, df.columns.get_loc("close")] = 105.95
        df.iloc[-1, df.columns.get_loc("bb_upper")] = 106.0
        df.iloc[-1, df.columns.get_loc("kc_upper")] = 106.0

        strict_gen = SignalGenerator(SignalConfig())
        runtime_gen = SignalGenerator(SignalConfig(layer1_soft_pass_enabled=True))

        strict_side, strict_layer1 = strict_gen._layer1_pump_detection(df)
        runtime_side, runtime_layer1 = runtime_gen._layer1_pump_detection(df)

        self.assertIsNone(strict_side)
        self.assertEqual(strict_layer1.get("soft_pass_used"), 0.0)
        self.assertEqual(runtime_side, "SHORT")
        self.assertEqual(runtime_layer1.get("soft_pass_used"), 1.0)
        self.assertEqual(runtime_layer1.get("soft_pass_reason"), "near_upper_band_context")
        self.assertEqual(runtime_layer1.get("failed_reason"), "none")
        self.assertEqual(runtime_layer1.get("missing_conditions"), "")

    def test_layer2_weakness_semantics_pass(self):
        df = self._build_df()
        signal_gen = SignalGenerator(SignalConfig())

        passed, layer2 = signal_gen._layer2_weakness_confirmation(df, "SHORT")

        self.assertTrue(passed)
        self.assertEqual(layer2.get("passed"), 1.0)
        self.assertEqual(layer2.get("failed_reason"), "none")
        self.assertIn("price_up_or_near_high", layer2)
        self.assertIn("price_up", layer2)
        self.assertIn("near_high_context", layer2)
        self.assertIn("obv_bearish_divergence", layer2)
        self.assertIn("cvd_bearish_divergence", layer2)
        self.assertIn("close_last_used", layer2)
        self.assertIn("close_ref_used", layer2)
        self.assertIn("obv_last_used", layer2)
        self.assertIn("obv_ref_used", layer2)
        self.assertIn("cvd_last_used", layer2)
        self.assertIn("cvd_ref_used", layer2)
        self.assertIn("weakness_lookback_used", layer2)
        self.assertIn("weakness_strength", layer2)
        self.assertIn("layer2_subconditions_state", layer2)

    def test_layer2_weakness_semantics_fail(self):
        df = self._build_df()
        df.iloc[-1, df.columns.get_loc("close")] = 99.0
        df.iloc[-1, df.columns.get_loc("obv")] = 1400.0
        df.iloc[-1, df.columns.get_loc("cvd")] = 1300.0
        signal_gen = SignalGenerator(SignalConfig())

        passed, layer2 = signal_gen._layer2_weakness_confirmation(df, "SHORT")

        self.assertFalse(passed)
        self.assertEqual(layer2.get("passed"), 0.0)
        self.assertTrue(str(layer2.get("failed_reason", "")).startswith("missing:"))
        missing = str(layer2.get("missing_conditions", ""))
        self.assertIn("price_up_or_near_high", missing)
        self.assertIn("obv_bearish_divergence", missing)
        self.assertIn("cvd_bearish_divergence", missing)
        self.assertEqual(float(layer2.get("weakness_strength", 1.0)), 0.0)
        self.assertEqual(
            layer2.get("layer2_subconditions_state", {}),
            {
                "price_up_or_near_high": False,
                "price_up": False,
                "near_high_context": False,
                "obv_bearish_divergence": False,
                "cvd_bearish_divergence": False,
            },
        )

    def test_layer3_entry_location_semantics_pass(self):
        df = self._build_df()
        signal_gen = SignalGenerator(SignalConfig())
        vp = VolumeProfileLevels(poc=102.0, vah=109.0, val=98.0)

        passed, layer3 = signal_gen._layer3_entry_location(df, "SHORT", vp)

        self.assertTrue(passed)
        self.assertEqual(layer3.get("entry_location_passed"), 1.0)
        self.assertEqual(layer3.get("failed_reason"), "none")
        self.assertIn("below_vah_or_rejected_from_vah", layer3)
        self.assertIn("near_poc_or_value_area_context", layer3)
        self.assertIn("msb_bearish_confirmed", layer3)
        self.assertIn("entry_location_strength", layer3)

    def test_layer3_entry_location_semantics_fail(self):
        df = self._build_df()
        df.iloc[-2, df.columns.get_loc("close")] = 114.0
        df.iloc[-1, df.columns.get_loc("close")] = 115.0
        signal_gen = SignalGenerator(SignalConfig())
        vp = VolumeProfileLevels(poc=102.0, vah=109.0, val=98.0)

        passed, layer3 = signal_gen._layer3_entry_location(df, "SHORT", vp)

        self.assertFalse(passed)
        self.assertEqual(layer3.get("entry_location_passed"), 0.0)
        self.assertTrue(str(layer3.get("failed_reason", "")).startswith("missing:"))
        missing = str(layer3.get("missing_conditions", ""))
        self.assertIn("below_vah_or_rejected_from_vah", missing)
        self.assertIn("near_poc_or_value_area_context", missing)
        self.assertIn("msb_bearish_confirmed", missing)
        self.assertLessEqual(float(layer3.get("entry_location_strength", 1.0)), 0.34)

    def test_layer4_fake_filter_pass_with_degraded_unavailable_sources(self):
        df = self._build_df()
        signal_gen = SignalGenerator(SignalConfig())

        passed, layer4 = signal_gen._layer4_fake_filter(
            df=df,
            side="SHORT",
            sentiment_index=None,
            sentiment_source="unavailable",
            funding_rate=None,
            long_short_ratio=None,
            open_interest=None,
            open_interest_source="unavailable",
        )

        self.assertTrue(passed)
        self.assertEqual(layer4.get("passed"), 1.0)
        self.assertEqual(layer4.get("degraded_mode"), 1.0)
        self.assertEqual(layer4.get("failed_reason"), "none")
        self.assertEqual(layer4.get("price_above_vwap"), 1.0)
        self.assertIn("source_flags", layer4)
        self.assertEqual(layer4.get("source_flags", {}).get("sentiment_quality"), "unavailable")
        self.assertEqual(layer4.get("source_flags", {}).get("funding_quality"), "unavailable")

    def test_layer4_fake_filter_pass_with_softened_sentiment_threshold(self):
        df = self._build_df()
        cfg = SignalConfig()
        self.assertLess(cfg.fake_filter_sentiment_euphoric_soft, cfg.sentiment_bullish_threshold)

        signal_gen = SignalGenerator(cfg)
        passed, layer4 = signal_gen._layer4_fake_filter(
            df=df,
            side="SHORT",
            sentiment_index=cfg.fake_filter_sentiment_euphoric_soft,
            sentiment_source="live_api",
            funding_rate=0.001,
            funding_source="live_api",
            long_short_ratio=1.2,
            long_short_ratio_source="live_api",
            open_interest_ratio=1.5,
            oi_source="live_api",
            open_interest_source="live_api",
        )

        self.assertTrue(passed)
        self.assertEqual(layer4.get("sentiment_euphoric"), 1.0)
        self.assertEqual(layer4.get("sentiment_euphoric_threshold"), float(cfg.fake_filter_sentiment_euphoric_soft))

    def test_layer4_fake_filter_price_structure_still_blocks_after_sentiment_softening(self):
        df = self._build_df()
        df.iloc[-1, df.columns.get_loc("vwap")] = 120.0

        signal_gen = SignalGenerator(SignalConfig())
        passed, layer4 = signal_gen._layer4_fake_filter(
            df=df,
            side="SHORT",
            sentiment_index=90.0,
            sentiment_source="live_api",
            funding_rate=0.001,
            funding_source="live_api",
            long_short_ratio=1.2,
            long_short_ratio_source="live_api",
            open_interest_ratio=1.5,
            oi_source="live_api",
            open_interest_source="live_api",
        )

        self.assertFalse(passed)
        self.assertIn("price_above_vwap", str(layer4.get("missing_conditions", "")))
        self.assertEqual(layer4.get("sentiment_euphoric"), 1.0)
        self.assertEqual(layer4.get("blocker_price_above_vwap"), 1.0)
        self.assertEqual(layer4.get("fail_due_to_price_structure"), 1.0)

    def test_layer4_fake_filter_derivatives_context_unchanged_after_sentiment_softening(self):
        df = self._build_df()

        signal_gen = SignalGenerator(SignalConfig())
        passed, layer4 = signal_gen._layer4_fake_filter(
            df=df,
            side="SHORT",
            sentiment_index=90.0,
            sentiment_source="live_api",
            funding_rate=-0.01,
            funding_source="live_api",
            long_short_ratio=0.9,
            long_short_ratio_source="live_api",
            open_interest_ratio=1.0,
            oi_source="live_api",
            open_interest_source="live_api",
        )

        self.assertFalse(passed)
        missing = str(layer4.get("missing_conditions", ""))
        self.assertIn("funding_supports_short", missing)
        self.assertIn("long_short_ratio_extreme", missing)
        self.assertIn("oi_overheated", missing)
        self.assertEqual(layer4.get("fail_due_to_derivatives_context"), 1.0)
        self.assertEqual(layer4.get("fail_due_to_price_structure"), 0.0)
    def test_layer4_fake_filter_pass_with_softened_funding_threshold(self):
        df = self._build_df()
        cfg = SignalConfig()
        self.assertGreater(cfg.fake_filter_funding_supports_short_soft, cfg.funding_tolerance)

        signal_gen = SignalGenerator(cfg)
        passed, layer4 = signal_gen._layer4_fake_filter(
            df=df,
            side="SHORT",
            sentiment_index=90.0,
            sentiment_source="live_api",
            funding_rate=-0.0004,
            funding_source="live_api",
            long_short_ratio=1.2,
            long_short_ratio_source="live_api",
            open_interest_ratio=1.5,
            oi_source="live_api",
            open_interest_source="live_api",
        )

        self.assertTrue(passed)
        self.assertEqual(layer4.get("funding_supports_short"), 1.0)
        self.assertEqual(layer4.get("funding_supports_short_threshold"), float(cfg.fake_filter_funding_supports_short_soft))

    def test_layer4_fake_filter_vwap_still_blocks_after_funding_softening(self):
        df = self._build_df()
        df.iloc[-1, df.columns.get_loc("vwap")] = 120.0

        signal_gen = SignalGenerator(SignalConfig())
        passed, layer4 = signal_gen._layer4_fake_filter(
            df=df,
            side="SHORT",
            sentiment_index=90.0,
            sentiment_source="live_api",
            funding_rate=-0.0004,
            funding_source="live_api",
            long_short_ratio=1.2,
            long_short_ratio_source="live_api",
            open_interest_ratio=1.5,
            oi_source="live_api",
            open_interest_source="live_api",
        )

        self.assertFalse(passed)
        self.assertIn("price_above_vwap", str(layer4.get("missing_conditions", "")))
        self.assertEqual(layer4.get("funding_supports_short"), 1.0)
        self.assertEqual(layer4.get("blocker_price_above_vwap"), 1.0)
        self.assertEqual(layer4.get("fail_due_to_price_structure"), 1.0)

    def test_layer4_fake_filter_lsr_oi_unchanged_after_funding_softening(self):
        df = self._build_df()

        signal_gen = SignalGenerator(SignalConfig())
        passed, layer4 = signal_gen._layer4_fake_filter(
            df=df,
            side="SHORT",
            sentiment_index=90.0,
            sentiment_source="live_api",
            funding_rate=-0.0004,
            funding_source="live_api",
            long_short_ratio=0.9,
            long_short_ratio_source="live_api",
            open_interest_ratio=1.0,
            oi_source="live_api",
            open_interest_source="live_api",
        )

        self.assertFalse(passed)
        missing = str(layer4.get("missing_conditions", ""))
        self.assertIn("long_short_ratio_extreme", missing)
        self.assertIn("oi_overheated", missing)
        self.assertNotIn("funding_supports_short", missing)
        self.assertEqual(layer4.get("funding_supports_short"), 1.0)
        self.assertEqual(layer4.get("fail_due_to_derivatives_context"), 1.0)

    def test_layer4_fake_filter_pass_with_softened_lsr_threshold(self):
        df = self._build_df()
        cfg = SignalConfig()

        signal_gen = SignalGenerator(cfg)
        passed, layer4 = signal_gen._layer4_fake_filter(
            df=df,
            side="SHORT",
            sentiment_index=90.0,
            sentiment_source="live_api",
            funding_rate=-0.0004,
            funding_source="live_api",
            long_short_ratio=cfg.fake_filter_lsr_extreme_soft,
            long_short_ratio_source="live_api",
            open_interest_ratio=1.5,
            oi_source="live_api",
            open_interest_source="live_api",
        )

        self.assertTrue(passed)
        self.assertEqual(layer4.get("long_short_ratio_extreme"), 1.0)
        self.assertEqual(layer4.get("long_short_ratio_extreme_threshold"), float(cfg.fake_filter_lsr_extreme_soft))

    def test_layer4_fake_filter_vwap_still_blocks_after_lsr_softening(self):
        df = self._build_df()
        df.iloc[-1, df.columns.get_loc("vwap")] = 120.0

        cfg = SignalConfig()
        signal_gen = SignalGenerator(cfg)
        passed, layer4 = signal_gen._layer4_fake_filter(
            df=df,
            side="SHORT",
            sentiment_index=90.0,
            sentiment_source="live_api",
            funding_rate=-0.0004,
            funding_source="live_api",
            long_short_ratio=cfg.fake_filter_lsr_extreme_soft,
            long_short_ratio_source="live_api",
            open_interest_ratio=1.5,
            oi_source="live_api",
            open_interest_source="live_api",
        )

        self.assertFalse(passed)
        self.assertIn("price_above_vwap", str(layer4.get("missing_conditions", "")))
        self.assertEqual(layer4.get("long_short_ratio_extreme"), 1.0)
        self.assertEqual(layer4.get("blocker_price_above_vwap"), 1.0)
        self.assertEqual(layer4.get("fail_due_to_price_structure"), 1.0)

    def test_layer4_fake_filter_oi_unchanged_after_lsr_softening(self):
        df = self._build_df()

        cfg = SignalConfig()
        signal_gen = SignalGenerator(cfg)
        passed, layer4 = signal_gen._layer4_fake_filter(
            df=df,
            side="SHORT",
            sentiment_index=90.0,
            sentiment_source="live_api",
            funding_rate=-0.0004,
            funding_source="live_api",
            long_short_ratio=cfg.fake_filter_lsr_extreme_soft,
            long_short_ratio_source="live_api",
            open_interest_ratio=1.0,
            oi_source="live_api",
            open_interest_source="live_api",
        )

        self.assertFalse(passed)
        missing = str(layer4.get("missing_conditions", ""))
        self.assertIn("oi_overheated", missing)
        self.assertNotIn("long_short_ratio_extreme", missing)
        self.assertEqual(layer4.get("long_short_ratio_extreme"), 1.0)
        self.assertEqual(layer4.get("fail_due_to_derivatives_context"), 1.0)

    def test_layer4_fake_filter_fail_with_missing_conditions(self):
        df = self._build_df()
        df.iloc[-1, df.columns.get_loc("vwap")] = 120.0

        signal_gen = SignalGenerator(SignalConfig())
        passed, layer4 = signal_gen._layer4_fake_filter(
            df=df,
            side="SHORT",
            sentiment_index=55.0,
            sentiment_source="provided",
            funding_rate=-0.01,
            long_short_ratio=0.9,
            open_interest=100.0,
            open_interest_source="provided",
        )

        self.assertFalse(passed)
        self.assertEqual(layer4.get("passed"), 0.0)
        self.assertTrue(str(layer4.get("failed_reason", "")).startswith("missing:"))
        missing = str(layer4.get("missing_conditions", ""))
        self.assertIn("price_above_vwap", missing)
        self.assertIn("sentiment_euphoric", missing)
        self.assertIn("funding_supports_short", missing)
        self.assertIn("long_short_ratio_extreme", missing)
        self.assertEqual(layer4.get("blocker_price_above_vwap"), 1.0)
        self.assertEqual(layer4.get("fail_due_to_price_structure"), 1.0)
        self.assertEqual(layer4.get("hard_fail"), 1.0)

    def test_layer4_soft_pass_candidate_diagnostic_only(self):
        df = self._build_df()

        signal_gen = SignalGenerator(SignalConfig())
        passed, layer4 = signal_gen._layer4_fake_filter(
            df=df,
            side="SHORT",
            sentiment_index=55.0,
            sentiment_source="live_api",
            funding_rate=0.001,
            funding_source="live_api",
            long_short_ratio=0.9,
            long_short_ratio_source="live_api",
            open_interest=100.0,
            open_interest_source="live_api",
            oi_source="live_api",
        )

        self.assertFalse(passed)
        self.assertEqual(layer4.get("passed"), 0.0)
        self.assertEqual(layer4.get("price_above_vwap"), 1.0)
        self.assertEqual(layer4.get("soft_pass_candidate"), 1.0)
        self.assertEqual(layer4.get("fail_due_to_sentiment"), 1.0)
        self.assertEqual(layer4.get("fail_due_to_derivatives_context"), 1.0)
        self.assertEqual(layer4.get("hard_fail"), 1.0)

    def test_generate_stops_at_layer4_fake_filter(self):
        df = self._build_df()

        signal_gen = SignalGenerator(SignalConfig())
        ctx = SignalContext(
            symbol="BTC/USDT",
            df=df,
            volume_profile=VolumeProfileLevels(poc=102.0, vah=109.0, val=98.0),
            regime=MarketRegime.PUMP,
            sentiment_index=55.0,
            sentiment_source="provided",
            funding_rate=-0.01,
            long_short_ratio=0.9,
            open_interest=100.0,
            open_interest_source="provided",
        )

        signal = signal_gen.generate(ctx)

        self.assertIsNone(signal)
        trace = signal_gen.last_diagnostics
        self.assertEqual(trace.get("failed_layer"), "layer4_fake_filter")
        self.assertTrue(trace.get("layers", {}).get("layer1_pump_detection", {}).get("passed", False))
        self.assertTrue(trace.get("layers", {}).get("layer2_weakness_confirmation", {}).get("passed", False))
        self.assertTrue(trace.get("layers", {}).get("layer3_entry_location", {}).get("passed", False))
        self.assertTrue(trace.get("layers", {}).get("layer4_fake_filter", {}).get("passed") in (0, 0.0, False))

    def test_layer5_tp_sl_pass_with_vp_target(self):
        df = self._build_df()
        signal_gen = SignalGenerator(SignalConfig())
        vp = VolumeProfileLevels(poc=102.0, vah=109.0, val=98.0)

        passed, layer5 = signal_gen._layer5_tp_sl_levels(df, "SHORT", vp)

        self.assertTrue(passed)
        self.assertEqual(layer5.get("passed"), 1.0)
        self.assertEqual(layer5.get("volume_profile_available"), 1.0)
        self.assertEqual(layer5.get("fallback_rr_used"), 0.0)
        self.assertEqual(layer5.get("tp_reference"), "vp_poc")
        self.assertEqual(layer5.get("stop_above_invalidation"), 1.0)
        self.assertEqual(layer5.get("tp_at_poc_or_better"), 1.0)

    def test_layer5_tp_sl_pass_with_rr_fallback(self):
        df = self._build_df()
        signal_gen = SignalGenerator(SignalConfig())
        vp = None

        passed, layer5 = signal_gen._layer5_tp_sl_levels(df, "SHORT", vp)

        self.assertTrue(passed)
        self.assertEqual(layer5.get("passed"), 1.0)
        self.assertEqual(layer5.get("volume_profile_available"), 0.0)
        self.assertEqual(layer5.get("fallback_rr_used"), 1.0)
        self.assertEqual(layer5.get("tp_reference"), "rr_fallback")
        self.assertGreater(float(layer5.get("risk_reward_ratio", 0.0)), 0.02)

    def test_layer5_tp_sl_fail_with_missing_conditions(self):
        df = self._build_df()
        signal_gen = SignalGenerator(SignalConfig())
        vp = VolumeProfileLevels(poc=107.99, vah=109.0, val=98.0)

        passed, layer5 = signal_gen._layer5_tp_sl_levels(df, "SHORT", vp)

        self.assertFalse(passed)
        self.assertEqual(layer5.get("passed"), 0.0)
        self.assertTrue(str(layer5.get("failed_reason", "")).startswith("missing:"))
        missing = str(layer5.get("missing_conditions", ""))
        self.assertIn("risk_reward_ratio_soft_min", missing)
        self.assertLess(float(layer5.get("risk_reward_ratio", 1.0)), 0.02)

    def test_generate_stops_at_layer5_tp_sl(self):
        df = self._build_df()

        signal_gen = SignalGenerator(SignalConfig())
        ctx = SignalContext(
            symbol="BTC/USDT",
            df=df,
            volume_profile=VolumeProfileLevels(poc=107.99, vah=109.0, val=98.0),
            regime=MarketRegime.PUMP,
            sentiment_index=78.0,
            sentiment_source="provided",
            funding_rate=0.001,
            long_short_ratio=1.2,
            open_interest=100.0,
            open_interest_source="provided",
        )

        signal = signal_gen.generate(ctx)

        self.assertIsNone(signal)
        trace = signal_gen.last_diagnostics
        self.assertEqual(trace.get("failed_layer"), "layer5_tp_sl")
        self.assertTrue(trace.get("layers", {}).get("layer4_fake_filter", {}).get("passed", False))
        self.assertTrue(trace.get("layers", {}).get("layer5_tp_sl", {}).get("passed") in (0, 0.0, False))

    def test_layer4_live_source_quality_with_canonical_inputs(self):
        df = self._build_df()
        signal_gen = SignalGenerator(SignalConfig())

        passed, layer4 = signal_gen._layer4_fake_filter(
            df=df,
            side="SHORT",
            sentiment_index=10.0,
            sentiment_source="live_api",
            funding_rate=0.001,
            long_short_ratio=1.2,
            open_interest=100.0,
            open_interest_source="live_api",
            sentiment_value=82.0,
            sentiment_degraded=False,
            funding_source="live_api",
            funding_degraded=False,
            long_short_ratio_source="live_api",
            long_short_ratio_degraded=False,
            open_interest_ratio=1.5,
            oi_source="live_api",
            oi_degraded=False,
        )

        self.assertTrue(passed)
        flags = layer4.get("source_flags", {})
        self.assertEqual(flags.get("sentiment_quality"), "live")
        self.assertEqual(flags.get("funding_quality"), "live")
        self.assertEqual(flags.get("long_short_ratio_quality"), "live")
        self.assertEqual(flags.get("open_interest_quality"), "live")
        self.assertEqual(layer4.get("sentiment_euphoric"), 1.0)

    def test_generate_trace_exposes_regime_and_layer4_source_quality(self):
        df = self._build_df()
        signal_gen = SignalGenerator(SignalConfig())

        ctx = SignalContext(
            symbol="BTC/USDT",
            df=df,
            volume_profile=VolumeProfileLevels(poc=102.0, vah=109.0, val=98.0),
            regime=MarketRegime.PUMP,
            sentiment_index=None,
            sentiment_value=79.0,
            sentiment_source="live_api",
            sentiment_degraded=False,
            funding_rate=0.001,
            funding_source="live_api",
            funding_degraded=False,
            long_short_ratio=1.2,
            long_short_ratio_source="live_api",
            long_short_ratio_degraded=False,
            open_interest=120.0,
            open_interest_ratio=1.5,
            oi_source="live_api",
            oi_degraded=False,
            open_interest_source="live_api",
            news_veto=False,
            news_source="live_news",
            news_degraded=False,
        )
        signal = signal_gen.generate(ctx)

        self.assertIsNotNone(signal)
        trace = signal.details.get("layer_trace", {})
        regime_flags = trace.get("layers", {}).get("regime_filter", {}).get("details", {}).get("source_flags", {})
        layer4_flags = trace.get("layers", {}).get("layer4_fake_filter", {}).get("details", {}).get("source_flags", {})
        self.assertEqual(regime_flags.get("news_quality"), "live")
        self.assertEqual(layer4_flags.get("sentiment_quality"), "live")
        self.assertEqual(layer4_flags.get("funding_quality"), "live")

    def test_layered_strategy_metadata_keeps_source_visibility(self):
        df = self._build_df()
        strategy = LayeredPumpStrategy(SignalConfig())
        exchange = ExchangeSnapshot(
            symbol="BTC/USDT",
            account=AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0),
            positions=[],
            open_orders=[],
        )

        intent = strategy.generate(
            StrategyContext(
                symbol="BTC/USDT",
                market_ohlcv=df,
                mark_price=float(df.iloc[-1]["close"]),
                exchange=exchange,
                synced_state=TradeState.FLAT,
                sentiment_value=79.0,
                sentiment_source="live_api",
                sentiment_degraded=False,
                funding_rate=0.001,
                funding_source="live_api",
                funding_degraded=False,
                long_short_ratio=1.2,
                long_short_ratio_source="live_api",
                long_short_ratio_degraded=False,
                open_interest_ratio=1.5,
                oi_source="live_api",
                oi_degraded=False,
                news_veto=False,
                news_source="live_news",
                news_degraded=False,
            )
        )

        self.assertIsInstance(intent.metadata, dict)
        source_quality = intent.metadata.get("source_quality", {})
        self.assertIn("regime_filter", source_quality)
        self.assertIn("layer4_fake_filter", source_quality)
    def test_runtime_source_adapter_classifies_live_payload_values(self):
        df = self._build_df()
        payload = {
            "sentiment_value": 79.0,
            "sentiment_source": "live_sentiment_api",
            "funding_rate": 0.001,
            "funding_source": "live_funding_api",
            "long_short_ratio": 1.15,
            "long_short_ratio_source": "live_lsr_api",
            "open_interest_ratio": 1.4,
            "oi_source": "live_oi_api",
            "news_veto": False,
            "news_source": "live_news_api",
        }

        result = build_runtime_signal_inputs(df, runtime_payload=payload)

        self.assertEqual(result.get("sentiment_quality"), "live")
        self.assertEqual(result.get("funding_quality"), "live")
        self.assertEqual(result.get("long_short_ratio_quality"), "live")
        self.assertEqual(result.get("oi_quality"), "live")
        self.assertEqual(result.get("news_quality"), "live")
        self.assertFalse(bool(result.get("sentiment_degraded")))

    def test_runtime_source_adapter_classifies_derived_values_as_fallback(self):
        df = self._build_df()
        df["sentiment_value"] = 81.0
        df["funding_rate"] = 0.0005
        df["long_short_ratio"] = 1.1
        df["open_interest_ratio"] = 1.3
        df["news_veto"] = False

        result = build_runtime_signal_inputs(df)

        self.assertEqual(result.get("sentiment_quality"), "fallback")
        self.assertEqual(result.get("funding_quality"), "fallback")
        self.assertEqual(result.get("long_short_ratio_quality"), "fallback")
        self.assertEqual(result.get("oi_quality"), "fallback")
        self.assertEqual(result.get("news_quality"), "fallback")
        self.assertTrue(str(result.get("sentiment_source", "")).startswith("fallback:ohlcv:"))
        self.assertTrue(bool(result.get("sentiment_degraded")))

    def test_runtime_source_adapter_classifies_missing_values_as_unavailable(self):
        df = self._build_df()

        result = build_runtime_signal_inputs(df)

        self.assertEqual(result.get("sentiment_quality"), "unavailable")
        self.assertEqual(result.get("funding_quality"), "unavailable")
        self.assertEqual(result.get("long_short_ratio_quality"), "unavailable")
        self.assertEqual(result.get("oi_quality"), "unavailable")
        self.assertEqual(result.get("news_quality"), "unavailable")
        self.assertEqual(result.get("sentiment_source"), "unavailable")
        self.assertIsNone(result.get("sentiment_value"))


    def test_strategy_audit_collector_tracks_layer_and_quality_distributions(self):
        collector = StrategyAuditCollector()

        collector.record(
            {
                "failed_layer": None,
                "layers": {
                    "regime_filter": {
                        "passed": True,
                        "details": {"source_flags": {"news_quality": "live"}, "degraded_mode": 0.0},
                    },
                    "layer1_pump_detection": {"passed": True, "details": {}},
                    "layer2_weakness_confirmation": {"passed": True, "details": {}},
                    "layer3_entry_location": {"passed": True, "details": {}},
                    "layer4_fake_filter": {
                        "passed": True,
                        "details": {
                            "source_flags": {
                                "sentiment_quality": "live",
                                "funding_quality": "live",
                            },
                            "degraded_mode": 0.0,
                        },
                    },
                    "layer5_tp_sl": {
                        "passed": True,
                        "details": {"fallback_rr_used": 1.0},
                    },
                },
            },
            signal_side="SHORT",
        )

        collector.record(
            {
                "failed_layer": "layer4_fake_filter",
                "layers": {
                    "regime_filter": {
                        "passed": True,
                        "details": {"source_flags": {"news_quality": "unavailable"}, "degraded_mode": 1.0},
                    },
                    "layer1_pump_detection": {"passed": True, "details": {}},
                    "layer2_weakness_confirmation": {"passed": True, "details": {}},
                    "layer3_entry_location": {"passed": True, "details": {}},
                    "layer4_fake_filter": {
                        "passed": False,
                        "details": {
                            "source_flags": {
                                "sentiment_quality": "unavailable",
                                "funding_quality": "fallback",
                            },
                            "degraded_mode": 1.0,
                            "blocker_price_above_vwap": 0.0,
                            "blocker_sentiment_euphoric": 1.0,
                            "blocker_funding_supports_short": 1.0,
                            "blocker_long_short_ratio_extreme": 0.0,
                            "blocker_oi_overheated": 0.0,
                            "fail_due_to_price_structure": 0.0,
                            "fail_due_to_sentiment": 1.0,
                            "fail_due_to_derivatives_context": 1.0,
                            "fail_due_to_degraded_mode_only": 0.0,
                            "hard_fail": 0.0,
                            "soft_fail": 1.0,
                            "degraded_data_fail": 0.0,
                            "soft_pass_candidate": 1.0,
                            "price_above_vwap": 1.0,
                            "sentiment_euphoric": 0.0,
                            "funding_supports_short": 0.0,
                            "long_short_ratio_extreme": 0.0,
                            "oi_overheated": 1.0,
                        },
                    },
                },
            },
            signal_side=None,
        )

        snap = collector.snapshot()

        self.assertEqual(snap.get("evaluated_count"), 2)
        self.assertEqual(snap.get("reached_layer5_count"), 1)
        self.assertEqual(snap.get("passed_layer4_count"), 1)
        self.assertEqual(snap.get("passed_layer5_count"), 1)
        self.assertEqual(snap.get("failed_layer_counts", {}).get("layer4_fake_filter"), 1)
        self.assertEqual(snap.get("layer4_fail_count"), 1)
        self.assertEqual(snap.get("layer4_sentiment_blocker_count"), 1)
        self.assertEqual(snap.get("layer4_funding_blocker_count"), 1)
        self.assertEqual(snap.get("layer4_lsr_blocker_count"), 0)
        self.assertEqual(snap.get("no_signal_count"), 1)
        self.assertEqual(snap.get("short_signal_count"), 1)
        self.assertEqual(snap.get("layer5_fallback_rr_used_count"), 1)
        self.assertEqual(snap.get("regime_filter_degraded_mode_count"), 1)
        self.assertEqual(snap.get("layer4_fake_filter_degraded_mode_count"), 1)
        self.assertEqual(snap.get("layer4_soft_pass_candidate_count"), 1)

        blockers = snap.get("layer4_blocker_counts", {})
        self.assertEqual(blockers.get("price_above_vwap"), 0)
        self.assertEqual(blockers.get("sentiment_euphoric"), 1)
        self.assertEqual(blockers.get("funding_supports_short"), 1)
        self.assertEqual(blockers.get("oi_overheated"), 0)

        fail_types = snap.get("layer4_fail_type_counts", {})
        self.assertEqual(fail_types.get("fail_due_to_price_structure"), 0)
        self.assertEqual(fail_types.get("fail_due_to_sentiment"), 1)
        self.assertEqual(fail_types.get("fail_due_to_derivatives_context"), 1)
        self.assertEqual(fail_types.get("fail_due_to_degraded_mode_only"), 0)
        self.assertEqual(fail_types.get("hard_fail"), 0)
        self.assertEqual(fail_types.get("soft_fail"), 1)

        quality = snap.get("source_quality_counts", {})
        self.assertEqual(quality.get("regime_filter", {}).get("news_quality", {}).get("live"), 1)
        self.assertEqual(quality.get("regime_filter", {}).get("news_quality", {}).get("unavailable"), 1)
        self.assertEqual(quality.get("layer4_fake_filter", {}).get("sentiment_quality", {}).get("live"), 1)
        self.assertEqual(quality.get("layer4_fake_filter", {}).get("sentiment_quality", {}).get("unavailable"), 1)

        compact = collector.compact_snapshot()
        self.assertEqual(compact.get("top_failed_layer"), "layer4_fake_filter")
        self.assertEqual(compact.get("top_failed_count"), 1)
        self.assertEqual(compact.get("layer4_fail_count"), 1)
        self.assertEqual(compact.get("layer4_sentiment_blocker_count"), 1)
        self.assertEqual(compact.get("layer4_funding_blocker_count"), 1)
        self.assertEqual(compact.get("layer4_lsr_blocker_count"), 0)
        self.assertEqual(compact.get("top_layer4_blocker_count"), 1)
        self.assertIn(compact.get("top_layer4_blocker"), {"sentiment_euphoric", "funding_supports_short"})

    def test_strategy_audit_collector_does_not_count_layer4_fail_types_when_layer4_not_reached(self):
        collector = StrategyAuditCollector()
        collector.record(
            {
                "failed_layer": "regime_filter",
                "layers": {
                    "regime_filter": {
                        "passed": False,
                        "details": {
                            "passed": 0.0,
                            "missing_conditions": "stretched_from_vwap",
                            "failed_reason": "missing:stretched_from_vwap",
                        },
                    }
                },
            },
            signal_side=None,
        )

        snap = collector.snapshot()
        self.assertEqual(int(snap.get("reached_layer4_count", 0)), 0)
        self.assertEqual(int(snap.get("layer4_fail_count", 0)), 0)
        fail_types = snap.get("layer4_fail_type_counts", {})
        self.assertEqual(int(fail_types.get("fail_due_to_price_structure", 0)), 0)
        self.assertEqual(int(fail_types.get("fail_due_to_sentiment", 0)), 0)
        self.assertEqual(int(fail_types.get("fail_due_to_derivatives_context", 0)), 0)
        self.assertEqual(int(fail_types.get("fail_due_to_degraded_mode_only", 0)), 0)
        self.assertEqual(int(fail_types.get("hard_fail", 0)), 0)
        self.assertEqual(int(fail_types.get("soft_fail", 0)), 0)
        self.assertEqual(int(fail_types.get("degraded_data_fail", 0)), 0)

    def test_strategy_audit_collector_counts_layer4_oi_and_price_blockers(self):
        collector = StrategyAuditCollector()

        collector.record(
            {
                "failed_layer": "layer4_fake_filter",
                "layers": {
                    "layer4_fake_filter": {
                        "passed": False,
                        "details": {
                            "passed": 0.0,
                            "blocker_price_above_vwap": 1.0,
                            "blocker_oi_overheated": 0.0,
                            "blocker_sentiment_euphoric": 0.0,
                            "blocker_funding_supports_short": 0.0,
                            "blocker_long_short_ratio_extreme": 0.0,
                            "missing_conditions": "price_above_vwap",
                        },
                    }
                },
            },
            signal_side=None,
        )
        collector.record(
            {
                "failed_layer": "layer4_fake_filter",
                "layers": {
                    "layer4_fake_filter": {
                        "passed": False,
                        "details": {
                            "passed": 0.0,
                            "blocker_price_above_vwap": 0.0,
                            "blocker_oi_overheated": 1.0,
                            "blocker_sentiment_euphoric": 0.0,
                            "blocker_funding_supports_short": 0.0,
                            "blocker_long_short_ratio_extreme": 0.0,
                            "missing_conditions": "oi_overheated",
                        },
                    }
                },
            },
            signal_side=None,
        )

        snap = collector.snapshot()
        self.assertEqual(int(snap.get("layer4_price_blocker_count", 0)), 1)
        self.assertEqual(int(snap.get("layer4_oi_blocker_count", 0)), 1)

    def test_strategy_audit_collector_counts_layer4_soft_pass_candidate(self):
        collector = StrategyAuditCollector()

        collector.record(
            {
                "failed_layer": "layer4_fake_filter",
                "layers": {
                    "layer4_fake_filter": {
                        "passed": False,
                        "details": {
                            "passed": 0.0,
                            "price_above_vwap": 1.0,
                            "sentiment_euphoric": 0.0,
                            "funding_supports_short": 1.0,
                            "long_short_ratio_extreme": 0.0,
                            "oi_overheated": 0.0,
                            "soft_pass_candidate": 1.0,
                            "missing_conditions": "sentiment_euphoric",
                        },
                    }
                },
            },
            signal_side=None,
        )

        snap = collector.snapshot()
        self.assertEqual(int(snap.get("layer4_soft_pass_candidate_count", 0)), 1)

    def test_strategy_audit_collector_distinguishes_layer5_vp_vs_rr_fallback(self):
        collector = StrategyAuditCollector()

        collector.record(
            {
                "failed_layer": None,
                "layers": {
                    "layer5_tp_sl": {
                        "passed": True,
                        "details": {
                            "passed": 1.0,
                            "fallback_rr_used": 1.0,
                            "tp_reference": "rr_fallback",
                            "atr_available": 1.0,
                            "volume_profile_available": 0.0,
                        },
                    }
                },
            },
            signal_side="SHORT",
        )
        collector.record(
            {
                "failed_layer": None,
                "layers": {
                    "layer5_tp_sl": {
                        "passed": True,
                        "details": {
                            "passed": 1.0,
                            "fallback_rr_used": 0.0,
                            "tp_reference": "vp_poc",
                            "atr_available": 1.0,
                            "volume_profile_available": 1.0,
                        },
                    }
                },
            },
            signal_side="SHORT",
        )
        collector.record(
            {
                "failed_layer": "layer5_tp_sl",
                "layers": {
                    "layer5_tp_sl": {
                        "passed": False,
                        "details": {
                            "passed": 0.0,
                            "fallback_rr_used": 0.0,
                            "atr_available": 0.0,
                            "volume_profile_available": 0.0,
                            "missing_conditions": "atr,volume_profile",
                        },
                    }
                },
            },
            signal_side=None,
        )

        snap = collector.snapshot()
        self.assertEqual(int(snap.get("layer5_fallback_rr_used_count", 0)), 1)
        self.assertEqual(int(snap.get("layer5_vp_based_count", 0)), 1)
        self.assertEqual(int(snap.get("layer5_fail_missing_atr_count", 0)), 1)
        self.assertEqual(int(snap.get("layer5_fail_missing_volume_profile_count", 0)), 1)

    def test_strategy_audit_compact_snapshot_exposes_observation_counters(self):
        collector = StrategyAuditCollector()
        collector.record(
            {
                "failed_layer": "layer4_fake_filter",
                "layers": {
                    "layer4_fake_filter": {
                        "passed": False,
                        "details": {
                            "passed": 0.0,
                            "blocker_price_above_vwap": 1.0,
                            "blocker_oi_overheated": 1.0,
                            "blocker_sentiment_euphoric": 0.0,
                            "blocker_funding_supports_short": 0.0,
                            "blocker_long_short_ratio_extreme": 0.0,
                            "missing_conditions": "price_above_vwap,oi_overheated",
                        },
                    }
                },
            },
            signal_side=None,
        )

        compact = collector.compact_snapshot()
        for key in (
            "evaluations_total",
            "regime_filter_fail_count",
            "top_regime_filter_blocker",
            "top_regime_filter_blocker_count",
            "layer1_rsi_high_blocker_count",
            "layer1_volume_spike_blocker_count",
            "top_layer1_blocker",
            "top_layer1_blocker_count",
            "layer4_oi_blocker_count",
            "layer4_price_blocker_count",
            "layer5_vp_based_count",
            "no_signal_ratio",
            "short_signal_ratio",
        ):
            self.assertIn(key, compact)

    def test_strategy_audit_compact_snapshot_keeps_top_blockers_empty_when_counts_are_zero(self):
        collector = StrategyAuditCollector()
        collector.record(
            {
                "failed_layer": None,
                "layers": {
                    "regime_filter": {
                        "passed": True,
                        "details": {
                            "passed": 1.0,
                            "htf_trend_ok": 1.0,
                            "stretched_from_vwap": 1.0,
                            "volatility_regime_ok": 1.0,
                            "news_veto": 1.0,
                            "missing_conditions": "",
                            "failed_reason": "none",
                        },
                    }
                },
            },
            signal_side="SHORT",
        )

        compact = collector.compact_snapshot()
        self.assertEqual(compact.get("top_regime_filter_blocker"), "")
        self.assertEqual(int(compact.get("top_regime_filter_blocker_count", 0)), 0)
        self.assertEqual(compact.get("top_layer1_blocker"), "")
        self.assertEqual(int(compact.get("top_layer1_blocker_count", 0)), 0)
        self.assertEqual(compact.get("top_layer4_blocker"), "")
        self.assertEqual(int(compact.get("top_layer4_blocker_count", 0)), 0)

    def test_runtime_strategy_audit_log_payload_exposes_observation_fields(self):
        df = self._build_df()
        strategy = LayeredPumpStrategy(SignalConfig())
        exchange = ExchangeSnapshot(
            symbol="BTC/USDT",
            account=AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0),
            positions=[],
            open_orders=[],
        )

        strategy.generate(
            StrategyContext(
                symbol="BTC/USDT",
                market_ohlcv=df,
                mark_price=float(df.iloc[-1]["close"]),
                exchange=exchange,
                synced_state=TradeState.FLAT,
                sentiment_index=78.0,
                sentiment_source="provided",
                funding_rate=0.001,
                long_short_ratio=1.2,
            )
        )

        df_fail = df.copy()
        df_fail.iloc[-1, df_fail.columns.get_loc("vwap")] = 120.0
        strategy.generate(
            StrategyContext(
                symbol="BTC/USDT",
                market_ohlcv=df_fail,
                mark_price=float(df_fail.iloc[-1]["close"]),
                exchange=exchange,
                synced_state=TradeState.FLAT,
                sentiment_index=55.0,
                sentiment_source="provided",
                funding_rate=-0.01,
                long_short_ratio=0.9,
                open_interest=100.0,
                open_interest_source="provided",
            )
        )

        payload = _strategy_audit_log_payload(strategy)
        self.assertIn("strategy_audit_compact", payload)
        self.assertIn("strategy_audit_regime_filter", payload)
        self.assertIn("strategy_audit_regime_diagnostics", payload)
        self.assertIn("strategy_audit_layer1", payload)
        self.assertIn("strategy_audit_layer1_diagnostics", payload)
        self.assertIn("strategy_audit_layer2", payload)
        self.assertIn("strategy_audit_layer2_diagnostics", payload)
        self.assertIn("strategy_audit_layer4", payload)
        self.assertIn("strategy_audit_source_quality", payload)

        compact = payload.get("strategy_audit_compact", {})
        self.assertIn("evaluations_total", compact)
        self.assertIn("regime_filter_fail_count", compact)
        self.assertIn("top_regime_filter_blocker", compact)
        self.assertIn("layer1_rsi_high_blocker_count", compact)
        self.assertIn("top_layer1_blocker", compact)
        self.assertIn("layer4_price_blocker_count", compact)
        self.assertIn("layer4_oi_blocker_count", compact)

        regime_filter = payload.get("strategy_audit_regime_filter", {})
        self.assertIn("regime_filter_htf_trend_blocker_count", regime_filter)
        self.assertIn("regime_filter_vwap_stretch_blocker_count", regime_filter)
        self.assertIn("regime_filter_volatility_blocker_count", regime_filter)
        self.assertIn("regime_filter_news_blocker_count", regime_filter)
        self.assertIn("top_regime_filter_blocker", regime_filter)
        self.assertIn("top_regime_filter_blocker_count", regime_filter)
        for key in (
            "htf_trend_metric_used",
            "htf_trend_threshold_used",
            "htf_trend_direction_context",
            "vwap_distance_metric_used",
            "vwap_stretch_threshold_used",
            "atr_norm",
            "volatility_threshold_used",
            "failed_reason",
            "missing_conditions",
            "degraded_mode",
            "fail_due_to_degraded_mode_only",
            "soft_pass_candidate",
            "soft_pass_used",
            "soft_pass_reason",
            "source_flags",
            "regime_filter_subconditions_state",
        ):
            self.assertIn(key, regime_filter)
        self.assertIsInstance(regime_filter.get("source_flags", {}), dict)
        self.assertIsInstance(regime_filter.get("regime_filter_subconditions_state", {}), dict)

        layer1 = payload.get("strategy_audit_layer1", {})
        self.assertIn("layer1_pass_count", layer1)
        self.assertIn("layer1_fail_count", layer1)
        self.assertIn("layer1_rsi_high_blocker_count", layer1)
        self.assertIn("layer1_volume_spike_blocker_count", layer1)
        self.assertIn("layer1_above_bollinger_upper_blocker_count", layer1)
        self.assertIn("layer1_above_keltner_upper_blocker_count", layer1)
        self.assertIn("top_layer1_blocker", layer1)
        self.assertIn("top_layer1_blocker_count", layer1)
        self.assertIn("failed_reason", layer1)
        self.assertIn("missing_conditions", layer1)
        self.assertIn("soft_pass_candidate", layer1)
        self.assertIn("pump_bar_offset", layer1)
        self.assertIn("layer1_subconditions_state", layer1)

        layer2 = payload.get("strategy_audit_layer2", {})
        self.assertIn("reached_layer2_count", layer2)
        self.assertIn("passed_layer2_count", layer2)
        self.assertIn("layer2_fail_count", layer2)
        self.assertIn("failed_reason", layer2)
        self.assertIn("missing_conditions", layer2)
        self.assertIn("layer2_subconditions_state", layer2)

        layer4 = payload.get("strategy_audit_layer4", {})
        self.assertIn("layer4_sentiment_blocker_count", layer4)
        self.assertIn("layer4_funding_blocker_count", layer4)
        self.assertIn("layer4_lsr_blocker_count", layer4)
        self.assertIn("layer4_oi_blocker_count", layer4)
        self.assertIn("layer4_price_blocker_count", layer4)

        source_quality = payload.get("strategy_audit_source_quality", {})
        self.assertIn("regime_filter", source_quality)
        self.assertIn("layer4_fake_filter", source_quality)

    def test_runtime_payload_exposes_all_required_raw_regime_diagnostics_fields(self):
        df = self._build_df()
        strategy = LayeredPumpStrategy(SignalConfig())
        exchange = ExchangeSnapshot(
            symbol="BTC/USDT",
            account=AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0),
            positions=[],
            open_orders=[],
        )

        strategy.generate(
            StrategyContext(
                symbol="BTC/USDT",
                market_ohlcv=df,
                mark_price=float(df.iloc[-1]["close"]),
                exchange=exchange,
                synced_state=TradeState.FLAT,
                sentiment_index=78.0,
                sentiment_source="provided",
                funding_rate=0.001,
                long_short_ratio=1.2,
            )
        )
        payload = _strategy_audit_log_payload(strategy)
        regime_diag = payload.get("strategy_audit_regime_diagnostics", {})
        for key in (
            "htf_trend_metric_used",
            "htf_trend_threshold_used",
            "htf_trend_direction_context",
            "vwap_distance_metric_used",
            "vwap_stretch_threshold_used",
            "atr_norm",
            "volatility_threshold_used",
            "failed_reason",
            "missing_conditions",
            "degraded_mode",
            "fail_due_to_degraded_mode_only",
            "soft_pass_candidate",
            "soft_pass_used",
            "soft_pass_reason",
            "source_flags",
            "regime_filter_subconditions_state",
        ):
            self.assertIn(key, regime_diag)
        self.assertIsInstance(regime_diag.get("source_flags", {}), dict)
        self.assertIsInstance(regime_diag.get("regime_filter_subconditions_state", {}), dict)

    def test_runtime_payload_exposes_all_required_raw_layer1_diagnostics_fields(self):
        df = self._build_df()
        df.iloc[-1, df.columns.get_loc("bb_upper")] = 120.0
        df.iloc[-1, df.columns.get_loc("kc_upper")] = 120.0

        strategy = LayeredPumpStrategy(SignalConfig())
        exchange = ExchangeSnapshot(
            symbol="BTC/USDT",
            account=AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0),
            positions=[],
            open_orders=[],
        )

        strategy.generate(
            StrategyContext(
                symbol="BTC/USDT",
                market_ohlcv=df,
                mark_price=float(df.iloc[-1]["close"]),
                exchange=exchange,
                synced_state=TradeState.FLAT,
                sentiment_index=78.0,
                sentiment_source="provided",
                funding_rate=0.001,
                long_short_ratio=1.2,
            )
        )
        payload = _strategy_audit_log_payload(strategy)
        layer1_diag = payload.get("strategy_audit_layer1_diagnostics", {})
        for key in (
            "rsi",
            "rsi_high_threshold_used",
            "volume_spike",
            "volume_spike_high",
            "volume_spike_threshold_used",
            "close_metric_used",
            "bollinger_upper_metric_used",
            "keltner_upper_metric_used",
            "above_bollinger_upper",
            "above_keltner_upper",
            "upper_band_breakout",
            "pump_context_strength",
            "failed_reason",
            "missing_conditions",
            "soft_pass_candidate",
            "soft_pass_used",
            "soft_pass_reason",
            "pump_bar_offset",
            "layer1_subconditions_state",
        ):
            self.assertIn(key, layer1_diag)
        self.assertIsInstance(layer1_diag.get("layer1_subconditions_state", {}), dict)

    def test_runtime_payload_exposes_all_required_raw_layer2_diagnostics_fields(self):
        df = self._build_df()
        df.iloc[-1, df.columns.get_loc("close")] = 99.0
        df.iloc[-1, df.columns.get_loc("obv")] = 1400.0
        df.iloc[-1, df.columns.get_loc("cvd")] = 1300.0

        strategy = LayeredPumpStrategy(SignalConfig())
        exchange = ExchangeSnapshot(
            symbol="BTC/USDT",
            account=AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0),
            positions=[],
            open_orders=[],
        )

        strategy.generate(
            StrategyContext(
                symbol="BTC/USDT",
                market_ohlcv=df,
                mark_price=float(df.iloc[-1]["close"]),
                exchange=exchange,
                synced_state=TradeState.FLAT,
                sentiment_index=78.0,
                sentiment_source="provided",
                funding_rate=0.001,
                long_short_ratio=1.2,
            )
        )
        payload = _strategy_audit_log_payload(strategy)
        layer2_diag = payload.get("strategy_audit_layer2_diagnostics", {})
        for key in (
            "price_up_or_near_high",
            "price_up",
            "near_high_context",
            "obv_bearish_divergence",
            "cvd_bearish_divergence",
            "close_last_used",
            "close_ref_used",
            "obv_last_used",
            "obv_ref_used",
            "cvd_last_used",
            "cvd_ref_used",
            "weakness_lookback_used",
            "weakness_strength",
            "failed_reason",
            "missing_conditions",
            "layer2_subconditions_state",
        ):
            self.assertIn(key, layer2_diag)
        self.assertIsInstance(layer2_diag.get("layer2_subconditions_state", {}), dict)

    def test_runtime_payload_handles_missing_regime_raw_values_safely(self):
        class LegacyLikeStrategy:
            @staticmethod
            def audit_observation_snapshot():
                return {
                    "strategy_audit_compact": {"evaluations_total": 1},
                    "strategy_audit_regime_filter": {"regime_filter_pass_count": 0},
                }

            @staticmethod
            def audit_snapshot():
                return {}

        payload = _strategy_audit_log_payload(LegacyLikeStrategy())
        regime_diag = payload.get("strategy_audit_regime_diagnostics", {})
        self.assertEqual(regime_diag.get("htf_trend_metric_used"), None)
        self.assertEqual(regime_diag.get("vwap_distance_metric_used"), None)
        self.assertEqual(regime_diag.get("atr_norm"), None)
        self.assertEqual(regime_diag.get("htf_trend_direction_context"), "")
        self.assertEqual(regime_diag.get("failed_reason"), "")
        self.assertEqual(regime_diag.get("missing_conditions"), "")
        self.assertEqual(regime_diag.get("degraded_mode"), 0.0)
        self.assertEqual(regime_diag.get("fail_due_to_degraded_mode_only"), 0.0)
        self.assertEqual(regime_diag.get("soft_pass_candidate"), 0.0)
        self.assertEqual(regime_diag.get("soft_pass_used"), 0.0)
        self.assertEqual(regime_diag.get("soft_pass_reason"), "")
        self.assertEqual(regime_diag.get("source_flags"), {})
        self.assertEqual(regime_diag.get("regime_filter_subconditions_state"), {})

        regime_filter = payload.get("strategy_audit_regime_filter", {})
        self.assertIn("regime_filter_pass_count", regime_filter)
        self.assertIn("htf_trend_metric_used", regime_filter)
        self.assertIn("vwap_distance_metric_used", regime_filter)
        self.assertIn("atr_norm", regime_filter)
        self.assertEqual(regime_filter.get("failed_reason"), "")
        self.assertEqual(regime_filter.get("missing_conditions"), "")
        self.assertEqual(regime_filter.get("degraded_mode"), 0.0)
        self.assertEqual(regime_filter.get("fail_due_to_degraded_mode_only"), 0.0)
        self.assertEqual(regime_filter.get("soft_pass_candidate"), 0.0)
        self.assertEqual(regime_filter.get("soft_pass_used"), 0.0)
        self.assertEqual(regime_filter.get("soft_pass_reason"), "")
        self.assertEqual(regime_filter.get("source_flags"), {})
        self.assertEqual(regime_filter.get("regime_filter_subconditions_state"), {})

    def test_runtime_payload_handles_missing_layer1_raw_values_safely(self):
        class LegacyLikeStrategy:
            @staticmethod
            def audit_observation_snapshot():
                return {
                    "strategy_audit_compact": {"evaluations_total": 1},
                    "strategy_audit_layer1": {"layer1_pass_count": 0},
                }

            @staticmethod
            def audit_snapshot():
                return {}

        payload = _strategy_audit_log_payload(LegacyLikeStrategy())
        layer1_diag = payload.get("strategy_audit_layer1_diagnostics", {})
        self.assertEqual(layer1_diag.get("rsi"), None)
        self.assertEqual(layer1_diag.get("volume_spike"), None)
        self.assertEqual(layer1_diag.get("pump_context_strength"), 0.0)
        self.assertEqual(layer1_diag.get("failed_reason"), "")
        self.assertEqual(layer1_diag.get("missing_conditions"), "")
        self.assertEqual(layer1_diag.get("soft_pass_candidate"), 0.0)
        self.assertEqual(layer1_diag.get("soft_pass_used"), 0.0)
        self.assertEqual(layer1_diag.get("soft_pass_reason"), "")
        self.assertEqual(layer1_diag.get("pump_bar_offset"), None)
        self.assertEqual(layer1_diag.get("layer1_subconditions_state"), {})

        layer1 = payload.get("strategy_audit_layer1", {})
        self.assertIn("layer1_pass_count", layer1)
        self.assertIn("rsi", layer1)
        self.assertIn("volume_spike", layer1)
        self.assertEqual(layer1.get("failed_reason"), "")
        self.assertEqual(layer1.get("missing_conditions"), "")
        self.assertEqual(layer1.get("soft_pass_candidate"), 0.0)
        self.assertEqual(layer1.get("soft_pass_used"), 0.0)
        self.assertEqual(layer1.get("soft_pass_reason"), "")
        self.assertEqual(layer1.get("pump_bar_offset"), None)
        self.assertEqual(layer1.get("layer1_subconditions_state"), {})

    def test_runtime_payload_handles_missing_layer2_raw_values_safely(self):
        class LegacyLikeStrategy:
            @staticmethod
            def audit_observation_snapshot():
                return {
                    "strategy_audit_compact": {"evaluations_total": 1},
                    "strategy_audit_layer2": {"layer2_fail_count": 0},
                }

            @staticmethod
            def audit_snapshot():
                return {}

        payload = _strategy_audit_log_payload(LegacyLikeStrategy())
        layer2_diag = payload.get("strategy_audit_layer2_diagnostics", {})
        self.assertEqual(layer2_diag.get("close_last_used"), None)
        self.assertEqual(layer2_diag.get("obv_last_used"), None)
        self.assertEqual(layer2_diag.get("cvd_last_used"), None)
        self.assertEqual(layer2_diag.get("weakness_strength"), 0.0)
        self.assertEqual(layer2_diag.get("failed_reason"), "")
        self.assertEqual(layer2_diag.get("missing_conditions"), "")
        self.assertEqual(layer2_diag.get("layer2_subconditions_state"), {})

        layer2 = payload.get("strategy_audit_layer2", {})
        self.assertIn("reached_layer2_count", layer2)
        self.assertIn("passed_layer2_count", layer2)
        self.assertIn("layer2_fail_count", layer2)
        self.assertEqual(layer2.get("failed_reason"), "")
        self.assertEqual(layer2.get("missing_conditions"), "")
        self.assertEqual(layer2.get("layer2_subconditions_state"), {})

    def test_runtime_payload_preserves_per_sample_regime_semantics(self):
        df = self._build_df()
        df.iloc[-1, df.columns.get_loc("close")] = 100.0
        df["vwap"] = 104.0
        df["atr"] = 0.02
        df["adx"] = 10.0
        df["ema20"] = 95.0
        df["ema50"] = 90.0

        strategy = LayeredPumpStrategy(SignalConfig())
        exchange = ExchangeSnapshot(
            symbol="BTC/USDT",
            account=AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0),
            positions=[],
            open_orders=[],
        )

        strategy.generate(
            StrategyContext(
                symbol="BTC/USDT",
                market_ohlcv=df,
                mark_price=float(df.iloc[-1]["close"]),
                exchange=exchange,
                synced_state=TradeState.FLAT,
                sentiment_index=55.0,
                sentiment_source="provided",
                funding_rate=-0.01,
                long_short_ratio=0.9,
                open_interest=100.0,
                open_interest_source="provided",
            )
        )

        payload = _strategy_audit_log_payload(strategy)
        regime_filter = payload.get("strategy_audit_regime_filter", {})
        regime_diag = payload.get("strategy_audit_regime_diagnostics", {})

        self.assertTrue(str(regime_filter.get("failed_reason", "")).startswith("missing:"))
        self.assertIn("stretched_from_vwap", str(regime_filter.get("missing_conditions", "")))
        self.assertIn("volatility_regime_ok", str(regime_filter.get("missing_conditions", "")))
        self.assertEqual(regime_filter.get("source_flags", {}).get("vwap_quality"), "live")
        self.assertIn("news_quality", regime_filter.get("source_flags", {}))
        self.assertEqual(
            regime_filter.get("regime_filter_subconditions_state", {}),
            {
                "htf_trend_ok": True,
                "stretched_from_vwap": False,
                "volatility_regime_ok": False,
                "news_veto": True,
            },
        )

        self.assertEqual(regime_diag.get("failed_reason"), regime_filter.get("failed_reason"))
        self.assertEqual(regime_diag.get("missing_conditions"), regime_filter.get("missing_conditions"))
        self.assertEqual(regime_diag.get("degraded_mode"), regime_filter.get("degraded_mode"))
        self.assertEqual(
            regime_diag.get("fail_due_to_degraded_mode_only"),
            regime_filter.get("fail_due_to_degraded_mode_only"),
        )
        self.assertEqual(regime_diag.get("soft_pass_candidate"), regime_filter.get("soft_pass_candidate"))
        self.assertEqual(regime_diag.get("soft_pass_used"), regime_filter.get("soft_pass_used"))
        self.assertEqual(regime_diag.get("soft_pass_reason"), regime_filter.get("soft_pass_reason"))
        self.assertEqual(regime_diag.get("source_flags"), regime_filter.get("source_flags"))
        self.assertEqual(
            regime_diag.get("regime_filter_subconditions_state"),
            regime_filter.get("regime_filter_subconditions_state"),
        )

    def test_runtime_payload_preserves_per_sample_layer1_semantics(self):
        df = self._build_df()
        df.iloc[-1, df.columns.get_loc("bb_upper")] = 120.0
        df.iloc[-1, df.columns.get_loc("kc_upper")] = 120.0

        strategy = LayeredPumpStrategy(SignalConfig())
        exchange = ExchangeSnapshot(
            symbol="BTC/USDT",
            account=AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0),
            positions=[],
            open_orders=[],
        )

        strategy.generate(
            StrategyContext(
                symbol="BTC/USDT",
                market_ohlcv=df,
                mark_price=float(df.iloc[-1]["close"]),
                exchange=exchange,
                synced_state=TradeState.FLAT,
                sentiment_index=78.0,
                sentiment_source="provided",
                funding_rate=0.001,
                long_short_ratio=1.2,
            )
        )

        payload = _strategy_audit_log_payload(strategy)
        layer1 = payload.get("strategy_audit_layer1", {})
        layer1_diag = payload.get("strategy_audit_layer1_diagnostics", {})

        self.assertEqual(layer1.get("failed_reason"), "missing:upper_band_breakout")
        self.assertEqual(layer1.get("missing_conditions"), "upper_band_breakout")
        self.assertEqual(layer1.get("soft_pass_candidate"), 1.0)
        self.assertEqual(
            layer1.get("layer1_subconditions_state", {}),
            {
                "rsi_high": True,
                "volume_spike_high": True,
                "upper_band_breakout": False,
                "above_bollinger_upper": False,
                "above_keltner_upper": False,
            },
        )
        self.assertEqual(layer1_diag.get("failed_reason"), layer1.get("failed_reason"))
        self.assertEqual(layer1_diag.get("missing_conditions"), layer1.get("missing_conditions"))
        self.assertEqual(layer1_diag.get("soft_pass_candidate"), layer1.get("soft_pass_candidate"))
        self.assertEqual(layer1_diag.get("pump_bar_offset"), layer1.get("pump_bar_offset"))
        self.assertEqual(
            layer1_diag.get("layer1_subconditions_state"),
            layer1.get("layer1_subconditions_state"),
        )

    def test_observability_payload_step_does_not_change_strategy_decisions(self):
        df = self._build_df()
        strategy = LayeredPumpStrategy(SignalConfig())
        exchange = ExchangeSnapshot(
            symbol="BTC/USDT",
            account=AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0),
            positions=[],
            open_orders=[],
        )
        context = StrategyContext(
            symbol="BTC/USDT",
            market_ohlcv=df,
            mark_price=float(df.iloc[-1]["close"]),
            exchange=exchange,
            synced_state=TradeState.FLAT,
            sentiment_index=78.0,
            sentiment_source="provided",
            funding_rate=0.001,
            long_short_ratio=1.2,
        )

        before = strategy.generate(context)
        _ = _strategy_audit_log_payload(strategy)
        after = strategy.generate(context)

        self.assertEqual(before.action, after.action)
        self.assertEqual(before.reason, after.reason)
        self.assertEqual(
            before.metadata.get("layer_failed"),
            after.metadata.get("layer_failed"),
        )

    def test_layer1_audit_no_reach_keeps_blockers_zero(self):
        collector = StrategyAuditCollector()

        collector.record(
            {
                "failed_layer": "regime_filter",
                "layers": {
                    "regime_filter": {
                        "passed": False,
                        "details": {
                            "passed": 0.0,
                            "missing_conditions": "stretched_from_vwap,volatility_regime_ok",
                            "failed_reason": "missing:stretched_from_vwap,volatility_regime_ok",
                        },
                    }
                },
            },
            signal_side=None,
        )

        snap = collector.snapshot()
        self.assertEqual(int(snap.get("reached_layer1_count", 0)), 0)
        self.assertEqual(int(snap.get("layer1_pass_count", 0)), 0)
        self.assertEqual(int(snap.get("layer1_fail_count", 0)), 0)
        self.assertEqual(int(snap.get("layer1_rsi_high_blocker_count", 0)), 0)
        self.assertEqual(int(snap.get("layer1_volume_spike_blocker_count", 0)), 0)
        self.assertEqual(int(snap.get("layer1_above_bollinger_upper_blocker_count", 0)), 0)
        self.assertEqual(int(snap.get("layer1_above_keltner_upper_blocker_count", 0)), 0)
        self.assertEqual(int(snap.get("layer1_soft_pass_candidate_count", 0)), 0)
        self.assertEqual(int(snap.get("layer1_soft_pass_used_count", 0)), 0)

    def test_layer1_audit_reached_fail_counts_only_valid_blockers(self):
        collector = StrategyAuditCollector()

        collector.record(
            {
                "failed_layer": "layer1_pump_detection",
                "layers": {
                    "layer1_pump_detection": {
                        "passed": False,
                        "details": {
                            "passed": 0.0,
                            "rsi_high": 0.0,
                            "volume_spike": 8.0,
                            "volume_spike_high": 1.0,
                            "above_bollinger_upper": 1.0,
                            "above_keltner_upper": 1.0,
                            "upper_band_breakout": 1.0,
                            "missing_conditions": "rsi_high",
                            "failed_reason": "missing:rsi_high",
                        },
                    }
                },
            },
            signal_side=None,
        )

        snap = collector.snapshot()
        self.assertEqual(int(snap.get("reached_layer1_count", 0)), 1)
        self.assertEqual(int(snap.get("layer1_pass_count", 0)), 0)
        self.assertEqual(int(snap.get("layer1_fail_count", 0)), 1)
        self.assertEqual(int(snap.get("layer1_rsi_high_blocker_count", 0)), 1)
        self.assertEqual(int(snap.get("layer1_volume_spike_blocker_count", 0)), 0)
        self.assertEqual(int(snap.get("layer1_above_bollinger_upper_blocker_count", 0)), 0)
        self.assertEqual(int(snap.get("layer1_above_keltner_upper_blocker_count", 0)), 0)

    def test_layer1_audit_reached_pass_increments_pass_without_blockers(self):
        collector = StrategyAuditCollector()

        collector.record(
            {
                "failed_layer": None,
                "layers": {
                    "layer1_pump_detection": {
                        "passed": True,
                        "details": {
                            "passed": 1.0,
                            "rsi_high": 1.0,
                            "volume_spike": 2.0,
                            "volume_spike_high": 1.0,
                            "above_bollinger_upper": 1.0,
                            "above_keltner_upper": 0.0,
                            "upper_band_breakout": 1.0,
                            "missing_conditions": "",
                            "failed_reason": "none",
                        },
                    }
                },
            },
            signal_side="SHORT",
        )

        snap = collector.snapshot()
        self.assertEqual(int(snap.get("reached_layer1_count", 0)), 1)
        self.assertEqual(int(snap.get("layer1_pass_count", 0)), 1)
        self.assertEqual(int(snap.get("layer1_fail_count", 0)), 0)
        self.assertEqual(int(snap.get("layer1_rsi_high_blocker_count", 0)), 0)
        self.assertEqual(int(snap.get("layer1_volume_spike_blocker_count", 0)), 0)
        self.assertEqual(int(snap.get("layer1_above_bollinger_upper_blocker_count", 0)), 0)
        self.assertEqual(int(snap.get("layer1_above_keltner_upper_blocker_count", 0)), 0)

    def test_runtime_strategy_audit_layer1_payload_zero_when_layer1_not_reached(self):
        df = self._build_df()
        df["atr"] = 0.01
        df["vwap"] = df["close"]

        strategy = LayeredPumpStrategy(SignalConfig())
        exchange = ExchangeSnapshot(
            symbol="BTC/USDT",
            account=AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0),
            positions=[],
            open_orders=[],
        )

        strategy.generate(
            StrategyContext(
                symbol="BTC/USDT",
                market_ohlcv=df,
                mark_price=float(df.iloc[-1]["close"]),
                exchange=exchange,
                synced_state=TradeState.FLAT,
                sentiment_index=55.0,
                sentiment_source="provided",
                funding_rate=-0.01,
                long_short_ratio=0.9,
            )
        )

        payload = _strategy_audit_log_payload(strategy)
        layer1 = payload.get("strategy_audit_layer1", {})
        self.assertEqual(int(layer1.get("layer1_pass_count", 0)), 0)
        self.assertEqual(int(layer1.get("layer1_fail_count", 0)), 0)
        self.assertEqual(int(layer1.get("layer1_rsi_high_blocker_count", 0)), 0)
        self.assertEqual(int(layer1.get("layer1_volume_spike_blocker_count", 0)), 0)
        self.assertEqual(int(layer1.get("layer1_above_bollinger_upper_blocker_count", 0)), 0)
        self.assertEqual(int(layer1.get("layer1_above_keltner_upper_blocker_count", 0)), 0)

    def test_strategy_audit_collector_layer1_aggregation_correctness(self):
        collector = StrategyAuditCollector()

        collector.record(
            {
                "failed_layer": None,
                "layers": {
                    "layer1_pump_detection": {
                        "passed": True,
                        "details": {
                            "passed": 1.0,
                            "rsi_high": 1.0,
                            "volume_spike": 8.0,
                            "volume_spike_high": 1.0,
                            "above_bollinger_upper": 1.0,
                            "above_keltner_upper": 0.0,
                            "upper_band_breakout": 1.0,
                            "missing_conditions": "",
                            "failed_reason": "none",
                        },
                    }
                },
            },
            signal_side="SHORT",
        )

        def _layer1_fail(details: dict) -> None:
            collector.record(
                {
                    "failed_layer": "layer1_pump_detection",
                    "layers": {
                        "layer1_pump_detection": {
                            "passed": False,
                            "details": details,
                        }
                    },
                },
                signal_side=None,
            )

        _layer1_fail(
            {
                "passed": 0.0,
                "rsi_high": 0.0,
                "volume_spike": 8.0,
                "volume_spike_high": 1.0,
                "above_bollinger_upper": 1.0,
                "above_keltner_upper": 0.0,
                "upper_band_breakout": 1.0,
                "missing_conditions": "rsi_high",
                "failed_reason": "missing:rsi_high",
            }
        )
        _layer1_fail(
            {
                "passed": 0.0,
                "rsi_high": 1.0,
                "volume_spike": 1.1,
                "volume_spike_high": 0.0,
                "above_bollinger_upper": 1.0,
                "above_keltner_upper": 0.0,
                "upper_band_breakout": 1.0,
                "missing_conditions": "volume_spike",
                "failed_reason": "missing:volume_spike",
            }
        )
        _layer1_fail(
            {
                "passed": 0.0,
                "rsi_high": 1.0,
                "volume_spike": 8.0,
                "volume_spike_high": 1.0,
                "above_bollinger_upper": 0.0,
                "above_keltner_upper": 0.0,
                "upper_band_breakout": 0.0,
                "missing_conditions": "upper_band_breakout",
                "failed_reason": "missing:upper_band_breakout",
            }
        )
        _layer1_fail(
            {
                "passed": 0.0,
                "rsi_high": 0.0,
                "volume_spike": 1.1,
                "volume_spike_high": 0.0,
                "above_bollinger_upper": 1.0,
                "above_keltner_upper": 0.0,
                "upper_band_breakout": 1.0,
                "missing_conditions": "rsi_high,volume_spike",
                "failed_reason": "missing:rsi_high,volume_spike",
                "soft_pass_candidate": 1.0,
            }
        )

        snap = collector.snapshot()
        compact = collector.compact_snapshot()

        self.assertEqual(int(snap.get("layer1_pass_count", 0)), 1)
        self.assertEqual(int(snap.get("layer1_fail_count", 0)), 4)
        self.assertEqual(int(snap.get("layer1_rsi_high_blocker_count", 0)), 2)
        self.assertEqual(int(snap.get("layer1_volume_spike_blocker_count", 0)), 2)
        self.assertEqual(int(snap.get("layer1_above_bollinger_upper_blocker_count", 0)), 1)
        self.assertEqual(int(snap.get("layer1_above_keltner_upper_blocker_count", 0)), 1)
        self.assertEqual(int(snap.get("layer1_soft_pass_candidate_count", 0)), 1)
        self.assertEqual(int(snap.get("layer1_soft_pass_used_count", 0)), 0)

        self.assertEqual(compact.get("top_layer1_blocker"), "rsi_high")
        self.assertEqual(int(compact.get("top_layer1_blocker_count", 0)), 2)

    def test_strategy_audit_collector_regime_filter_aggregation_correctness(self):
        collector = StrategyAuditCollector()

        collector.record(
            {
                "failed_layer": None,
                "layers": {
                    "regime_filter": {
                        "passed": True,
                        "details": {
                            "passed": 1.0,
                            "htf_trend_ok": 1.0,
                            "stretched_from_vwap": 1.0,
                            "volatility_regime_ok": 1.0,
                            "news_veto": 1.0,
                            "degraded_mode": 0.0,
                            "missing_conditions": "",
                            "failed_reason": "none",
                        },
                    }
                },
            },
            signal_side="SHORT",
        )

        def _regime_fail(details: dict) -> None:
            collector.record(
                {
                    "failed_layer": "regime_filter",
                    "layers": {
                        "regime_filter": {
                            "passed": False,
                            "details": details,
                        }
                    },
                },
                signal_side=None,
            )

        _regime_fail(
            {
                "passed": 0.0,
                "htf_trend_ok": 0.0,
                "stretched_from_vwap": 1.0,
                "volatility_regime_ok": 1.0,
                "news_veto": 1.0,
                "degraded_mode": 0.0,
                "missing_conditions": "htf_trend_ok",
                "failed_reason": "missing:htf_trend_ok",
                "soft_pass_candidate": 0.0,
            }
        )
        _regime_fail(
            {
                "passed": 0.0,
                "htf_trend_ok": 0.0,
                "stretched_from_vwap": 1.0,
                "volatility_regime_ok": 1.0,
                "news_veto": 1.0,
                "degraded_mode": 0.0,
                "missing_conditions": "htf_trend_ok",
                "failed_reason": "missing:htf_trend_ok",
                "soft_pass_candidate": 0.0,
            }
        )
        _regime_fail(
            {
                "passed": 0.0,
                "htf_trend_ok": 1.0,
                "stretched_from_vwap": 0.0,
                "volatility_regime_ok": 1.0,
                "news_veto": 1.0,
                "degraded_mode": 0.0,
                "missing_conditions": "stretched_from_vwap",
                "failed_reason": "missing:stretched_from_vwap",
                "soft_pass_candidate": 0.0,
            }
        )
        _regime_fail(
            {
                "passed": 0.0,
                "htf_trend_ok": 1.0,
                "stretched_from_vwap": 1.0,
                "volatility_regime_ok": 0.0,
                "news_veto": 1.0,
                "degraded_mode": 0.0,
                "missing_conditions": "volatility_regime_ok",
                "failed_reason": "missing:volatility_regime_ok",
                "soft_pass_candidate": 0.0,
            }
        )
        _regime_fail(
            {
                "passed": 0.0,
                "htf_trend_ok": 1.0,
                "stretched_from_vwap": 1.0,
                "volatility_regime_ok": 1.0,
                "news_veto": 0.0,
                "degraded_mode": 0.0,
                "missing_conditions": "news_veto",
                "failed_reason": "missing:news_veto",
                "soft_pass_candidate": 0.0,
            }
        )
        _regime_fail(
            {
                "passed": 0.0,
                "htf_trend_ok": 1.0,
                "stretched_from_vwap": 1.0,
                "volatility_regime_ok": 1.0,
                "news_veto": 1.0,
                "degraded_mode": 1.0,
                "missing_conditions": "",
                "failed_reason": "missing:",
                "fail_due_to_degraded_mode_only": 1.0,
                "soft_pass_candidate": 1.0,
            }
        )

        snap = collector.snapshot()
        compact = collector.compact_snapshot()

        self.assertEqual(int(snap.get("regime_filter_pass_count", 0)), 1)
        self.assertEqual(int(snap.get("regime_filter_fail_count", 0)), 6)
        self.assertEqual(int(snap.get("regime_filter_htf_trend_blocker_count", 0)), 2)
        self.assertEqual(int(snap.get("regime_filter_vwap_stretch_blocker_count", 0)), 1)
        self.assertEqual(int(snap.get("regime_filter_volatility_blocker_count", 0)), 1)
        self.assertEqual(int(snap.get("regime_filter_news_blocker_count", 0)), 1)
        self.assertEqual(int(snap.get("regime_filter_degraded_only_count", 0)), 1)
        self.assertEqual(int(snap.get("regime_filter_soft_pass_candidate_count", 0)), 1)

        self.assertEqual(compact.get("top_regime_filter_blocker"), "htf_trend_ok")
        self.assertEqual(int(compact.get("top_regime_filter_blocker_count", 0)), 2)
    def test_audit_comparison_softened_sentiment_reduces_sentiment_blockers(self):
        df = self._build_df()

        strict_gen = SignalGenerator(SignalConfig(fake_filter_sentiment_euphoric_soft=68.0))
        soft_gen = SignalGenerator(SignalConfig(fake_filter_sentiment_euphoric_soft=66.0))
        strict_audit = StrategyAuditCollector()
        soft_audit = StrategyAuditCollector()

        strict_passed, strict_layer4 = strict_gen._layer4_fake_filter(
            df=df,
            side="SHORT",
            sentiment_index=66.0,
            sentiment_source="live_api",
            funding_rate=0.001,
            funding_source="live_api",
            long_short_ratio=1.2,
            long_short_ratio_source="live_api",
            open_interest_ratio=1.5,
            oi_source="live_api",
            open_interest_source="live_api",
        )
        soft_passed, soft_layer4 = soft_gen._layer4_fake_filter(
            df=df,
            side="SHORT",
            sentiment_index=66.0,
            sentiment_source="live_api",
            funding_rate=0.001,
            funding_source="live_api",
            long_short_ratio=1.2,
            long_short_ratio_source="live_api",
            open_interest_ratio=1.5,
            oi_source="live_api",
            open_interest_source="live_api",
        )

        strict_trace = {
            "failed_layer": None if strict_passed else "layer4_fake_filter",
            "layers": {
                "layer4_fake_filter": {
                    "passed": strict_passed,
                    "details": strict_layer4,
                }
            },
        }
        soft_trace = {
            "failed_layer": None if soft_passed else "layer4_fake_filter",
            "layers": {
                "layer4_fake_filter": {
                    "passed": soft_passed,
                    "details": soft_layer4,
                }
            },
        }

        strict_audit.record(strict_trace, signal_side="SHORT" if strict_passed else None)
        soft_audit.record(soft_trace, signal_side="SHORT" if soft_passed else None)

        strict_snap = strict_audit.snapshot()
        soft_snap = soft_audit.snapshot()

        self.assertFalse(strict_passed)
        self.assertTrue(soft_passed)
        self.assertEqual(int(strict_snap.get("layer4_fail_count", 0)), 1)
        self.assertEqual(int(strict_snap.get("layer4_sentiment_blocker_count", 0)), 1)
        self.assertEqual(int(strict_snap.get("short_signal_count", 0)), 0)
        self.assertEqual(int(strict_snap.get("no_signal_count", 0)), 1)

        self.assertEqual(int(soft_snap.get("layer4_fail_count", 0)), 0)
        self.assertEqual(int(soft_snap.get("layer4_sentiment_blocker_count", 0)), 0)
        self.assertEqual(int(soft_snap.get("short_signal_count", 0)), 1)
        self.assertEqual(int(soft_snap.get("no_signal_count", 0)), 0)
    def test_audit_comparison_softened_funding_reduces_funding_blockers(self):
        df = self._build_df()

        strict_gen = SignalGenerator(SignalConfig(fake_filter_funding_supports_short_soft=0.0003))
        soft_gen = SignalGenerator(SignalConfig(fake_filter_funding_supports_short_soft=0.0005))
        strict_audit = StrategyAuditCollector()
        soft_audit = StrategyAuditCollector()

        strict_passed, strict_layer4 = strict_gen._layer4_fake_filter(
            df=df,
            side="SHORT",
            sentiment_index=90.0,
            sentiment_source="live_api",
            funding_rate=-0.0004,
            funding_source="live_api",
            long_short_ratio=1.2,
            long_short_ratio_source="live_api",
            open_interest_ratio=1.5,
            oi_source="live_api",
            open_interest_source="live_api",
        )
        soft_passed, soft_layer4 = soft_gen._layer4_fake_filter(
            df=df,
            side="SHORT",
            sentiment_index=90.0,
            sentiment_source="live_api",
            funding_rate=-0.0004,
            funding_source="live_api",
            long_short_ratio=1.2,
            long_short_ratio_source="live_api",
            open_interest_ratio=1.5,
            oi_source="live_api",
            open_interest_source="live_api",
        )

        strict_trace = {
            "failed_layer": None if strict_passed else "layer4_fake_filter",
            "layers": {
                "layer4_fake_filter": {
                    "passed": strict_passed,
                    "details": strict_layer4,
                }
            },
        }
        soft_trace = {
            "failed_layer": None if soft_passed else "layer4_fake_filter",
            "layers": {
                "layer4_fake_filter": {
                    "passed": soft_passed,
                    "details": soft_layer4,
                }
            },
        }

        strict_audit.record(strict_trace, signal_side="SHORT" if strict_passed else None)
        soft_audit.record(soft_trace, signal_side="SHORT" if soft_passed else None)

        strict_snap = strict_audit.snapshot()
        soft_snap = soft_audit.snapshot()

        self.assertFalse(strict_passed)
        self.assertTrue(soft_passed)
        self.assertEqual(int(strict_snap.get("layer4_fail_count", 0)), 1)
        self.assertEqual(int(strict_snap.get("layer4_funding_blocker_count", 0)), 1)
        self.assertEqual(int(strict_snap.get("layer4_sentiment_blocker_count", 0)), 0)
        self.assertEqual(int(strict_snap.get("short_signal_count", 0)), 0)
        self.assertEqual(int(strict_snap.get("no_signal_count", 0)), 1)

        self.assertEqual(int(soft_snap.get("layer4_fail_count", 0)), 0)
        self.assertEqual(int(soft_snap.get("layer4_funding_blocker_count", 0)), 0)
        self.assertEqual(int(soft_snap.get("layer4_sentiment_blocker_count", 0)), 0)
        self.assertEqual(int(soft_snap.get("short_signal_count", 0)), 1)
        self.assertEqual(int(soft_snap.get("no_signal_count", 0)), 0)
    def test_audit_comparison_softened_lsr_reduces_lsr_blockers(self):
        df = self._build_df()

        strict_gen = SignalGenerator(SignalConfig(fake_filter_lsr_extreme_soft=1.02))
        soft_gen = SignalGenerator(SignalConfig(fake_filter_lsr_extreme_soft=1.01))
        strict_audit = StrategyAuditCollector()
        soft_audit = StrategyAuditCollector()

        strict_passed, strict_layer4 = strict_gen._layer4_fake_filter(
            df=df,
            side="SHORT",
            sentiment_index=90.0,
            sentiment_source="live_api",
            funding_rate=-0.0004,
            funding_source="live_api",
            long_short_ratio=1.01,
            long_short_ratio_source="live_api",
            open_interest_ratio=1.5,
            oi_source="live_api",
            open_interest_source="live_api",
        )
        soft_passed, soft_layer4 = soft_gen._layer4_fake_filter(
            df=df,
            side="SHORT",
            sentiment_index=90.0,
            sentiment_source="live_api",
            funding_rate=-0.0004,
            funding_source="live_api",
            long_short_ratio=1.01,
            long_short_ratio_source="live_api",
            open_interest_ratio=1.5,
            oi_source="live_api",
            open_interest_source="live_api",
        )

        strict_trace = {
            "failed_layer": None if strict_passed else "layer4_fake_filter",
            "layers": {
                "layer4_fake_filter": {
                    "passed": strict_passed,
                    "details": strict_layer4,
                }
            },
        }
        soft_trace = {
            "failed_layer": None if soft_passed else "layer4_fake_filter",
            "layers": {
                "layer4_fake_filter": {
                    "passed": soft_passed,
                    "details": soft_layer4,
                }
            },
        }

        strict_audit.record(strict_trace, signal_side="SHORT" if strict_passed else None)
        soft_audit.record(soft_trace, signal_side="SHORT" if soft_passed else None)

        strict_snap = strict_audit.snapshot()
        soft_snap = soft_audit.snapshot()

        self.assertFalse(strict_passed)
        self.assertTrue(soft_passed)
        self.assertEqual(int(strict_snap.get("layer4_fail_count", 0)), 1)
        self.assertEqual(int(strict_snap.get("layer4_sentiment_blocker_count", 0)), 0)
        self.assertEqual(int(strict_snap.get("layer4_funding_blocker_count", 0)), 0)
        self.assertEqual(int(strict_snap.get("layer4_lsr_blocker_count", 0)), 1)
        self.assertEqual(int(strict_snap.get("short_signal_count", 0)), 0)
        self.assertEqual(int(strict_snap.get("no_signal_count", 0)), 1)

        self.assertEqual(int(soft_snap.get("layer4_fail_count", 0)), 0)
        self.assertEqual(int(soft_snap.get("layer4_sentiment_blocker_count", 0)), 0)
        self.assertEqual(int(soft_snap.get("layer4_funding_blocker_count", 0)), 0)
        self.assertEqual(int(soft_snap.get("layer4_lsr_blocker_count", 0)), 0)
        self.assertEqual(int(soft_snap.get("short_signal_count", 0)), 1)
        self.assertEqual(int(soft_snap.get("no_signal_count", 0)), 0)
    def test_layered_strategy_exposes_runtime_audit_snapshot(self):
        df = self._build_df()
        strategy = LayeredPumpStrategy(SignalConfig())
        exchange = ExchangeSnapshot(
            symbol="BTC/USDT",
            account=AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0),
            positions=[],
            open_orders=[],
        )

        strategy.generate(
            StrategyContext(
                symbol="BTC/USDT",
                market_ohlcv=df,
                mark_price=float(df.iloc[-1]["close"]),
                exchange=exchange,
                synced_state=TradeState.FLAT,
                sentiment_index=78.0,
                sentiment_source="provided",
                funding_rate=0.001,
                long_short_ratio=1.2,
            )
        )

        df_fail = df.copy()
        df_fail.iloc[-1, df_fail.columns.get_loc("vwap")] = 120.0
        strategy.generate(
            StrategyContext(
                symbol="BTC/USDT",
                market_ohlcv=df_fail,
                mark_price=float(df_fail.iloc[-1]["close"]),
                exchange=exchange,
                synced_state=TradeState.FLAT,
                sentiment_index=55.0,
                sentiment_source="provided",
                funding_rate=-0.01,
                long_short_ratio=0.9,
                open_interest=100.0,
                open_interest_source="provided",
            )
        )

        snap = strategy.audit_snapshot()
        self.assertGreaterEqual(int(snap.get("evaluated_count", 0)), 2)
        self.assertIn("short_signal_count", snap)
        self.assertIn("no_signal_count", snap)
        self.assertIn("failed_layer_counts", snap)
        self.assertGreaterEqual(int(snap.get("no_signal_count", 0)), 1)

    def test_regime_filter_pass_case_live_sources(self):
        df = self._build_df()
        df["adx"] = 10.0
        df["ema20"] = 103.0
        df["ema50"] = 104.0
        df["vwap"] = 104.0
        df["atr"] = 1.8

        signal_gen = SignalGenerator(SignalConfig())
        passed, regime_diag = signal_gen._regime_filter(
            df,
            MarketRegime.PUMP,
            news_veto=False,
            news_source="live_api",
        )

        self.assertTrue(passed)
        self.assertEqual(regime_diag.get("passed"), 1.0)
        self.assertEqual(regime_diag.get("htf_trend_ok"), 1.0)
        self.assertEqual(regime_diag.get("stretched_from_vwap"), 1.0)
        self.assertEqual(regime_diag.get("volatility_regime_ok"), 1.0)
        self.assertEqual(regime_diag.get("news_veto"), 1.0)
        self.assertEqual(regime_diag.get("degraded_mode"), 0.0)
        self.assertEqual(regime_diag.get("htf_trend_direction_context"), "not_strong_uptrend")
        self.assertGreaterEqual(float(regime_diag.get("htf_trend_metric_used", -1.0)), 0.0)
        self.assertGreater(float(regime_diag.get("htf_trend_threshold_used", 0.0)), 0.0)

    def test_regime_filter_fail_by_htf_trend(self):
        df = self._build_df()
        df["adx"] = 35.0
        df["ema20"] = 107.0
        df["ema50"] = 106.0
        df["vwap"] = 104.0
        df["atr"] = 1.8
        df.iloc[-1, df.columns.get_loc("close")] = 108.0

        signal_gen = SignalGenerator(SignalConfig())
        passed, regime_diag = signal_gen._regime_filter(
            df,
            MarketRegime.TREND,
            news_veto=False,
            news_source="live_api",
        )

        self.assertFalse(passed)
        self.assertEqual(regime_diag.get("htf_trend_ok"), 0.0)
        self.assertEqual(regime_diag.get("soft_pass_candidate"), 1.0)
        self.assertEqual(regime_diag.get("soft_pass_used"), 0.0)
        self.assertIn("htf_trend_ok", str(regime_diag.get("missing_conditions", "")))
        self.assertEqual(regime_diag.get("htf_trend_direction_context"), "strong_uptrend")
        self.assertGreaterEqual(
            float(regime_diag.get("htf_trend_metric_used", -1.0)),
            float(regime_diag.get("htf_trend_threshold_used", 0.0)),
        )

    def test_regime_filter_softened_htf_trend_gate_vs_strict(self):
        df = self._build_df()
        df["adx"] = 29.5
        df["ema20"] = 107.0
        df["ema50"] = 106.0
        df["vwap"] = 104.0
        df["atr"] = 1.8
        df.iloc[-1, df.columns.get_loc("close")] = 108.0

        strict_gen = SignalGenerator(SignalConfig(regime_strong_trend_adx=29.0))
        soft_gen = SignalGenerator(SignalConfig(regime_strong_trend_adx=30.0))

        strict_passed, strict_diag = strict_gen._regime_filter(
            df,
            MarketRegime.TREND,
            news_veto=False,
            news_source="live_api",
        )
        soft_passed, soft_diag = soft_gen._regime_filter(
            df,
            MarketRegime.TREND,
            news_veto=False,
            news_source="live_api",
        )

        self.assertFalse(strict_passed)
        self.assertTrue(soft_passed)
        self.assertEqual(strict_diag.get("htf_trend_ok"), 0.0)
        self.assertEqual(soft_diag.get("htf_trend_ok"), 1.0)
        self.assertEqual(strict_diag.get("htf_trend_direction_context"), "strong_uptrend")
        self.assertEqual(soft_diag.get("htf_trend_direction_context"), "not_strong_uptrend")
        self.assertLess(
            float(strict_diag.get("htf_trend_threshold_used", 0.0)),
            float(soft_diag.get("htf_trend_threshold_used", 0.0)),
        )

        strict_audit = StrategyAuditCollector()
        soft_audit = StrategyAuditCollector()
        strict_audit.record(
            {
                "failed_layer": "regime_filter",
                "layers": {
                    "regime_filter": {
                        "passed": strict_passed,
                        "details": strict_diag,
                    }
                },
            },
            signal_side=None,
        )
        soft_audit.record(
            {
                "failed_layer": None if soft_passed else "regime_filter",
                "layers": {
                    "regime_filter": {
                        "passed": soft_passed,
                        "details": soft_diag,
                    }
                },
            },
            signal_side="SHORT" if soft_passed else None,
        )

        self.assertEqual(int(strict_audit.snapshot().get("regime_filter_htf_trend_blocker_count", 0)), 1)
        self.assertEqual(int(soft_audit.snapshot().get("regime_filter_htf_trend_blocker_count", 0)), 0)

    def test_regime_filter_stretched_behavior_unchanged_with_htf_softening(self):
        df = self._build_df()
        df["adx"] = 10.0
        df["ema20"] = 103.0
        df["ema50"] = 104.0
        df["vwap"] = 104.0
        df["atr"] = 1.8
        df.iloc[-1, df.columns.get_loc("close")] = 104.02

        strict_gen = SignalGenerator(SignalConfig(regime_strong_trend_adx=29.0))
        soft_gen = SignalGenerator(SignalConfig(regime_strong_trend_adx=30.0))

        strict_passed, strict_diag = strict_gen._regime_filter(
            df,
            MarketRegime.PUMP,
            news_veto=False,
            news_source="live_api",
        )
        soft_passed, soft_diag = soft_gen._regime_filter(
            df,
            MarketRegime.PUMP,
            news_veto=False,
            news_source="live_api",
        )

        self.assertFalse(strict_passed)
        self.assertFalse(soft_passed)
        self.assertEqual(strict_diag.get("stretched_from_vwap"), 0.0)
        self.assertEqual(soft_diag.get("stretched_from_vwap"), 0.0)
        self.assertIn("stretched_from_vwap", str(strict_diag.get("missing_conditions", "")))
        self.assertIn("stretched_from_vwap", str(soft_diag.get("missing_conditions", "")))

    def test_regime_filter_volatility_behavior_unchanged_with_htf_softening(self):
        df = self._build_df()
        df["adx"] = 10.0
        df["ema20"] = 103.0
        df["ema50"] = 104.0
        df["vwap"] = 104.0
        df["atr"] = 0.01
        df.iloc[-1, df.columns.get_loc("close")] = 108.0

        strict_gen = SignalGenerator(SignalConfig(regime_strong_trend_adx=29.0))
        soft_gen = SignalGenerator(SignalConfig(regime_strong_trend_adx=30.0))

        strict_passed, strict_diag = strict_gen._regime_filter(
            df,
            MarketRegime.PUMP,
            news_veto=False,
            news_source="live_api",
        )
        soft_passed, soft_diag = soft_gen._regime_filter(
            df,
            MarketRegime.PUMP,
            news_veto=False,
            news_source="live_api",
        )

        self.assertFalse(strict_passed)
        self.assertFalse(soft_passed)
        self.assertEqual(strict_diag.get("volatility_regime_ok"), 0.0)
        self.assertEqual(soft_diag.get("volatility_regime_ok"), 0.0)
        self.assertIn("volatility_regime_ok", str(strict_diag.get("missing_conditions", "")))
        self.assertIn("volatility_regime_ok", str(soft_diag.get("missing_conditions", "")))
        self.assertEqual(
            float(strict_diag.get("volatility_threshold_used", 0.0)),
            float(soft_diag.get("volatility_threshold_used", 0.0)),
        )

    def test_layer4_behavior_unchanged_with_htf_trend_softening(self):
        df = self._build_df()
        strict_gen = SignalGenerator(SignalConfig(regime_strong_trend_adx=29.0))
        soft_gen = SignalGenerator(SignalConfig(regime_strong_trend_adx=30.0))

        strict_passed, strict_layer4 = strict_gen._layer4_fake_filter(
            df=df,
            side="SHORT",
            sentiment_index=90.0,
            sentiment_source="live_api",
            funding_rate=0.001,
            funding_source="live_api",
            long_short_ratio=1.2,
            long_short_ratio_source="live_api",
            open_interest_ratio=1.5,
            oi_source="live_api",
            open_interest_source="live_api",
        )
        soft_passed, soft_layer4 = soft_gen._layer4_fake_filter(
            df=df,
            side="SHORT",
            sentiment_index=90.0,
            sentiment_source="live_api",
            funding_rate=0.001,
            funding_source="live_api",
            long_short_ratio=1.2,
            long_short_ratio_source="live_api",
            open_interest_ratio=1.5,
            oi_source="live_api",
            open_interest_source="live_api",
        )

        self.assertEqual(strict_passed, soft_passed)
        self.assertEqual(strict_layer4.get("passed"), soft_layer4.get("passed"))
        self.assertEqual(strict_layer4.get("missing_conditions"), soft_layer4.get("missing_conditions"))
        self.assertEqual(strict_layer4.get("failed_reason"), soft_layer4.get("failed_reason"))
    def test_regime_filter_fail_by_vwap_stretch(self):
        df = self._build_df()
        df["adx"] = 10.0
        df["ema20"] = 103.0
        df["ema50"] = 104.0
        df["vwap"] = 104.0
        df["atr"] = 1.8
        df.iloc[-1, df.columns.get_loc("close")] = 104.02

        signal_gen = SignalGenerator(SignalConfig())
        passed, regime_diag = signal_gen._regime_filter(
            df,
            MarketRegime.PUMP,
            news_veto=False,
            news_source="live_api",
        )

        self.assertFalse(passed)
        self.assertEqual(regime_diag.get("stretched_from_vwap"), 0.0)
        self.assertEqual(regime_diag.get("soft_pass_candidate"), 1.0)
        self.assertEqual(regime_diag.get("soft_pass_used"), 0.0)
        self.assertEqual(regime_diag.get("soft_pass_reason"), "")
        self.assertEqual(regime_diag.get("fail_due_to_degraded_mode_only"), 0.0)
        self.assertIn("stretched_from_vwap", str(regime_diag.get("missing_conditions", "")))
        self.assertIn("vwap_distance_metric_used", regime_diag)
        self.assertIn("vwap_stretch_threshold_used", regime_diag)
        self.assertLess(
            float(regime_diag.get("vwap_distance_metric_used", 0.0)),
            float(regime_diag.get("vwap_stretch_threshold_used", 0.0)),
        )

    def test_regime_filter_softened_vwap_stretch_gate_vs_strict(self):
        df = self._build_df()
        df["adx"] = 10.0
        df["ema20"] = 99.0
        df["ema50"] = 100.0
        df["vwap"] = 100.0
        df["atr"] = 1.2
        df.iloc[-1, df.columns.get_loc("close")] = 100.13

        strict_gen = SignalGenerator(SignalConfig(regime_vwap_stretch_soft=0.0015))
        soft_gen = SignalGenerator(SignalConfig(regime_vwap_stretch_soft=0.0012))

        strict_passed, strict_diag = strict_gen._regime_filter(
            df,
            MarketRegime.PUMP,
            news_veto=False,
            news_source="live_api",
        )
        soft_passed, soft_diag = soft_gen._regime_filter(
            df,
            MarketRegime.PUMP,
            news_veto=False,
            news_source="live_api",
        )

        self.assertFalse(strict_passed)
        self.assertTrue(soft_passed)
        self.assertEqual(strict_diag.get("htf_trend_ok"), soft_diag.get("htf_trend_ok"))
        self.assertEqual(strict_diag.get("volatility_regime_ok"), soft_diag.get("volatility_regime_ok"))
        self.assertEqual(strict_diag.get("news_veto"), soft_diag.get("news_veto"))
        self.assertEqual(strict_diag.get("stretched_from_vwap"), 0.0)
        self.assertEqual(soft_diag.get("stretched_from_vwap"), 1.0)
        self.assertGreater(
            float(strict_diag.get("vwap_stretch_threshold_used", 0.0)),
            float(soft_diag.get("vwap_stretch_threshold_used", 0.0)),
        )
        self.assertIn("stretched_from_vwap", str(strict_diag.get("missing_conditions", "")))
        self.assertEqual(str(soft_diag.get("failed_reason", "")), "none")

        strict_audit = StrategyAuditCollector()
        soft_audit = StrategyAuditCollector()
        strict_audit.record(
            {
                "failed_layer": "regime_filter",
                "layers": {
                    "regime_filter": {
                        "passed": strict_passed,
                        "details": strict_diag,
                    }
                },
            },
            signal_side=None,
        )
        soft_audit.record(
            {
                "failed_layer": None if soft_passed else "regime_filter",
                "layers": {
                    "regime_filter": {
                        "passed": soft_passed,
                        "details": soft_diag,
                    }
                },
            },
            signal_side="SHORT" if soft_passed else None,
        )

        self.assertEqual(int(strict_audit.snapshot().get("regime_filter_vwap_stretch_blocker_count", 0)), 1)
        self.assertEqual(int(soft_audit.snapshot().get("regime_filter_vwap_stretch_blocker_count", 0)), 0)

    def test_regime_filter_fail_by_volatility_regime(self):
        df = self._build_df()
        df["adx"] = 10.0
        df["ema20"] = 103.0
        df["ema50"] = 104.0
        df["vwap"] = 104.0
        df["atr"] = 0.01

        signal_gen = SignalGenerator(SignalConfig())
        passed, regime_diag = signal_gen._regime_filter(
            df,
            MarketRegime.PUMP,
            news_veto=False,
            news_source="live_api",
        )

        self.assertFalse(passed)
        self.assertEqual(regime_diag.get("volatility_regime_ok"), 0.0)
        self.assertEqual(regime_diag.get("soft_pass_candidate"), 1.0)
        self.assertEqual(regime_diag.get("fail_due_to_degraded_mode_only"), 0.0)
        self.assertIn("volatility_regime_ok", str(regime_diag.get("missing_conditions", "")))
        self.assertIn("volatility_threshold_used", regime_diag)
        self.assertGreater(float(regime_diag.get("volatility_threshold_used", 0.0)), 0.0)
        self.assertIn("atr_norm", regime_diag)

    def test_regime_filter_softened_volatility_gate_vs_strict(self):
        df = self._build_df()
        df["adx"] = 10.0
        df["ema20"] = 103.0
        df["ema50"] = 104.0
        df["vwap"] = 104.0
        df.iloc[-1, df.columns.get_loc("close")] = 108.0
        df["atr"] = 0.168

        strict_gen = SignalGenerator(SignalConfig(regime_volatility_entry_tolerance_mult=0.40))
        soft_gen = SignalGenerator(SignalConfig(regime_volatility_entry_tolerance_mult=0.38))

        strict_passed, strict_diag = strict_gen._regime_filter(
            df,
            MarketRegime.PUMP,
            news_veto=False,
            news_source="live_api",
        )
        soft_passed, soft_diag = soft_gen._regime_filter(
            df,
            MarketRegime.PUMP,
            news_veto=False,
            news_source="live_api",
        )

        self.assertFalse(strict_passed)
        self.assertTrue(soft_passed)
        self.assertEqual(strict_diag.get("volatility_regime_ok"), 0.0)
        self.assertEqual(soft_diag.get("volatility_regime_ok"), 1.0)
        self.assertGreater(
            float(strict_diag.get("volatility_threshold_used", 0.0)),
            float(soft_diag.get("volatility_threshold_used", 0.0)),
        )
        self.assertIn("volatility_regime_ok", str(strict_diag.get("missing_conditions", "")))
        self.assertEqual(str(soft_diag.get("failed_reason", "")), "none")

        strict_audit = StrategyAuditCollector()
        soft_audit = StrategyAuditCollector()
        strict_audit.record(
            {
                "failed_layer": "regime_filter",
                "layers": {
                    "regime_filter": {
                        "passed": strict_passed,
                        "details": strict_diag,
                    }
                },
            },
            signal_side=None,
        )
        soft_audit.record(
            {
                "failed_layer": None if soft_passed else "regime_filter",
                "layers": {
                    "regime_filter": {
                        "passed": soft_passed,
                        "details": soft_diag,
                    }
                },
            },
            signal_side="SHORT" if soft_passed else None,
        )

        self.assertEqual(int(strict_audit.snapshot().get("regime_filter_volatility_blocker_count", 0)), 1)
        self.assertEqual(int(soft_audit.snapshot().get("regime_filter_volatility_blocker_count", 0)), 0)

    def test_layer4_behavior_unchanged_with_volatility_softening(self):
        df = self._build_df()
        strict_gen = SignalGenerator(SignalConfig(regime_volatility_entry_tolerance_mult=0.40))
        soft_gen = SignalGenerator(SignalConfig(regime_volatility_entry_tolerance_mult=0.38))

        strict_passed, strict_layer4 = strict_gen._layer4_fake_filter(
            df=df,
            side="SHORT",
            sentiment_index=90.0,
            sentiment_source="live_api",
            funding_rate=0.001,
            funding_source="live_api",
            long_short_ratio=1.2,
            long_short_ratio_source="live_api",
            open_interest_ratio=1.5,
            oi_source="live_api",
            open_interest_source="live_api",
        )
        soft_passed, soft_layer4 = soft_gen._layer4_fake_filter(
            df=df,
            side="SHORT",
            sentiment_index=90.0,
            sentiment_source="live_api",
            funding_rate=0.001,
            funding_source="live_api",
            long_short_ratio=1.2,
            long_short_ratio_source="live_api",
            open_interest_ratio=1.5,
            oi_source="live_api",
            open_interest_source="live_api",
        )

        self.assertEqual(strict_passed, soft_passed)
        self.assertEqual(strict_layer4.get("passed"), soft_layer4.get("passed"))
        self.assertEqual(strict_layer4.get("missing_conditions"), soft_layer4.get("missing_conditions"))
        self.assertEqual(strict_layer4.get("failed_reason"), soft_layer4.get("failed_reason"))

    def test_runtime_default_volatility_override_creates_targeted_softer_gate(self):
        df = self._build_df()
        df["adx"] = 10.0
        df["ema20"] = 103.0
        df["ema50"] = 104.0
        df.iloc[-1, df.columns.get_loc("close")] = 108.0
        df["vwap"] = 107.8
        df["atr"] = 0.0702

        base_gen = SignalGenerator(SignalConfig())
        runtime_strategy = LayeredPumpStrategy()

        base_passed, base_diag = base_gen._regime_filter(
            df,
            MarketRegime.PUMP,
            news_veto=False,
            news_source="live_api",
        )
        runtime_passed, runtime_diag = runtime_strategy._generator._regime_filter(
            df,
            MarketRegime.PUMP,
            news_veto=False,
            news_source="live_api",
        )

        self.assertFalse(base_passed)
        self.assertTrue(runtime_passed)
        self.assertEqual(base_diag.get("volatility_regime_ok"), 0.0)
        self.assertEqual(runtime_diag.get("volatility_regime_ok"), 1.0)
        self.assertAlmostEqual(float(base_diag.get("volatility_threshold_used", 0.0)), 0.001, places=8)
        self.assertAlmostEqual(float(runtime_diag.get("volatility_threshold_used", 0.0)), 0.0006, places=8)
        self.assertEqual(runtime_diag.get("failed_reason"), "none")

    def test_runtime_default_regime_soft_pass_allows_vwap_only_miss(self):
        df = self._build_df()
        df["adx"] = 10.0
        df["ema20"] = 103.0
        df["ema50"] = 104.0
        df.iloc[-1, df.columns.get_loc("close")] = 108.0
        df["vwap"] = 108.0
        df["atr"] = 0.25

        base_gen = SignalGenerator(SignalConfig())
        runtime_gen = LayeredPumpStrategy()._generator

        base_passed, base_diag = base_gen._regime_filter(df, MarketRegime.PUMP, news_veto=False, news_source="live_api")
        runtime_passed, runtime_diag = runtime_gen._regime_filter(df, MarketRegime.PUMP, news_veto=False, news_source="live_api")

        self.assertFalse(base_passed)
        self.assertTrue(runtime_passed)
        self.assertEqual(base_diag.get("stretched_from_vwap"), 0.0)
        self.assertEqual(runtime_diag.get("stretched_from_vwap"), 0.0)
        self.assertEqual(base_diag.get("soft_pass_used"), 0.0)
        self.assertEqual(runtime_diag.get("soft_pass_used"), 1.0)
        self.assertEqual(runtime_diag.get("soft_pass_reason"), "stretched_from_vwap")
        self.assertEqual(runtime_diag.get("failed_reason"), "none")
        self.assertEqual(runtime_diag.get("missing_conditions"), "")

    def test_runtime_default_regime_soft_pass_allows_htf_only_miss(self):
        df = self._build_df()
        df["adx"] = 40.0
        df["ema20"] = 108.5
        df["ema50"] = 107.5
        df.iloc[-1, df.columns.get_loc("close")] = 109.0
        df["vwap"] = 108.8
        df["atr"] = 0.25

        base_gen = SignalGenerator(SignalConfig())
        runtime_gen = LayeredPumpStrategy()._generator

        base_passed, base_diag = base_gen._regime_filter(df, MarketRegime.PUMP, news_veto=False, news_source="live_api")
        runtime_passed, runtime_diag = runtime_gen._regime_filter(df, MarketRegime.PUMP, news_veto=False, news_source="live_api")

        self.assertFalse(base_passed)
        self.assertTrue(runtime_passed)
        self.assertEqual(base_diag.get("htf_trend_ok"), 0.0)
        self.assertEqual(runtime_diag.get("htf_trend_ok"), 0.0)
        self.assertIn("htf_trend_ok", str(base_diag.get("missing_conditions", "")))
        self.assertEqual(runtime_diag.get("soft_pass_candidate"), 1.0)
        self.assertEqual(runtime_diag.get("soft_pass_used"), 1.0)
        self.assertEqual(runtime_diag.get("soft_pass_reason"), "htf_trend_ok")
        self.assertEqual(runtime_diag.get("failed_reason"), "none")
        self.assertEqual(runtime_diag.get("missing_conditions"), "")

    def test_regime_filter_vwap_behavior_uses_runtime_soft_pass(self):
        df = self._build_df()
        df["adx"] = 10.0
        df["ema20"] = 103.0
        df["ema50"] = 104.0
        df.iloc[-1, df.columns.get_loc("close")] = 108.0
        df["vwap"] = 108.0
        df["atr"] = 0.25

        base_gen = SignalGenerator(SignalConfig())
        runtime_gen = LayeredPumpStrategy()._generator

        base_passed, base_diag = base_gen._regime_filter(df, MarketRegime.PUMP, news_veto=False, news_source="live_api")
        runtime_passed, runtime_diag = runtime_gen._regime_filter(df, MarketRegime.PUMP, news_veto=False, news_source="live_api")

        self.assertFalse(base_passed)
        self.assertTrue(runtime_passed)
        self.assertEqual(base_diag.get("stretched_from_vwap"), 0.0)
        self.assertEqual(runtime_diag.get("stretched_from_vwap"), 0.0)
        self.assertIn("stretched_from_vwap", str(base_diag.get("missing_conditions", "")))
        self.assertEqual(runtime_diag.get("soft_pass_used"), 1.0)
        self.assertEqual(runtime_diag.get("soft_pass_reason"), "stretched_from_vwap")

    def test_runtime_default_layer1_soft_pass_allows_recent_pump_near_miss(self):
        df = self._build_df()
        for offset in range(1, 13):
            df.iloc[-offset, df.columns.get_loc("rsi")] = 55.0
            df.iloc[-offset, df.columns.get_loc("volume_spike")] = 1.1
            df.iloc[-offset, df.columns.get_loc("bb_upper")] = 120.0
            df.iloc[-offset, df.columns.get_loc("kc_upper")] = 120.0
        df.iloc[-8, df.columns.get_loc("rsi")] = 85.0
        df.iloc[-8, df.columns.get_loc("volume_spike")] = 8.0
        df.iloc[-8, df.columns.get_loc("bb_upper")] = 120.0
        df.iloc[-8, df.columns.get_loc("kc_upper")] = 120.0

        base_gen = SignalGenerator(SignalConfig(layer1_pump_lookback_bars=4))
        runtime_gen = LayeredPumpStrategy()._generator

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

        base_signal = base_gen.generate(ctx)
        runtime_signal = runtime_gen.generate(ctx)

        self.assertIsNone(base_signal)
        self.assertIsNotNone(runtime_signal)
        self.assertEqual(runtime_signal.side, "SHORT")
        layer1 = runtime_signal.details.get("layer1", {})
        self.assertEqual(layer1.get("soft_pass_candidate"), 1.0)
        self.assertEqual(layer1.get("soft_pass_used"), 1.0)
        self.assertEqual(layer1.get("soft_pass_reason"), "upper_band_breakout")
        self.assertEqual(layer1.get("pump_bar_offset"), 7)

    def test_runtime_default_layer1_window_context_soft_pass_allows_split_recent_pump(self):
        df = self._build_df()
        for offset in range(1, 13):
            df.iloc[-offset, df.columns.get_loc("rsi")] = 55.0
            df.iloc[-offset, df.columns.get_loc("volume_spike")] = 1.1
            df.iloc[-offset, df.columns.get_loc("bb_upper")] = 120.0
            df.iloc[-offset, df.columns.get_loc("kc_upper")] = 120.0
            df.iloc[-offset, df.columns.get_loc("close")] = 103.0

        df.iloc[-10, df.columns.get_loc("rsi")] = 85.0
        df.iloc[-6, df.columns.get_loc("volume_spike")] = 8.0
        df.iloc[-3, df.columns.get_loc("close")] = 121.0

        base_gen = SignalGenerator(SignalConfig(layer1_pump_lookback_bars=12))
        runtime_gen = LayeredPumpStrategy()._generator

        base_side, base_layer1 = base_gen._layer1_pump_detection(df)
        runtime_side, runtime_layer1 = runtime_gen._layer1_pump_detection(df)

        self.assertIsNone(base_side)
        self.assertEqual(base_layer1.get("soft_pass_used"), 0.0)
        self.assertEqual(runtime_side, "SHORT")
        self.assertEqual(runtime_layer1.get("soft_pass_used"), 1.0)
        self.assertEqual(runtime_layer1.get("soft_pass_reason"), "window_pump_context")

    def test_runtime_default_layer1_near_upper_band_soft_pass_allows_current_bar_fade(self):
        df = self._build_df()
        for offset in range(1, 13):
            df.iloc[-offset, df.columns.get_loc("rsi")] = 55.0
            df.iloc[-offset, df.columns.get_loc("volume_spike")] = 1.1
            df.iloc[-offset, df.columns.get_loc("bb_upper")] = 120.0
            df.iloc[-offset, df.columns.get_loc("kc_upper")] = 120.0
        df.iloc[-1, df.columns.get_loc("rsi")] = 52.0
        df.iloc[-1, df.columns.get_loc("volume_spike")] = 1.45
        df.iloc[-1, df.columns.get_loc("close")] = 105.95
        df.iloc[-1, df.columns.get_loc("bb_upper")] = 106.0
        df.iloc[-1, df.columns.get_loc("kc_upper")] = 106.0

        base_gen = SignalGenerator(SignalConfig())
        runtime_gen = LayeredPumpStrategy()._generator

        base_side, base_layer1 = base_gen._layer1_pump_detection(df)
        runtime_side, runtime_layer1 = runtime_gen._layer1_pump_detection(df)

        self.assertIsNone(base_side)
        self.assertEqual(base_layer1.get("soft_pass_used"), 0.0)
        self.assertEqual(runtime_side, "SHORT")
        self.assertEqual(runtime_layer1.get("soft_pass_used"), 1.0)
        self.assertEqual(runtime_layer1.get("soft_pass_reason"), "near_upper_band_context")

    def test_layer4_behavior_unchanged_with_runtime_volatility_override(self):
        df = self._build_df()
        base_gen = SignalGenerator(SignalConfig())
        runtime_gen = LayeredPumpStrategy()._generator

        base_passed, base_layer4 = base_gen._layer4_fake_filter(
            df=df,
            side="SHORT",
            sentiment_index=90.0,
            sentiment_source="live_api",
            funding_rate=0.001,
            funding_source="live_api",
            long_short_ratio=1.2,
            long_short_ratio_source="live_api",
            open_interest_ratio=1.5,
            oi_source="live_api",
            open_interest_source="live_api",
        )
        runtime_passed, runtime_layer4 = runtime_gen._layer4_fake_filter(
            df=df,
            side="SHORT",
            sentiment_index=90.0,
            sentiment_source="live_api",
            funding_rate=0.001,
            funding_source="live_api",
            long_short_ratio=1.2,
            long_short_ratio_source="live_api",
            open_interest_ratio=1.5,
            oi_source="live_api",
            open_interest_source="live_api",
        )

        self.assertEqual(base_passed, runtime_passed)
        self.assertEqual(base_layer4.get("passed"), runtime_layer4.get("passed"))
        self.assertEqual(base_layer4.get("missing_conditions"), runtime_layer4.get("missing_conditions"))
        self.assertEqual(base_layer4.get("failed_reason"), runtime_layer4.get("failed_reason"))

    def test_regime_filter_fail_by_news_veto(self):
        df = self._build_df()
        df["adx"] = 10.0
        df["ema20"] = 103.0
        df["ema50"] = 104.0
        df["vwap"] = 104.0
        df["atr"] = 1.8

        signal_gen = SignalGenerator(SignalConfig())
        passed, regime_diag = signal_gen._regime_filter(
            df,
            MarketRegime.PUMP,
            news_veto=True,
            news_source="live_api",
        )

        self.assertFalse(passed)
        self.assertEqual(regime_diag.get("news_veto"), 0.0)
        self.assertIn("news_veto", str(regime_diag.get("missing_conditions", "")))

    def test_regime_filter_pass_with_optional_unavailable_news_not_marked_degraded(self):
        df = self._build_df()
        signal_gen = SignalGenerator(SignalConfig())

        passed, regime_diag = signal_gen._regime_filter(
            df,
            MarketRegime.PUMP,
            news_veto=None,
            news_source="unavailable",
        )

        self.assertTrue(passed)
        self.assertEqual(regime_diag.get("passed"), 1.0)
        self.assertEqual(regime_diag.get("degraded_mode"), 0.0)
        self.assertEqual(regime_diag.get("fail_due_to_degraded_mode_only"), 0.0)
        self.assertEqual(regime_diag.get("soft_pass_candidate"), 0.0)
        self.assertEqual(regime_diag.get("failed_reason"), "none")
        self.assertIn("source_flags", regime_diag)
        self.assertEqual(regime_diag.get("source_flags", {}).get("news_quality"), "unavailable")

    def test_regime_filter_fail_collects_missing_conditions(self):
        df = self._build_df()
        df.iloc[-1, df.columns.get_loc("close")] = 100.0
        df["vwap"] = 104.0
        df["atr"] = 0.02
        df["adx"] = 35.0
        df["ema20"] = 95.0
        df["ema50"] = 90.0

        signal_gen = SignalGenerator(SignalConfig())
        passed, regime_diag = signal_gen._regime_filter(
            df,
            MarketRegime.TREND,
            news_veto=True,
            news_source="provided",
        )

        self.assertFalse(passed)
        self.assertEqual(regime_diag.get("passed"), 0.0)
        self.assertTrue(str(regime_diag.get("failed_reason", "")).startswith("missing:"))
        missing = str(regime_diag.get("missing_conditions", ""))
        self.assertIn("htf_trend_ok", missing)
        self.assertIn("stretched_from_vwap", missing)
        self.assertIn("volatility_regime_ok", missing)
        self.assertIn("news_veto", missing)

    def test_generate_stops_at_regime_filter_before_layer1(self):
        df = self._build_df()
        df.iloc[-1, df.columns.get_loc("close")] = 100.0
        df["vwap"] = 104.0
        df["atr"] = 0.02
        df["adx"] = 35.0
        df["ema20"] = 95.0
        df["ema50"] = 90.0

        signal_gen = SignalGenerator(SignalConfig())
        ctx = SignalContext(
            symbol="BTC/USDT",
            df=df,
            volume_profile=VolumeProfileLevels(poc=102.0, vah=109.0, val=98.0),
            regime=MarketRegime.TREND,
            sentiment_index=78.0,
            sentiment_source="provided",
            funding_rate=0.001,
            long_short_ratio=1.2,
            news_veto=True,
            news_source="provided",
        )

        signal = signal_gen.generate(ctx)
        self.assertIsNone(signal)
        trace = signal_gen.last_diagnostics
        self.assertEqual(trace.get("failed_layer"), "regime_filter")
        self.assertTrue(trace.get("layers", {}).get("regime_filter", {}).get("passed") in (0, 0.0, False))
        self.assertNotIn("layer1_pump_detection", trace.get("layers", {}))

    def test_calibration_normalized_metrics_computed_correctly(self):
        records = [
            {
                "ts": "2026-03-12T00:00:00+00:00",
                "compact": {
                    "evaluations_total": 1,
                    "insufficient_history_count": 0,
                    "regime_filter_pass_count": 0,
                    "regime_filter_fail_count": 1,
                    "layer1_pass_count": 0,
                    "layer1_fail_count": 0,
                    "layer2_fail_count": 0,
                    "layer3_fail_count": 0,
                    "layer4_fail_count": 0,
                    "layer5_fail_count": 0,
                    "short_signal_count": 0,
                    "no_signal_count": 1,
                    "regime_filter_htf_trend_blocker_count": 1,
                    "regime_filter_vwap_stretch_blocker_count": 1,
                    "regime_filter_volatility_blocker_count": 0,
                    "regime_filter_news_blocker_count": 0,
                    "regime_filter_degraded_only_count": 0,
                    "regime_filter_soft_pass_candidate_count": 0,
                    "layer1_rsi_high_blocker_count": 0,
                    "layer1_volume_spike_blocker_count": 0,
                    "layer1_above_bollinger_upper_blocker_count": 0,
                    "layer1_above_keltner_upper_blocker_count": 0,
                    "layer4_sentiment_blocker_count": 0,
                    "layer4_funding_blocker_count": 0,
                    "layer4_lsr_blocker_count": 0,
                    "layer4_oi_blocker_count": 0,
                    "layer4_price_blocker_count": 0,
                    "layer4_degraded_mode_count": 0,
                    "layer4_soft_pass_candidate_count": 0,
                    "layer5_fallback_rr_used_count": 0,
                    "layer5_vp_based_count": 0,
                    "layer5_fail_missing_atr_count": 0,
                    "layer5_fail_missing_volume_profile_count": 0,
                    "regime_source_quality": {"live": 1, "fallback": 0, "unavailable": 1},
                    "layer4_source_quality": {"live": 0, "fallback": 0, "unavailable": 0},
                },
                "audit": {
                    "reached_layer1_count": 0,
                    "failed_layer_counts": {"regime_filter": 1},
                },
            },
            {
                "ts": "2026-03-12T00:01:00+00:00",
                "compact": {
                    "evaluations_total": 1,
                    "insufficient_history_count": 0,
                    "regime_filter_pass_count": 1,
                    "regime_filter_fail_count": 0,
                    "layer1_pass_count": 0,
                    "layer1_fail_count": 1,
                    "layer2_fail_count": 0,
                    "layer3_fail_count": 0,
                    "layer4_fail_count": 0,
                    "layer5_fail_count": 0,
                    "short_signal_count": 0,
                    "no_signal_count": 1,
                    "regime_filter_htf_trend_blocker_count": 0,
                    "regime_filter_vwap_stretch_blocker_count": 0,
                    "regime_filter_volatility_blocker_count": 0,
                    "regime_filter_news_blocker_count": 0,
                    "regime_filter_degraded_only_count": 0,
                    "regime_filter_soft_pass_candidate_count": 0,
                    "layer1_rsi_high_blocker_count": 1,
                    "layer1_volume_spike_blocker_count": 0,
                    "layer1_above_bollinger_upper_blocker_count": 0,
                    "layer1_above_keltner_upper_blocker_count": 0,
                    "layer4_sentiment_blocker_count": 0,
                    "layer4_funding_blocker_count": 0,
                    "layer4_lsr_blocker_count": 0,
                    "layer4_oi_blocker_count": 0,
                    "layer4_price_blocker_count": 0,
                    "layer4_degraded_mode_count": 0,
                    "layer4_soft_pass_candidate_count": 0,
                    "layer5_fallback_rr_used_count": 0,
                    "layer5_vp_based_count": 0,
                    "layer5_fail_missing_atr_count": 0,
                    "layer5_fail_missing_volume_profile_count": 0,
                    "regime_source_quality": {"live": 1, "fallback": 0, "unavailable": 1},
                    "layer4_source_quality": {"live": 0, "fallback": 0, "unavailable": 0},
                },
                "audit": {
                    "reached_layer1_count": 1,
                    "failed_layer_counts": {"layer1_pump_detection": 1},
                },
            },
        ]

        summary = aggregate_observation(records)
        self.assertEqual(int(summary.get("evaluations_total", 0)), 2)
        self.assertAlmostEqual(float(summary.get("regime_filter_pass_rate", 0.0)), 0.5, places=6)
        self.assertAlmostEqual(float(summary.get("regime_filter_fail_rate", 0.0)), 0.5, places=6)
        self.assertAlmostEqual(float(summary.get("layer1_reach_rate", 0.0)), 0.5, places=6)
        self.assertAlmostEqual(float(summary.get("layer1_fail_rate_given_reach", 0.0)), 1.0, places=6)
        self.assertAlmostEqual(
            float(summary.get("blocker_rate_per_sample", {}).get("regime_filter", {}).get("htf_trend_ok", 0.0)),
            0.5,
            places=6,
        )

    def test_calibration_zero_regime_pass_forbids_layer1_target(self):
        before_summary = {
            "evaluations_total": 60,
            "regime_filter_pass_rate": 0.0,
            "regime_filter_fail_count": 60,
            "regime_filter_htf_trend_blocker_count": 10,
            "regime_filter_vwap_stretch_blocker_count": 8,
            "regime_filter_volatility_blocker_count": 42,
            "regime_filter_news_blocker_count": 0,
            "top_regime_filter_blocker": "volatility_regime_ok",
            "blocker_dominance_share": {"regime_filter": 0.70},
            "blocker_rate_per_regime_fail": {
                "htf_trend_ok": 10 / 60,
                "stretched_from_vwap": 8 / 60,
                "volatility_regime_ok": 42 / 60,
                "news_veto": 0.0,
            },
            "layer1_fail_count": 0,
            "layer2_fail_count": 0,
            "layer3_fail_count": 0,
            "layer4_fail_count": 0,
            "layer5_fail_count": 0,
        }
        after_summary = {
            "evaluations_total": 60,
            "regime_filter_pass_rate": 0.0,
            "regime_filter_fail_count": 60,
            "regime_filter_htf_trend_blocker_count": 11,
            "regime_filter_vwap_stretch_blocker_count": 6,
            "regime_filter_volatility_blocker_count": 43,
            "regime_filter_news_blocker_count": 0,
            "top_regime_filter_blocker": "volatility_regime_ok",
            "blocker_dominance_share": {"regime_filter": 0.717},
            "blocker_rate_per_regime_fail": {
                "htf_trend_ok": 11 / 60,
                "stretched_from_vwap": 6 / 60,
                "volatility_regime_ok": 43 / 60,
                "news_veto": 0.0,
            },
            "layer1_fail_count": 8,
            "layer2_fail_count": 0,
            "layer3_fail_count": 0,
            "layer4_fail_count": 0,
            "layer5_fail_count": 0,
        }

        rec = recommend_calibration_step(after_summary=after_summary, before_summary=before_summary)
        self.assertTrue(bool(rec.get("SAFE_TO_CONTINUE")))
        self.assertEqual(str(rec.get("NEXT_TARGET_LAYER", "")), "regime_filter")
        self.assertNotEqual(str(rec.get("NEXT_TARGET_LAYER", "")), "layer1_pump_detection")

    def test_calibration_guardrail_inconsistent_windows_requires_observe_more(self):
        before_summary = {
            "evaluations_total": 30,
            "regime_filter_pass_rate": 0.0,
            "regime_filter_fail_count": 30,
            "top_regime_filter_blocker": "volatility_regime_ok",
            "blocker_dominance_share": {"regime_filter": 0.8},
            "blocker_rate_per_regime_fail": {
                "htf_trend_ok": 0.1,
                "stretched_from_vwap": 0.1,
                "volatility_regime_ok": 0.8,
                "news_veto": 0.0,
            },
            "regime_filter_htf_trend_blocker_count": 3,
            "regime_filter_vwap_stretch_blocker_count": 3,
            "regime_filter_volatility_blocker_count": 24,
            "regime_filter_news_blocker_count": 0,
            "layer1_fail_count": 0,
            "layer2_fail_count": 0,
            "layer3_fail_count": 0,
            "layer4_fail_count": 0,
            "layer5_fail_count": 0,
        }
        after_summary = {
            "evaluations_total": 120,
            "regime_filter_pass_rate": 0.0,
            "regime_filter_fail_count": 120,
            "top_regime_filter_blocker": "volatility_regime_ok",
            "blocker_dominance_share": {"regime_filter": 0.8},
            "blocker_rate_per_regime_fail": {
                "htf_trend_ok": 0.1,
                "stretched_from_vwap": 0.1,
                "volatility_regime_ok": 0.8,
                "news_veto": 0.0,
            },
            "regime_filter_htf_trend_blocker_count": 12,
            "regime_filter_vwap_stretch_blocker_count": 12,
            "regime_filter_volatility_blocker_count": 96,
            "regime_filter_news_blocker_count": 0,
            "layer1_fail_count": 0,
            "layer2_fail_count": 0,
            "layer3_fail_count": 0,
            "layer4_fail_count": 0,
            "layer5_fail_count": 0,
        }

        quality = assess_window_quality(after_summary=after_summary, before_summary=before_summary)
        self.assertFalse(bool(quality.get("comparable_window_size", True)))

        rec = recommend_calibration_step(after_summary=after_summary, before_summary=before_summary)
        self.assertFalse(bool(rec.get("SAFE_TO_CONTINUE")))
        self.assertEqual(str(rec.get("STOP_REASON", "")), "window_size_not_comparable")
        self.assertIn("Collect another window with a comparable sample count", str(rec.get("RUNBOOK_ACTIONS", [""])[0]))

    def test_calibration_recommendation_stable_single_target_allowed(self):
        before_summary = {
            "evaluations_total": 80,
            "regime_filter_pass_rate": 0.0,
            "regime_filter_fail_count": 80,
            "regime_filter_htf_trend_blocker_count": 16,
            "regime_filter_vwap_stretch_blocker_count": 8,
            "regime_filter_volatility_blocker_count": 56,
            "regime_filter_news_blocker_count": 0,
            "top_regime_filter_blocker": "volatility_regime_ok",
            "top_blocker_by_raw_coverage": "volatility_regime_ok",
            "top_blocker_by_label": "volatility_regime_ok",
            "co_dominant_blockers": ["volatility_regime_ok"],
            "blocker_dominance_share": {"regime_filter": 0.70},
            "blocker_rate_per_regime_fail": {
                "htf_trend_ok": 0.20,
                "stretched_from_vwap": 0.10,
                "volatility_regime_ok": 0.70,
                "news_veto": 0.0,
            },
            "layer1_fail_count": 0,
            "layer2_fail_count": 0,
            "layer3_fail_count": 0,
            "layer4_fail_count": 0,
            "layer5_fail_count": 0,
        }
        after_summary = {
            "evaluations_total": 88,
            "regime_filter_pass_rate": 0.0,
            "regime_filter_fail_count": 88,
            "regime_filter_htf_trend_blocker_count": 18,
            "regime_filter_vwap_stretch_blocker_count": 9,
            "regime_filter_volatility_blocker_count": 61,
            "regime_filter_news_blocker_count": 0,
            "top_regime_filter_blocker": "volatility_regime_ok",
            "top_blocker_by_raw_coverage": "volatility_regime_ok",
            "top_blocker_by_label": "volatility_regime_ok",
            "co_dominant_blockers": ["volatility_regime_ok"],
            "blocker_dominance_share": {"regime_filter": 61 / 88},
            "blocker_rate_per_regime_fail": {
                "htf_trend_ok": 18 / 88,
                "stretched_from_vwap": 9 / 88,
                "volatility_regime_ok": 61 / 88,
                "news_veto": 0.0,
            },
            "layer1_fail_count": 0,
            "layer2_fail_count": 0,
            "layer3_fail_count": 0,
            "layer4_fail_count": 0,
            "layer5_fail_count": 0,
        }

        rec = recommend_calibration_step(after_summary=after_summary, before_summary=before_summary)
        self.assertTrue(bool(rec.get("SAFE_TO_CONTINUE")))
        self.assertEqual(str(rec.get("ACTION_VERDICT", "")), "single_blocker_ready")
        self.assertIn(
            "Prepare one isolated tweak for regime_filter.volatility_regime_ok.",
            str(rec.get("RUNBOOK_ACTIONS", [""])[0]),
        )
        self.assertEqual(str(rec.get("NEXT_TARGET_LAYER", "")), "regime_filter")
        self.assertEqual(str(rec.get("NEXT_TARGET_SUBCONDITION", "")), "volatility_regime_ok")
        self.assertIsInstance(rec.get("NEXT_TARGET_SUBCONDITION"), str)
        self.assertNotIn(",", str(rec.get("NEXT_TARGET_SUBCONDITION", "")))

    def test_calibration_blocks_co_dominant_regime_blockers(self):
        before_summary = {
            "evaluations_total": 80,
            "regime_filter_pass_rate": 0.0,
            "regime_filter_fail_count": 80,
            "regime_filter_htf_trend_blocker_count": 8,
            "regime_filter_vwap_stretch_blocker_count": 48,
            "regime_filter_volatility_blocker_count": 56,
            "regime_filter_news_blocker_count": 0,
            "top_regime_filter_blocker": "volatility_regime_ok",
            "top_blocker_by_raw_coverage": "volatility_regime_ok",
            "top_blocker_by_label": "volatility_regime_ok",
            "co_dominant_blockers": ["volatility_regime_ok"],
            "blocker_dominance_share": {"regime_filter": 56 / 112},
            "blocker_rate_per_regime_fail": {
                "htf_trend_ok": 8 / 80,
                "stretched_from_vwap": 48 / 80,
                "volatility_regime_ok": 56 / 80,
                "news_veto": 0.0,
            },
            "layer1_fail_count": 0,
            "layer2_fail_count": 0,
            "layer3_fail_count": 0,
            "layer4_fail_count": 0,
            "layer5_fail_count": 0,
        }
        after_summary = {
            "evaluations_total": 82,
            "regime_filter_pass_rate": 0.0,
            "regime_filter_fail_count": 82,
            "regime_filter_htf_trend_blocker_count": 6,
            "regime_filter_vwap_stretch_blocker_count": 58,
            "regime_filter_volatility_blocker_count": 64,
            "regime_filter_news_blocker_count": 0,
            "top_regime_filter_blocker": "volatility_regime_ok",
            "top_blocker_by_raw_coverage": "volatility_regime_ok",
            "top_blocker_by_label": "volatility_regime_ok",
            "co_dominant_blockers": ["volatility_regime_ok", "stretched_from_vwap"],
            "top_regime_filter_blocker_combination": "stretched_from_vwap + volatility_regime_ok",
            "top_regime_filter_blocker_combination_share": 0.70,
            "blocker_dominance_share": {"regime_filter": 64 / 128},
            "blocker_rate_per_regime_fail": {
                "htf_trend_ok": 6 / 82,
                "stretched_from_vwap": 58 / 82,
                "volatility_regime_ok": 64 / 82,
                "news_veto": 0.0,
            },
            "layer1_fail_count": 0,
            "layer2_fail_count": 0,
            "layer3_fail_count": 0,
            "layer4_fail_count": 0,
            "layer5_fail_count": 0,
        }

        rec = recommend_calibration_step(after_summary=after_summary, before_summary=before_summary)
        self.assertFalse(bool(rec.get("SAFE_TO_CONTINUE")))
        self.assertEqual(str(rec.get("ACTION_VERDICT", "")), "co_dominant_overlap")
        self.assertEqual(str(rec.get("STOP_REASON", "")), "co_dominant_regime_blockers")
        self.assertIn("co-dominant", str(rec.get("WHY_NOT_OTHERS", [""])[0]).lower())
        self.assertIn("stretched_from_vwap + volatility_regime_ok", str(rec.get("WHY_NOT_OTHERS", [""])[0]))
        self.assertIn("Pause threshold changes for this branch.", rec.get("RUNBOOK_ACTIONS", []))
        self.assertIn(
            "Review blocker combination 'stretched_from_vwap + volatility_regime_ok' as the current overlap driver.",
            rec.get("RUNBOOK_ACTIONS", []),
        )

    def test_calibration_blocks_when_raw_and_label_semantics_disagree(self):
        before_summary = {
            "evaluations_total": 80,
            "regime_filter_pass_rate": 0.0,
            "regime_filter_fail_count": 80,
            "regime_filter_htf_trend_blocker_count": 10,
            "regime_filter_vwap_stretch_blocker_count": 18,
            "regime_filter_volatility_blocker_count": 52,
            "regime_filter_news_blocker_count": 0,
            "top_regime_filter_blocker": "volatility_regime_ok",
            "top_blocker_by_raw_coverage": "volatility_regime_ok",
            "top_blocker_by_label": "volatility_regime_ok",
            "co_dominant_blockers": ["volatility_regime_ok"],
            "blocker_dominance_share": {"regime_filter": 52 / 80},
            "blocker_rate_per_regime_fail": {
                "htf_trend_ok": 10 / 80,
                "stretched_from_vwap": 18 / 80,
                "volatility_regime_ok": 52 / 80,
                "news_veto": 0.0,
            },
            "layer1_fail_count": 0,
            "layer2_fail_count": 0,
            "layer3_fail_count": 0,
            "layer4_fail_count": 0,
            "layer5_fail_count": 0,
        }
        after_summary = {
            "evaluations_total": 84,
            "regime_filter_pass_rate": 0.0,
            "regime_filter_fail_count": 84,
            "regime_filter_htf_trend_blocker_count": 9,
            "regime_filter_vwap_stretch_blocker_count": 21,
            "regime_filter_volatility_blocker_count": 54,
            "regime_filter_news_blocker_count": 0,
            "top_regime_filter_blocker": "volatility_regime_ok",
            "top_blocker_by_raw_coverage": "volatility_regime_ok",
            "top_blocker_by_label": "stretched_from_vwap",
            "co_dominant_blockers": ["volatility_regime_ok"],
            "blocker_dominance_share": {"regime_filter": 54 / 84},
            "blocker_rate_per_regime_fail": {
                "htf_trend_ok": 9 / 84,
                "stretched_from_vwap": 21 / 84,
                "volatility_regime_ok": 54 / 84,
                "news_veto": 0.0,
            },
            "layer1_fail_count": 0,
            "layer2_fail_count": 0,
            "layer3_fail_count": 0,
            "layer4_fail_count": 0,
            "layer5_fail_count": 0,
        }

        rec = recommend_calibration_step(after_summary=after_summary, before_summary=before_summary)
        self.assertFalse(bool(rec.get("SAFE_TO_CONTINUE")))
        self.assertEqual(str(rec.get("ACTION_VERDICT", "")), "co_dominant_overlap")
        self.assertEqual(str(rec.get("STOP_REASON", "")), "blocker_semantics_disagreement")
        self.assertIn("snapshot labels favor", str(rec.get("WHY_NOT_OTHERS", [""])[0]).lower())
        self.assertIn("Pause threshold changes for this branch.", rec.get("RUNBOOK_ACTIONS", []))
        self.assertIn("Inspect raw coverage and snapshot label semantics before choosing a target.", rec.get("RUNBOOK_ACTIONS", []))

    def test_calibration_unstable_blocker_requires_review(self):
        before_summary = {
            "evaluations_total": 90,
            "regime_filter_pass_rate": 0.0,
            "regime_filter_fail_count": 90,
            "regime_filter_htf_trend_blocker_count": 9,
            "regime_filter_vwap_stretch_blocker_count": 72,
            "regime_filter_volatility_blocker_count": 9,
            "regime_filter_news_blocker_count": 0,
            "top_regime_filter_blocker": "stretched_from_vwap",
            "blocker_dominance_share": {"regime_filter": 0.8},
            "blocker_rate_per_regime_fail": {
                "htf_trend_ok": 0.10,
                "stretched_from_vwap": 0.80,
                "volatility_regime_ok": 0.10,
                "news_veto": 0.0,
            },
            "layer1_fail_count": 0,
            "layer2_fail_count": 0,
            "layer3_fail_count": 0,
            "layer4_fail_count": 0,
            "layer5_fail_count": 0,
        }
        after_summary = {
            "evaluations_total": 92,
            "regime_filter_pass_rate": 0.0,
            "regime_filter_fail_count": 92,
            "regime_filter_htf_trend_blocker_count": 9,
            "regime_filter_vwap_stretch_blocker_count": 0,
            "regime_filter_volatility_blocker_count": 83,
            "regime_filter_news_blocker_count": 0,
            "top_regime_filter_blocker": "volatility_regime_ok",
            "blocker_dominance_share": {"regime_filter": 83 / 92},
            "blocker_rate_per_regime_fail": {
                "htf_trend_ok": 9 / 92,
                "stretched_from_vwap": 0.0,
                "volatility_regime_ok": 83 / 92,
                "news_veto": 0.0,
            },
            "layer1_fail_count": 0,
            "layer2_fail_count": 0,
            "layer3_fail_count": 0,
            "layer4_fail_count": 0,
            "layer5_fail_count": 0,
        }

        rec = recommend_calibration_step(after_summary=after_summary, before_summary=before_summary)
        self.assertFalse(bool(rec.get("SAFE_TO_CONTINUE")))
        self.assertEqual(str(rec.get("STOP_REASON", "")), "market_context_shift_detected")

    def test_calibration_comparison_keeps_backward_compatible_core_fields(self):
        before_summary = {
            "samples": 10,
            "evaluations_total": 10,
            "short_signal_count": 1,
            "no_signal_count": 9,
            "short_signal_ratio": 0.1,
            "no_signal_ratio": 0.9,
            "regime_filter_pass_count": 2,
            "regime_filter_fail_count": 8,
            "regime_filter_pass_rate": 0.2,
            "regime_filter_fail_rate": 0.8,
            "reached_layer1_count": 2,
            "layer1_reach_rate": 0.2,
            "layer1_fail_count": 1,
            "layer1_fail_rate_given_reach": 0.5,
            "regime_filter_htf_trend_blocker_count": 4,
            "regime_filter_vwap_stretch_blocker_count": 2,
            "regime_filter_volatility_blocker_count": 2,
            "regime_filter_news_blocker_count": 0,
            "layer1_rsi_high_blocker_count": 1,
            "layer1_volume_spike_blocker_count": 0,
            "layer1_above_bollinger_upper_blocker_count": 0,
            "layer1_above_keltner_upper_blocker_count": 0,
            "layer4_fail_count": 0,
            "layer5_fail_count": 0,
            "top_failed_layer": "regime_filter",
            "top_regime_filter_blocker": "htf_trend_ok",
            "top_regime_filter_blocker_combination": "htf_trend_ok + stretched_from_vwap",
            "top_regime_filter_blocker_combination_share": 0.4,
            "top_layer1_blocker": "rsi_high",
            "top_layer4_blocker": "price_above_vwap",
        }
        after_summary = dict(before_summary)
        after_summary["samples"] = 12
        after_summary["evaluations_total"] = 12
        after_summary["regime_filter_pass_count"] = 3
        after_summary["regime_filter_fail_count"] = 9
        after_summary["top_regime_filter_blocker_combination"] = "htf_trend_ok + stretched_from_vwap + volatility_regime_ok"
        after_summary["top_regime_filter_blocker_combination_share"] = 0.5

        delta = compare_observation_windows(before_summary=before_summary, after_summary=after_summary)
        self.assertIn("regime_filter_pass_count", delta)
        self.assertIn("regime_filter_fail_count", delta)
        self.assertIn("short_signal_ratio", delta)
        self.assertIn("top_regime_filter_blocker_before", delta)
        self.assertIn("top_regime_filter_blocker_after", delta)
        self.assertEqual(
            str(delta.get("top_regime_filter_blocker_combination_before", "")),
            "htf_trend_ok + stretched_from_vwap",
        )
        self.assertEqual(
            str(delta.get("top_regime_filter_blocker_combination_after", "")),
            "htf_trend_ok + stretched_from_vwap + volatility_regime_ok",
        )

    def test_calibration_aggregate_tracks_raw_label_and_co_dominance_fields(self):
        records = [
            {
                "ts": "2026-03-13T00:00:00+00:00",
                "compact": {
                    "evaluations_total": 1,
                    "regime_filter_fail_count": 1,
                    "no_signal_count": 1,
                    "regime_filter_htf_trend_blocker_count": 0,
                    "regime_filter_vwap_stretch_blocker_count": 1,
                    "regime_filter_volatility_blocker_count": 1,
                    "regime_filter_news_blocker_count": 0,
                    "top_regime_filter_blocker": "stretched_from_vwap",
                },
                "audit": {"failed_layer_counts": {"regime_filter": 1}},
            },
            {
                "ts": "2026-03-13T00:01:00+00:00",
                "compact": {
                    "evaluations_total": 1,
                    "regime_filter_fail_count": 1,
                    "no_signal_count": 1,
                    "regime_filter_htf_trend_blocker_count": 0,
                    "regime_filter_vwap_stretch_blocker_count": 1,
                    "regime_filter_volatility_blocker_count": 1,
                    "regime_filter_news_blocker_count": 0,
                    "top_regime_filter_blocker": "volatility_regime_ok",
                },
                "audit": {"failed_layer_counts": {"regime_filter": 1}},
            },
        ]

        summary = aggregate_observation(records)
        self.assertEqual(str(summary.get("top_blocker_by_raw_coverage", "")), "stretched_from_vwap")
        self.assertEqual(str(summary.get("top_blocker_by_label", "")), "stretched_from_vwap")
        self.assertEqual(int(summary.get("top_blocker_by_raw_coverage_count", 0)), 2)
        self.assertEqual(int(summary.get("top_blocker_by_label_count", 0)), 1)
        self.assertAlmostEqual(float(summary.get("top_blocker_by_raw_coverage_share", 0.0)), 1.0, places=6)
        self.assertAlmostEqual(float(summary.get("top_blocker_by_label_share", 0.0)), 0.5, places=6)
        self.assertEqual(list(summary.get("co_dominant_blockers", [])), ["stretched_from_vwap", "volatility_regime_ok"])
        self.assertEqual(str(summary.get("dominance_mode", "")), "co_dominant_raw")
        self.assertIn("co-dominant", str(summary.get("dominance_explanation", "")).lower())
        self.assertEqual(int(summary.get("regime_filter_failed_sample_count", 0)), 2)
        self.assertEqual(
            dict(summary.get("regime_filter_blocker_combination_counts", {})),
            {"stretched_from_vwap + volatility_regime_ok": 2},
        )
        self.assertEqual(
            str(summary.get("top_regime_filter_blocker_combination", "")),
            "stretched_from_vwap + volatility_regime_ok",
        )
        self.assertAlmostEqual(float(summary.get("top_regime_filter_blocker_combination_share", 0.0)), 1.0, places=6)

    def test_calibration_requires_previous_validated_window(self):
        after_summary = {
            "evaluations_total": 80,
            "regime_filter_pass_rate": 0.0,
            "regime_filter_fail_count": 80,
            "regime_filter_htf_trend_blocker_count": 20,
            "regime_filter_vwap_stretch_blocker_count": 15,
            "regime_filter_volatility_blocker_count": 45,
            "regime_filter_news_blocker_count": 0,
            "blocker_dominance_share": {"regime_filter": 45 / 80},
            "top_regime_filter_blocker": "volatility_regime_ok",
            "layer1_fail_count": 0,
            "layer2_fail_count": 0,
            "layer3_fail_count": 0,
            "layer4_fail_count": 0,
            "layer5_fail_count": 0,
        }
        rec = recommend_calibration_step(after_summary=after_summary, before_summary=None)
        self.assertFalse(bool(rec.get("SAFE_TO_CONTINUE")))
        self.assertEqual(str(rec.get("ACTION_VERDICT", "")), "pause_calibration")
        self.assertEqual(str(rec.get("STOP_REASON", "")), "previous_validated_window_missing")
        self.assertIn("Locate the most recent validated BEFORE window for this calibration branch.", rec.get("RUNBOOK_ACTIONS", []))

    def _run_triage_calibration_result(self, payload: dict) -> subprocess.CompletedProcess[str]:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        python_exe = os.path.join(repo_root, ".venv", "Scripts", "python.exe")
        tool_path = os.path.join(repo_root, "scripts", "observation", "triage_calibration_result.py")

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "calibration_result.json")
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)

            return subprocess.run(
                [python_exe, tool_path, json_path],
                cwd=repo_root,
                capture_output=True,
                text=True,
            )

    def _run_triage_calibration_result_ps1(self, payload: dict) -> subprocess.CompletedProcess[str]:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        ps_exe = r"C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe"
        script_path = os.path.join(repo_root, "scripts", "observation", "triage_calibration_result.ps1")

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "calibration_result.json")
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)

            return subprocess.run(
                [
                    ps_exe,
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    script_path,
                    "-ComparisonJson",
                    json_path,
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
            )

    def _run_inspect_regime_overlap(self, after_lines: list[dict], before_lines: list[dict] | None = None) -> subprocess.CompletedProcess[str]:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        python_exe = os.path.join(repo_root, ".venv", "Scripts", "python.exe")
        tool_path = os.path.join(repo_root, "scripts", "observation", "inspect_regime_overlap.py")

        with tempfile.TemporaryDirectory() as tmpdir:
            after_path = os.path.join(tmpdir, "after.log")
            with open(after_path, "w", encoding="utf-8") as fh:
                for line in after_lines:
                    fh.write(json.dumps(line, ensure_ascii=False) + "\n")

            args = [python_exe, tool_path, "--after", after_path]
            if before_lines is not None:
                before_path = os.path.join(tmpdir, "before.log")
                with open(before_path, "w", encoding="utf-8") as fh:
                    for line in before_lines:
                        fh.write(json.dumps(line, ensure_ascii=False) + "\n")
                args.extend(["--before", before_path])

            return subprocess.run(
                args,
                cwd=repo_root,
                capture_output=True,
                text=True,
            )

    def _run_inspect_vwap_semantics(self, after_lines: list[dict], before_lines: list[dict] | None = None) -> subprocess.CompletedProcess[str]:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        python_exe = os.path.join(repo_root, ".venv", "Scripts", "python.exe")
        tool_path = os.path.join(repo_root, "scripts", "observation", "inspect_vwap_semantics.py")

        with tempfile.TemporaryDirectory() as tmpdir:
            after_path = os.path.join(tmpdir, "after.log")
            with open(after_path, "w", encoding="utf-8") as fh:
                for line in after_lines:
                    fh.write(json.dumps(line, ensure_ascii=False) + "\n")

            args = [python_exe, tool_path, "--after", after_path]
            if before_lines is not None:
                before_path = os.path.join(tmpdir, "before.log")
                with open(before_path, "w", encoding="utf-8") as fh:
                    for line in before_lines:
                        fh.write(json.dumps(line, ensure_ascii=False) + "\n")
                args.extend(["--before", before_path])

            return subprocess.run(
                args,
                cwd=repo_root,
                capture_output=True,
                text=True,
            )

    def test_triage_calibration_result_formats_overlap_output(self):
        payload = {
            "after": {
                "summary": {
                    "top_regime_filter_blocker_combination": "stretched_from_vwap + volatility_regime_ok",
                }
            },
            "calibration_recommendation": {
                "SAFE_TO_CONTINUE": False,
                "ACTION_VERDICT": "co_dominant_overlap",
                "STOP_REASON": "co_dominant_regime_blockers",
                "RUNBOOK_ACTIONS": [
                    "Pause threshold changes for this branch.",
                    "Review blocker combination 'stretched_from_vwap + volatility_regime_ok' as the current overlap driver.",
                    "Collect another comparable window before attempting a single-threshold tweak.",
                    "Do not tune one regime threshold while blocker overlap remains co-dominant.",
                ],
            },
        }

        proc = self._run_triage_calibration_result(payload)
        self.assertEqual(proc.returncode, 10, msg=proc.stderr)
        self.assertEqual(
            proc.stdout.strip().splitlines(),
            [
                "VERDICT: co_dominant_overlap",
                "STOP_REASON: co_dominant_regime_blockers",
                "TOP_COMBINATION: stretched_from_vwap + volatility_regime_ok",
                "ACTION 1: Pause threshold changes for this branch.",
                "ACTION 2: Review blocker combination 'stretched_from_vwap + volatility_regime_ok' as the current overlap driver.",
                "ACTION 3: Collect another comparable window before attempting a single-threshold tweak.",
                "ACTION 4: Do not tune one regime threshold while blocker overlap remains co-dominant.",
            ],
        )

    def test_triage_calibration_result_ps1_formats_overlap_output(self):
        payload = {
            "after": {
                "summary": {
                    "top_regime_filter_blocker_combination": "stretched_from_vwap + volatility_regime_ok",
                }
            },
            "calibration_recommendation": {
                "SAFE_TO_CONTINUE": False,
                "ACTION_VERDICT": "co_dominant_overlap",
                "STOP_REASON": "co_dominant_regime_blockers",
                "RUNBOOK_ACTIONS": [
                    "Pause threshold changes for this branch.",
                    "Review blocker combination 'stretched_from_vwap + volatility_regime_ok' as the current overlap driver.",
                    "Collect another comparable window before attempting a single-threshold tweak.",
                    "Do not tune one regime threshold while blocker overlap remains co-dominant.",
                ],
            },
        }

        proc = self._run_triage_calibration_result_ps1(payload)
        self.assertEqual(proc.returncode, 10, msg=proc.stderr)
        self.assertEqual(
            proc.stdout.strip().splitlines(),
            [
                "VERDICT: co_dominant_overlap",
                "STOP_REASON: co_dominant_regime_blockers",
                "TOP_COMBINATION: stretched_from_vwap + volatility_regime_ok",
                "ACTION 1: Pause threshold changes for this branch.",
                "ACTION 2: Review blocker combination 'stretched_from_vwap + volatility_regime_ok' as the current overlap driver.",
                "ACTION 3: Collect another comparable window before attempting a single-threshold tweak.",
                "ACTION 4: Do not tune one regime threshold while blocker overlap remains co-dominant.",
            ],
        )

    def test_inspect_regime_overlap_buckets_and_margins(self):
        after_lines = [
            {
                "ts": "2026-03-16T00:00:00+00:00",
                "msg": (
                    "diag htf_trend_metric_used=35 htf_trend_threshold_used=30 "
                    "vwap_distance_metric_used=-0.00190 vwap_stretch_threshold_used=0.00125 "
                    "atr_norm=0.00044 volatility_threshold_used=0.0006 "
                    "strategy_audit_compact={'regime_filter_fail_count':1,'regime_filter_htf_trend_blocker_count':0,"
                    "'regime_filter_vwap_stretch_blocker_count':1,'regime_filter_volatility_blocker_count':1,"
                    "'regime_filter_news_blocker_count':0,'top_regime_filter_blocker':'stretched_from_vwap'}"
                ),
            },
            {
                "ts": "2026-03-16T00:01:00+00:00",
                "msg": (
                    "diag htf_trend_metric_used=36 htf_trend_threshold_used=30 "
                    "vwap_distance_metric_used=-0.00185 vwap_stretch_threshold_used=0.00125 "
                    "atr_norm=0.00046 volatility_threshold_used=0.0006 "
                    "strategy_audit_compact={'regime_filter_fail_count':1,'regime_filter_htf_trend_blocker_count':0,"
                    "'regime_filter_vwap_stretch_blocker_count':1,'regime_filter_volatility_blocker_count':1,"
                    "'regime_filter_news_blocker_count':0,'top_regime_filter_blocker':'stretched_from_vwap'}"
                ),
            },
            {
                "ts": "2026-03-16T00:02:00+00:00",
                "msg": (
                    "diag htf_trend_metric_used=37 htf_trend_threshold_used=30 "
                    "vwap_distance_metric_used=-0.00170 vwap_stretch_threshold_used=0.00125 "
                    "atr_norm=0.00070 volatility_threshold_used=0.0006 "
                    "strategy_audit_compact={'regime_filter_fail_count':1,'regime_filter_htf_trend_blocker_count':0,"
                    "'regime_filter_vwap_stretch_blocker_count':1,'regime_filter_volatility_blocker_count':0,"
                    "'regime_filter_news_blocker_count':0,'top_regime_filter_blocker':'stretched_from_vwap'}"
                ),
            },
            {
                "ts": "2026-03-16T00:03:00+00:00",
                "msg": (
                    "diag htf_trend_metric_used=25 htf_trend_threshold_used=30 "
                    "vwap_distance_metric_used=0.00010 vwap_stretch_threshold_used=0.00125 "
                    "atr_norm=0.00055 volatility_threshold_used=0.0006 "
                    "strategy_audit_compact={'regime_filter_fail_count':1,'regime_filter_htf_trend_blocker_count':0,"
                    "'regime_filter_vwap_stretch_blocker_count':0,'regime_filter_volatility_blocker_count':1,"
                    "'regime_filter_news_blocker_count':0,'top_regime_filter_blocker':'volatility_regime_ok'}"
                ),
            },
            {
                "ts": "2026-03-16T00:04:00+00:00",
                "msg": (
                    "diag htf_trend_metric_used=20 htf_trend_threshold_used=30 "
                    "vwap_distance_metric_used=-0.00195 vwap_stretch_threshold_used=0.00125 "
                    "atr_norm=0.00043 volatility_threshold_used=0.0006 "
                    "strategy_audit_compact={'regime_filter_fail_count':1,'regime_filter_htf_trend_blocker_count':1,"
                    "'regime_filter_vwap_stretch_blocker_count':1,'regime_filter_volatility_blocker_count':1,"
                    "'regime_filter_news_blocker_count':0,'top_regime_filter_blocker':'htf_trend_ok'}"
                ),
            },
        ]

        proc = self._run_inspect_regime_overlap(after_lines)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)

        payload = json.loads(proc.stdout)
        after = payload.get("after", {})
        self.assertEqual(int(after.get("failed_sample_count", 0)), 5)
        self.assertEqual(
            dict(after.get("bucket_counts", {})),
            {
                "vwap + volatility": 2,
                "htf + vwap + volatility": 1,
                "volatility_only": 1,
                "vwap_only": 1,
            },
        )

        vwap_vol_bucket = after.get("buckets", {}).get("vwap + volatility", {})
        self.assertEqual(int(vwap_vol_bucket.get("count", 0)), 2)
        self.assertAlmostEqual(float(vwap_vol_bucket.get("share_of_failed_samples", 0.0)), 0.4, places=8)
        self.assertEqual(str(vwap_vol_bucket.get("top_label_blocker", "")), "stretched_from_vwap")
        self.assertEqual(str(vwap_vol_bucket.get("top_raw_blocker", "")), "stretched_from_vwap")
        self.assertAlmostEqual(
            float(vwap_vol_bucket.get("margins", {}).get("vwap_metric_minus_threshold", {}).get("mean", 99.0)),
            -0.003125,
            places=8,
        )
        self.assertAlmostEqual(
            float(vwap_vol_bucket.get("margins", {}).get("volatility_metric_minus_threshold", {}).get("p50", 99.0)),
            -0.00015,
            places=8,
        )

        htf_bucket = after.get("buckets", {}).get("htf + vwap + volatility", {})
        self.assertEqual(str(htf_bucket.get("top_label_blocker", "")), "htf_trend_ok")
        self.assertEqual(str(htf_bucket.get("top_raw_blocker", "")), "htf_trend_ok")
        self.assertAlmostEqual(
            float(htf_bucket.get("margins", {}).get("htf_metric_minus_threshold", {}).get("mean", 99.0)),
            -10.0,
            places=8,
        )

    def test_inspect_regime_overlap_comparison_reports_bucket_deltas(self):
        before_lines = [
            {
                "ts": "2026-03-16T01:00:00+00:00",
                "msg": (
                    "diag htf_trend_metric_used=35 htf_trend_threshold_used=30 "
                    "vwap_distance_metric_used=-0.00190 vwap_stretch_threshold_used=0.00125 "
                    "atr_norm=0.00044 volatility_threshold_used=0.0006 "
                    "strategy_audit_compact={'regime_filter_fail_count':1,'regime_filter_htf_trend_blocker_count':0,"
                    "'regime_filter_vwap_stretch_blocker_count':1,'regime_filter_volatility_blocker_count':1,"
                    "'regime_filter_news_blocker_count':0,'top_regime_filter_blocker':'stretched_from_vwap'}"
                ),
            },
            {
                "ts": "2026-03-16T01:01:00+00:00",
                "msg": (
                    "diag htf_trend_metric_used=34 htf_trend_threshold_used=30 "
                    "vwap_distance_metric_used=-0.00195 vwap_stretch_threshold_used=0.00125 "
                    "atr_norm=0.00045 volatility_threshold_used=0.0006 "
                    "strategy_audit_compact={'regime_filter_fail_count':1,'regime_filter_htf_trend_blocker_count':0,"
                    "'regime_filter_vwap_stretch_blocker_count':1,'regime_filter_volatility_blocker_count':1,"
                    "'regime_filter_news_blocker_count':0,'top_regime_filter_blocker':'stretched_from_vwap'}"
                ),
            },
            {
                "ts": "2026-03-16T01:02:00+00:00",
                "msg": (
                    "diag htf_trend_metric_used=37 htf_trend_threshold_used=30 "
                    "vwap_distance_metric_used=-0.00170 vwap_stretch_threshold_used=0.00125 "
                    "atr_norm=0.00070 volatility_threshold_used=0.0006 "
                    "strategy_audit_compact={'regime_filter_fail_count':1,'regime_filter_htf_trend_blocker_count':0,"
                    "'regime_filter_vwap_stretch_blocker_count':1,'regime_filter_volatility_blocker_count':0,"
                    "'regime_filter_news_blocker_count':0,'top_regime_filter_blocker':'stretched_from_vwap'}"
                ),
            },
        ]
        after_lines = [
            {
                "ts": "2026-03-16T02:00:00+00:00",
                "msg": (
                    "diag htf_trend_metric_used=20 htf_trend_threshold_used=30 "
                    "vwap_distance_metric_used=-0.00195 vwap_stretch_threshold_used=0.00125 "
                    "atr_norm=0.00043 volatility_threshold_used=0.0006 "
                    "strategy_audit_compact={'regime_filter_fail_count':1,'regime_filter_htf_trend_blocker_count':1,"
                    "'regime_filter_vwap_stretch_blocker_count':1,'regime_filter_volatility_blocker_count':1,"
                    "'regime_filter_news_blocker_count':0,'top_regime_filter_blocker':'htf_trend_ok'}"
                ),
            },
            {
                "ts": "2026-03-16T02:01:00+00:00",
                "msg": (
                    "diag htf_trend_metric_used=21 htf_trend_threshold_used=30 "
                    "vwap_distance_metric_used=-0.00192 vwap_stretch_threshold_used=0.00125 "
                    "atr_norm=0.00044 volatility_threshold_used=0.0006 "
                    "strategy_audit_compact={'regime_filter_fail_count':1,'regime_filter_htf_trend_blocker_count':1,"
                    "'regime_filter_vwap_stretch_blocker_count':1,'regime_filter_volatility_blocker_count':1,"
                    "'regime_filter_news_blocker_count':0,'top_regime_filter_blocker':'htf_trend_ok'}"
                ),
            },
            {
                "ts": "2026-03-16T02:02:00+00:00",
                "msg": (
                    "diag htf_trend_metric_used=35 htf_trend_threshold_used=30 "
                    "vwap_distance_metric_used=-0.00188 vwap_stretch_threshold_used=0.00125 "
                    "atr_norm=0.00044 volatility_threshold_used=0.0006 "
                    "strategy_audit_compact={'regime_filter_fail_count':1,'regime_filter_htf_trend_blocker_count':0,"
                    "'regime_filter_vwap_stretch_blocker_count':1,'regime_filter_volatility_blocker_count':1,"
                    "'regime_filter_news_blocker_count':0,'top_regime_filter_blocker':'stretched_from_vwap'}"
                ),
            },
        ]

        proc = self._run_inspect_regime_overlap(after_lines, before_lines=before_lines)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)

        payload = json.loads(proc.stdout)
        comparison = payload.get("comparison", {})
        self.assertEqual(str(comparison.get("top_bucket_before", "")), "vwap + volatility")
        self.assertEqual(str(comparison.get("top_bucket_after", "")), "htf + vwap + volatility")
        self.assertEqual(int(comparison.get("bucket_count_delta", {}).get("htf + vwap + volatility", 0)), 2)
        self.assertEqual(int(comparison.get("bucket_count_delta", {}).get("vwap_only", 0)), -1)
        self.assertAlmostEqual(
            float(comparison.get("bucket_share_delta", {}).get("htf + vwap + volatility", 99.0)),
            2.0 / 3.0,
            places=8,
        )

    def test_inspect_vwap_semantics_focus_buckets_and_context(self):
        after_lines = [
            {
                "ts": "2026-03-16T03:00:00+00:00",
                "msg": (
                    "diag htf_trend_metric_used=35 htf_trend_threshold_used=30 "
                    "htf_trend_direction_context='not_strong_uptrend' "
                    "vwap_distance_metric_used=-0.00190 vwap_stretch_threshold_used=0.00125 "
                    "atr_norm=0.00044 volatility_threshold_used=0.0006 "
                    "strategy_audit_compact={'regime_filter_fail_count':1,'regime_filter_htf_trend_blocker_count':0,"
                    "'regime_filter_vwap_stretch_blocker_count':1,'regime_filter_volatility_blocker_count':1,"
                    "'regime_filter_news_blocker_count':0,'regime_filter_degraded_mode_count':1,"
                    "'regime_filter_soft_pass_candidate_count':0,'top_regime_filter_blocker':'stretched_from_vwap'} "
                    "strategy_audit_regime_filter={'failed_reason':'missing:stretched_from_vwap,volatility_regime_ok',"
                    "'missing_conditions':'stretched_from_vwap,volatility_regime_ok','degraded_mode':1.0,"
                    "'source_flags':{'vwap_quality':'live','news_quality':'unavailable'},"
                    "'regime_filter_subconditions_state':{'htf_trend_ok':True,'stretched_from_vwap':False,"
                    "'volatility_regime_ok':False,'news_veto':True}} "
                    "strategy_audit={'source_quality_counts':{'regime_filter':{'vwap_quality':{'live':1},'news_quality':{'unavailable':1}}}}"
                ),
            },
            {
                "ts": "2026-03-16T03:01:00+00:00",
                "msg": (
                    "diag htf_trend_metric_used=36 htf_trend_threshold_used=30 "
                    "htf_trend_direction_context='not_strong_uptrend' "
                    "vwap_distance_metric_used=-0.00185 vwap_stretch_threshold_used=0.00125 "
                    "atr_norm=0.00046 volatility_threshold_used=0.0006 "
                    "strategy_audit_compact={'regime_filter_fail_count':1,'regime_filter_htf_trend_blocker_count':0,"
                    "'regime_filter_vwap_stretch_blocker_count':1,'regime_filter_volatility_blocker_count':1,"
                    "'regime_filter_news_blocker_count':0,'regime_filter_degraded_mode_count':0,"
                    "'regime_filter_soft_pass_candidate_count':0,'top_regime_filter_blocker':'stretched_from_vwap'} "
                    "strategy_audit_regime_filter={'failed_reason':'missing:stretched_from_vwap,volatility_regime_ok',"
                    "'missing_conditions':'stretched_from_vwap,volatility_regime_ok','degraded_mode':0.0,"
                    "'source_flags':{'vwap_quality':'live','news_quality':'live'},"
                    "'regime_filter_subconditions_state':{'htf_trend_ok':True,'stretched_from_vwap':False,"
                    "'volatility_regime_ok':False,'news_veto':True}} "
                    "strategy_audit={'source_quality_counts':{'regime_filter':{'vwap_quality':{'live':1},'news_quality':{'live':1}}}}"
                ),
            },
            {
                "ts": "2026-03-16T03:02:00+00:00",
                "msg": (
                    "diag htf_trend_metric_used=37 htf_trend_threshold_used=30 "
                    "htf_trend_direction_context='strong_uptrend' "
                    "vwap_distance_metric_used=0.00090 vwap_stretch_threshold_used=0.00125 "
                    "atr_norm=0.00070 volatility_threshold_used=0.0006 "
                    "strategy_audit_compact={'regime_filter_fail_count':1,'regime_filter_htf_trend_blocker_count':0,"
                    "'regime_filter_vwap_stretch_blocker_count':1,'regime_filter_volatility_blocker_count':0,"
                    "'regime_filter_news_blocker_count':0,'regime_filter_degraded_mode_count':0,"
                    "'regime_filter_soft_pass_candidate_count':1,'top_regime_filter_blocker':'stretched_from_vwap'} "
                    "strategy_audit_regime_filter={'failed_reason':'missing:stretched_from_vwap',"
                    "'missing_conditions':'stretched_from_vwap','degraded_mode':0.0,"
                    "'source_flags':{'vwap_quality':'live','news_quality':'live'},"
                    "'regime_filter_subconditions_state':{'htf_trend_ok':True,'stretched_from_vwap':False,"
                    "'volatility_regime_ok':True,'news_veto':True}} "
                    "strategy_audit={'source_quality_counts':{'regime_filter':{'vwap_quality':{'live':1},'news_quality':{'live':1}}}}"
                ),
            },
            {
                "ts": "2026-03-16T03:03:00+00:00",
                "msg": (
                    "diag htf_trend_metric_used=20 htf_trend_threshold_used=30 "
                    "htf_trend_direction_context='strong_uptrend' "
                    "vwap_distance_metric_used=-0.00195 vwap_stretch_threshold_used=0.00125 "
                    "atr_norm=0.00043 volatility_threshold_used=0.0006 "
                    "strategy_audit_compact={'regime_filter_fail_count':1,'regime_filter_htf_trend_blocker_count':1,"
                    "'regime_filter_vwap_stretch_blocker_count':1,'regime_filter_volatility_blocker_count':1,"
                    "'regime_filter_news_blocker_count':0,'regime_filter_degraded_mode_count':0,"
                    "'regime_filter_soft_pass_candidate_count':0,'top_regime_filter_blocker':'htf_trend_ok'} "
                    "strategy_audit_regime_filter={'failed_reason':'missing:htf_trend_ok,stretched_from_vwap,volatility_regime_ok',"
                    "'missing_conditions':'htf_trend_ok,stretched_from_vwap,volatility_regime_ok','degraded_mode':1.0,"
                    "'source_flags':{'vwap_quality':'fallback','news_quality':'unavailable'},"
                    "'regime_filter_subconditions_state':{'htf_trend_ok':False,'stretched_from_vwap':False,"
                    "'volatility_regime_ok':False,'news_veto':True}} "
                    "strategy_audit={'source_quality_counts':{'regime_filter':{'vwap_quality':{'fallback':1},'news_quality':{'unavailable':1}}}}"
                ),
            },
            {
                "ts": "2026-03-16T03:04:00+00:00",
                "msg": (
                    "diag htf_trend_metric_used=25 htf_trend_threshold_used=30 "
                    "htf_trend_direction_context='not_strong_uptrend' "
                    "vwap_distance_metric_used=0.00010 vwap_stretch_threshold_used=0.00125 "
                    "atr_norm=0.00055 volatility_threshold_used=0.0006 "
                    "strategy_audit_compact={'regime_filter_fail_count':1,'regime_filter_htf_trend_blocker_count':0,"
                    "'regime_filter_vwap_stretch_blocker_count':0,'regime_filter_volatility_blocker_count':1,"
                    "'regime_filter_news_blocker_count':0,'regime_filter_degraded_mode_count':0,"
                    "'regime_filter_soft_pass_candidate_count':0,'top_regime_filter_blocker':'volatility_regime_ok'} "
                    "strategy_audit={'source_quality_counts':{'regime_filter':{'vwap_quality':{'live':1},'news_quality':{'unavailable':1}}}}"
                ),
            },
        ]

        proc = self._run_inspect_vwap_semantics(after_lines)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)

        payload = json.loads(proc.stdout)
        after = payload.get("after", {})
        self.assertEqual(int(after.get("failed_sample_count", 0)), 5)
        self.assertEqual(int(after.get("vwap_related_failed_sample_count", 0)), 4)
        self.assertEqual(int(after.get("reason_fields_available_count", 0)), 4)
        self.assertEqual(int(after.get("subcondition_state_available_count", 0)), 4)
        self.assertEqual(
            dict(after.get("focus_bucket_counts", {})),
            {
                "vwap + volatility": 2,
                "htf + vwap + volatility": 1,
                "vwap_only": 1,
            },
        )
        self.assertEqual(
            dict(after.get("independent_blocker_evidence", {})),
            {
                "vwap_only_count": 1,
                "vwap_without_volatility_count": 1,
                "volatility_only_count": 1,
                "volatility_without_vwap_count": 1,
            },
        )
        self.assertIn("Per-sample failed_reason", " ".join(after.get("context_limitations", [])))

        vwap_only_bucket = after.get("buckets", {}).get("vwap_only", {})
        self.assertEqual(int(vwap_only_bucket.get("count", 0)), 1)
        self.assertAlmostEqual(float(vwap_only_bucket.get("share_of_vwap_related_failed_samples", 0.0)), 0.25, places=8)
        self.assertEqual(int(vwap_only_bucket.get("reason_fields_available_count", 0)), 1)
        self.assertEqual(
            dict(vwap_only_bucket.get("vwap_position_context_distribution", {})),
            {"above_vwap_but_under_threshold": 1},
        )
        self.assertEqual(
            dict(vwap_only_bucket.get("soft_pass_candidate_distribution", {})),
            {"soft_candidate": 1},
        )
        self.assertAlmostEqual(
            float(vwap_only_bucket.get("margins", {}).get("volatility_metric_minus_threshold", {}).get("mean", 99.0)),
            0.0001,
            places=8,
        )
        self.assertEqual(
            dict(vwap_only_bucket.get("failed_reason_distribution", {})),
            {"missing:stretched_from_vwap": 1},
        )
        self.assertEqual(
            dict(vwap_only_bucket.get("subcondition_false_pattern_distribution", {})),
            {"stretched_from_vwap": 1},
        )

        vwap_vol_bucket = after.get("buckets", {}).get("vwap + volatility", {})
        self.assertEqual(str(vwap_vol_bucket.get("top_label_blocker", "")), "stretched_from_vwap")
        self.assertEqual(str(vwap_vol_bucket.get("top_raw_blocker", "")), "stretched_from_vwap")
        self.assertEqual(
            dict(vwap_vol_bucket.get("vwap_position_context_distribution", {})),
            {"below_vwap": 2},
        )
        self.assertEqual(
            dict(vwap_vol_bucket.get("vwap_quality_distribution", {})),
            {"live": 2},
        )
        self.assertEqual(
            dict(vwap_vol_bucket.get("degraded_mode_distribution", {})),
            {"degraded": 1, "not_degraded": 1},
        )
        self.assertEqual(
            dict(vwap_vol_bucket.get("semantic_path_distribution", {})),
            {"explicit_rule_fail": 1, "explicit_rule_fail_with_degraded_context": 1},
        )
        self.assertEqual(
            dict(vwap_vol_bucket.get("missing_conditions_pattern_distribution", {})),
            {"stretched_from_vwap,volatility_regime_ok": 2},
        )
        self.assertEqual(
            dict(vwap_vol_bucket.get("missing_condition_item_counts", {})),
            {"stretched_from_vwap": 2, "volatility_regime_ok": 2},
        )
        self.assertEqual(
            dict(vwap_vol_bucket.get("subcondition_false_pattern_distribution", {})),
            {"stretched_from_vwap + volatility_regime_ok": 2},
        )
        self.assertAlmostEqual(
            float(vwap_vol_bucket.get("margins", {}).get("vwap_metric_minus_threshold", {}).get("mean", 99.0)),
            -0.003125,
            places=8,
        )

        htf_vwap_vol_bucket = after.get("buckets", {}).get("htf + vwap + volatility", {})
        self.assertEqual(
            dict(htf_vwap_vol_bucket.get("htf_direction_context_distribution", {})),
            {"strong_uptrend": 1},
        )
        self.assertEqual(
            dict(htf_vwap_vol_bucket.get("vwap_quality_distribution", {})),
            {"fallback": 1},
        )
        self.assertEqual(str(htf_vwap_vol_bucket.get("top_failed_reason", "")), "missing:htf_trend_ok,stretched_from_vwap,volatility_regime_ok")
        self.assertEqual(str(htf_vwap_vol_bucket.get("top_subcondition_false_pattern", "")), "htf_trend_ok + stretched_from_vwap + volatility_regime_ok")

    def test_inspect_vwap_semantics_comparison_focus_deltas(self):
        before_lines = [
            {
                "ts": "2026-03-16T04:00:00+00:00",
                "msg": (
                    "diag htf_trend_metric_used=35 htf_trend_threshold_used=30 "
                    "htf_trend_direction_context='not_strong_uptrend' "
                    "vwap_distance_metric_used=-0.00190 vwap_stretch_threshold_used=0.00125 "
                    "atr_norm=0.00044 volatility_threshold_used=0.0006 "
                    "strategy_audit_compact={'regime_filter_fail_count':1,'regime_filter_htf_trend_blocker_count':0,"
                    "'regime_filter_vwap_stretch_blocker_count':1,'regime_filter_volatility_blocker_count':1,"
                    "'regime_filter_news_blocker_count':0,'top_regime_filter_blocker':'stretched_from_vwap'} "
                    "strategy_audit_regime_filter={'failed_reason':'missing:stretched_from_vwap,volatility_regime_ok',"
                    "'missing_conditions':'stretched_from_vwap,volatility_regime_ok','degraded_mode':0.0,"
                    "'source_flags':{'vwap_quality':'live','news_quality':'live'},"
                    "'regime_filter_subconditions_state':{'htf_trend_ok':True,'stretched_from_vwap':False,"
                    "'volatility_regime_ok':False,'news_veto':True}} "
                    "strategy_audit={'source_quality_counts':{'regime_filter':{'vwap_quality':{'live':1},'news_quality':{'live':1}}}}"
                ),
            },
            {
                "ts": "2026-03-16T04:01:00+00:00",
                "msg": (
                    "diag htf_trend_metric_used=36 htf_trend_threshold_used=30 "
                    "htf_trend_direction_context='not_strong_uptrend' "
                    "vwap_distance_metric_used=-0.00185 vwap_stretch_threshold_used=0.00125 "
                    "atr_norm=0.00046 volatility_threshold_used=0.0006 "
                    "strategy_audit_compact={'regime_filter_fail_count':1,'regime_filter_htf_trend_blocker_count':0,"
                    "'regime_filter_vwap_stretch_blocker_count':1,'regime_filter_volatility_blocker_count':1,"
                    "'regime_filter_news_blocker_count':0,'top_regime_filter_blocker':'stretched_from_vwap'} "
                    "strategy_audit_regime_filter={'failed_reason':'missing:stretched_from_vwap,volatility_regime_ok',"
                    "'missing_conditions':'stretched_from_vwap,volatility_regime_ok','degraded_mode':0.0,"
                    "'source_flags':{'vwap_quality':'live','news_quality':'live'},"
                    "'regime_filter_subconditions_state':{'htf_trend_ok':True,'stretched_from_vwap':False,"
                    "'volatility_regime_ok':False,'news_veto':True}} "
                    "strategy_audit={'source_quality_counts':{'regime_filter':{'vwap_quality':{'live':1},'news_quality':{'live':1}}}}"
                ),
            },
            {
                "ts": "2026-03-16T04:02:00+00:00",
                "msg": (
                    "diag htf_trend_metric_used=37 htf_trend_threshold_used=30 "
                    "htf_trend_direction_context='strong_uptrend' "
                    "vwap_distance_metric_used=0.00090 vwap_stretch_threshold_used=0.00125 "
                    "atr_norm=0.00070 volatility_threshold_used=0.0006 "
                    "strategy_audit_compact={'regime_filter_fail_count':1,'regime_filter_htf_trend_blocker_count':0,"
                    "'regime_filter_vwap_stretch_blocker_count':1,'regime_filter_volatility_blocker_count':0,"
                    "'regime_filter_news_blocker_count':0,'top_regime_filter_blocker':'stretched_from_vwap'} "
                    "strategy_audit_regime_filter={'failed_reason':'missing:stretched_from_vwap',"
                    "'missing_conditions':'stretched_from_vwap','degraded_mode':0.0,"
                    "'source_flags':{'vwap_quality':'live','news_quality':'live'},"
                    "'regime_filter_subconditions_state':{'htf_trend_ok':True,'stretched_from_vwap':False,"
                    "'volatility_regime_ok':True,'news_veto':True}} "
                    "strategy_audit={'source_quality_counts':{'regime_filter':{'vwap_quality':{'live':1},'news_quality':{'live':1}}}}"
                ),
            },
        ]
        after_lines = [
            {
                "ts": "2026-03-16T05:00:00+00:00",
                "msg": (
                    "diag htf_trend_metric_used=20 htf_trend_threshold_used=30 "
                    "htf_trend_direction_context='strong_uptrend' "
                    "vwap_distance_metric_used=-0.00195 vwap_stretch_threshold_used=0.00125 "
                    "atr_norm=0.00043 volatility_threshold_used=0.0006 "
                    "strategy_audit_compact={'regime_filter_fail_count':1,'regime_filter_htf_trend_blocker_count':1,"
                    "'regime_filter_vwap_stretch_blocker_count':1,'regime_filter_volatility_blocker_count':1,"
                    "'regime_filter_news_blocker_count':0,'top_regime_filter_blocker':'htf_trend_ok'} "
                    "strategy_audit_regime_filter={'failed_reason':'missing:htf_trend_ok,stretched_from_vwap,volatility_regime_ok',"
                    "'missing_conditions':'htf_trend_ok,stretched_from_vwap,volatility_regime_ok','degraded_mode':1.0,"
                    "'source_flags':{'vwap_quality':'fallback','news_quality':'unavailable'},"
                    "'regime_filter_subconditions_state':{'htf_trend_ok':False,'stretched_from_vwap':False,"
                    "'volatility_regime_ok':False,'news_veto':True}} "
                    "strategy_audit={'source_quality_counts':{'regime_filter':{'vwap_quality':{'fallback':1},'news_quality':{'unavailable':1}}}}"
                ),
            },
            {
                "ts": "2026-03-16T05:01:00+00:00",
                "msg": (
                    "diag htf_trend_metric_used=21 htf_trend_threshold_used=30 "
                    "htf_trend_direction_context='strong_uptrend' "
                    "vwap_distance_metric_used=-0.00192 vwap_stretch_threshold_used=0.00125 "
                    "atr_norm=0.00044 volatility_threshold_used=0.0006 "
                    "strategy_audit_compact={'regime_filter_fail_count':1,'regime_filter_htf_trend_blocker_count':1,"
                    "'regime_filter_vwap_stretch_blocker_count':1,'regime_filter_volatility_blocker_count':1,"
                    "'regime_filter_news_blocker_count':0,'top_regime_filter_blocker':'htf_trend_ok'} "
                    "strategy_audit_regime_filter={'failed_reason':'missing:htf_trend_ok,stretched_from_vwap,volatility_regime_ok',"
                    "'missing_conditions':'htf_trend_ok,stretched_from_vwap,volatility_regime_ok','degraded_mode':1.0,"
                    "'source_flags':{'vwap_quality':'fallback','news_quality':'unavailable'},"
                    "'regime_filter_subconditions_state':{'htf_trend_ok':False,'stretched_from_vwap':False,"
                    "'volatility_regime_ok':False,'news_veto':True}} "
                    "strategy_audit={'source_quality_counts':{'regime_filter':{'vwap_quality':{'fallback':1},'news_quality':{'unavailable':1}}}}"
                ),
            },
            {
                "ts": "2026-03-16T05:02:00+00:00",
                "msg": (
                    "diag htf_trend_metric_used=35 htf_trend_threshold_used=30 "
                    "htf_trend_direction_context='not_strong_uptrend' "
                    "vwap_distance_metric_used=-0.00188 vwap_stretch_threshold_used=0.00125 "
                    "atr_norm=0.00044 volatility_threshold_used=0.0006 "
                    "strategy_audit_compact={'regime_filter_fail_count':1,'regime_filter_htf_trend_blocker_count':0,"
                    "'regime_filter_vwap_stretch_blocker_count':1,'regime_filter_volatility_blocker_count':1,"
                    "'regime_filter_news_blocker_count':0,'top_regime_filter_blocker':'stretched_from_vwap'} "
                    "strategy_audit_regime_filter={'failed_reason':'missing:stretched_from_vwap,volatility_regime_ok',"
                    "'missing_conditions':'stretched_from_vwap,volatility_regime_ok','degraded_mode':0.0,"
                    "'source_flags':{'vwap_quality':'live','news_quality':'live'},"
                    "'regime_filter_subconditions_state':{'htf_trend_ok':True,'stretched_from_vwap':False,"
                    "'volatility_regime_ok':False,'news_veto':True}} "
                    "strategy_audit={'source_quality_counts':{'regime_filter':{'vwap_quality':{'live':1},'news_quality':{'live':1}}}}"
                ),
            },
        ]

        proc = self._run_inspect_vwap_semantics(after_lines, before_lines=before_lines)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)

        payload = json.loads(proc.stdout)
        comparison = payload.get("comparison", {})
        self.assertEqual(str(comparison.get("top_focus_bucket_before", "")), "vwap + volatility")
        self.assertEqual(str(comparison.get("top_focus_bucket_after", "")), "htf + vwap + volatility")
        self.assertEqual(int(comparison.get("focus_bucket_count_delta", {}).get("htf + vwap + volatility", 0)), 2)
        self.assertEqual(int(comparison.get("focus_bucket_count_delta", {}).get("vwap_only", 0)), -1)
        self.assertAlmostEqual(
            float(comparison.get("focus_bucket_share_of_failed_samples_delta", {}).get("htf + vwap + volatility", 99.0)),
            2.0 / 3.0,
            places=8,
        )
        self.assertEqual(int(comparison.get("vwap_only_count_before", 0)), 1)
        self.assertEqual(int(comparison.get("vwap_only_count_after", 0)), 0)
        self.assertEqual(int(comparison.get("volatility_without_vwap_count_before", 0)), 0)
        self.assertEqual(int(comparison.get("volatility_without_vwap_count_after", 0)), 0)

    def test_triage_calibration_result_exit_code_mapping(self):
        cases = [
            (
                {
                    "after": {"summary": {"top_regime_filter_blocker_combination": "volatility_regime_ok"}},
                    "calibration_recommendation": {
                        "SAFE_TO_CONTINUE": True,
                        "ACTION_VERDICT": "single_blocker_ready",
                        "STOP_REASON": "",
                        "RUNBOOK_ACTIONS": ["Prepare one isolated tweak for regime_filter.volatility_regime_ok."],
                    },
                },
                0,
                "VERDICT: single_blocker_ready",
            ),
            (
                {
                    "after": {"summary": {"top_regime_filter_blocker_combination": "stretched_from_vwap + volatility_regime_ok"}},
                    "calibration_recommendation": {
                        "SAFE_TO_CONTINUE": False,
                        "ACTION_VERDICT": "co_dominant_overlap",
                        "STOP_REASON": "blocker_semantics_disagreement",
                        "RUNBOOK_ACTIONS": ["Pause threshold changes for this branch."],
                    },
                },
                11,
                "STOP_REASON: blocker_semantics_disagreement",
            ),
            (
                {
                    "after": {"summary": {"top_regime_filter_blocker_combination": "none"}},
                    "calibration_recommendation": {
                        "SAFE_TO_CONTINUE": False,
                        "ACTION_VERDICT": "pause_calibration",
                        "STOP_REASON": "market_context_shift_detected",
                        "RUNBOOK_ACTIONS": ["Pause threshold changes for this branch."],
                    },
                },
                12,
                "STOP_REASON: market_context_shift_detected",
            ),
            (
                {
                    "after": {"summary": {"top_regime_filter_blocker_combination": "none"}},
                    "calibration_recommendation": {
                        "SAFE_TO_CONTINUE": False,
                        "ACTION_VERDICT": "pause_calibration",
                        "STOP_REASON": "window_size_not_comparable",
                        "RUNBOOK_ACTIONS": ["Collect another window with a comparable sample count to the reference window."],
                    },
                },
                13,
                "STOP_REASON: window_size_not_comparable",
            ),
        ]

        for payload, expected_code, expected_line in cases:
            proc = self._run_triage_calibration_result(payload)
            self.assertEqual(proc.returncode, expected_code, msg=proc.stderr)
            self.assertIn(expected_line, proc.stdout)

    def test_summarize_observation_failure_surfaces_python_stderr(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        ps_exe = r"C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe"
        script_path = os.path.join(repo_root, "scripts", "observation", "summarize_observation.ps1")

        with tempfile.TemporaryDirectory() as tmpdir:
            audit_file = os.path.join(tmpdir, "audit.log")
            with open(audit_file, "w", encoding="utf-8") as fh:
                fh.write("{}\n")

            fake_python = os.path.join(tmpdir, "fake_python_fail.cmd")
            with open(fake_python, "w", encoding="utf-8") as fh:
                fh.write("@echo off\r\n")
                fh.write("echo FAKE_STDOUT_LINE\r\n")
                fh.write("echo FAKE_STDERR_LINE 1>&2\r\n")
                fh.write("exit /b 7\r\n")

            proc = subprocess.run(
                [
                    ps_exe,
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    script_path,
                    "-AuditExtractFile",
                    audit_file,
                    "-PythonPath",
                    fake_python,
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(proc.returncode, 0)
            combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
            self.assertIn("FAKE_STDERR_LINE", combined)
            self.assertIn("FAKE_STDOUT_LINE", combined)
            self.assertIn("Python observation summary failed", combined)

    def test_summarize_observation_json_path_backward_compatible(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        ps_exe = r"C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe"
        script_path = os.path.join(repo_root, "scripts", "observation", "summarize_observation.ps1")

        with tempfile.TemporaryDirectory() as tmpdir:
            audit_file = os.path.join(tmpdir, "audit.log")
            with open(audit_file, "w", encoding="utf-8") as fh:
                fh.write("{}\n")

            out_json = os.path.join(tmpdir, "summary.json")
            fake_python = os.path.join(tmpdir, "fake_python_ok.cmd")
            with open(fake_python, "w", encoding="utf-8") as fh:
                fh.write("@echo off\r\n")
                fh.write(
                    "echo {\"after\":{\"file\":\"ok\",\"summary\":{\"samples\":1}},\"calibration_recommendation\":{\"SAFE_TO_CONTINUE\":false,\"NEXT_TARGET_LAYER\":\"none\",\"NEXT_TARGET_SUBCONDITION\":\"none\",\"WHY_THIS_TARGET\":\"\",\"WHY_NOT_OTHERS\":[],\"REQUIRED_WINDOW_QUALITY\":{},\"STOP_REASON\":\"demo\"}}\r\n"
                )
                fh.write("exit /b 0\r\n")

            proc = subprocess.run(
                [
                    ps_exe,
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    script_path,
                    "-AuditExtractFile",
                    audit_file,
                    "-OutJson",
                    out_json,
                    "-PythonPath",
                    fake_python,
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
            )

            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertIn("after", payload)
            self.assertIn("calibration_recommendation", payload)
            self.assertIn("SAFE_TO_CONTINUE", payload.get("calibration_recommendation", {}))
            self.assertTrue(os.path.exists(out_json))

    def test_summarize_observation_rejects_placeholder_paths_with_clear_message(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        ps_exe = r"C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe"
        script_path = os.path.join(repo_root, "scripts", "observation", "summarize_observation.ps1")

        proc = subprocess.run(
            [
                ps_exe,
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                script_path,
                "-AuditExtractFile",
                "<after_extract.log>",
                "-CompareAuditExtractFile",
                "<before_extract.log>",
                "-OutJson",
                ".\\logs\\observation\\comparison_latest.json",
                "-PrintTriage",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(proc.returncode, 0)
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        self.assertIn("Replace the placeholder in -AuditExtractFile with a real path", combined)

    def test_summarize_observation_prints_triage_when_requested(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        ps_exe = r"C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe"
        script_path = os.path.join(repo_root, "scripts", "observation", "summarize_observation.ps1")

        with tempfile.TemporaryDirectory() as tmpdir:
            audit_file = os.path.join(tmpdir, "audit.log")
            with open(audit_file, "w", encoding="utf-8") as fh:
                fh.write("{}\n")

            out_json = os.path.join(tmpdir, "comparison.json")
            fake_python = os.path.join(tmpdir, "fake_python_triage.cmd")
            with open(fake_python, "w", encoding="utf-8") as fh:
                fh.write("@echo off\r\n")
                fh.write(
                    "echo {\"after\":{\"file\":\"ok\",\"summary\":{\"samples\":1,\"top_regime_filter_blocker_combination\":\"stretched_from_vwap + volatility_regime_ok\"}},\"calibration_recommendation\":{\"SAFE_TO_CONTINUE\":false,\"ACTION_VERDICT\":\"co_dominant_overlap\",\"RUNBOOK_ACTIONS\":[\"Pause threshold changes for this branch.\",\"Review blocker combination 'stretched_from_vwap + volatility_regime_ok' as the current overlap driver.\"],\"NEXT_TARGET_LAYER\":\"none\",\"NEXT_TARGET_SUBCONDITION\":\"none\",\"WHY_THIS_TARGET\":\"\",\"WHY_NOT_OTHERS\":[],\"REQUIRED_WINDOW_QUALITY\":{},\"STOP_REASON\":\"co_dominant_regime_blockers\"}}\r\n"
                )
                fh.write("exit /b 0\r\n")

            proc = subprocess.run(
                [
                    ps_exe,
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    script_path,
                    "-AuditExtractFile",
                    audit_file,
                    "-OutJson",
                    out_json,
                    "-PythonPath",
                    fake_python,
                    "-PrintTriage",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
            )

            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("VERDICT: co_dominant_overlap", proc.stdout)
            self.assertIn("STOP_REASON: co_dominant_regime_blockers", proc.stdout)
            self.assertIn("TOP_COMBINATION: stretched_from_vwap + volatility_regime_ok", proc.stdout)
            self.assertIn("ACTION 1: Pause threshold changes for this branch.", proc.stdout)

            with open(out_json, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            self.assertEqual(str(payload["calibration_recommendation"]["ACTION_VERDICT"]), "co_dominant_overlap")

    def test_collect_observation_window_prints_triage_when_requested(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        ps_exe = r"C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe"
        script_path = os.path.join(repo_root, "scripts", "observation", "collect_observation_window.ps1")

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "observation")
            os.makedirs(log_dir, exist_ok=True)

            fake_python = os.path.join(tmpdir, "fake_python_collect.cmd")
            with open(fake_python, "w", encoding="utf-8") as fh:
                fh.write("@echo off\r\n")
                fh.write("if \"%~1\"==\"-m\" (\r\n")
                fh.write("  echo {\"ts\":\"2026-03-15T00:00:00+00:00\",\"msg\":\"strategy_audit_compact={\\\"top_regime_filter_blocker\\\":\\\"stretched_from_vwap\\\"}\"}\r\n")
                fh.write("  exit /b 0\r\n")
                fh.write(")\r\n")
                fh.write(
                    "echo {\"after\":{\"file\":\"ok\",\"summary\":{\"samples\":1,\"top_regime_filter_blocker_combination\":\"stretched_from_vwap + volatility_regime_ok\"}},\"calibration_recommendation\":{\"SAFE_TO_CONTINUE\":false,\"ACTION_VERDICT\":\"co_dominant_overlap\",\"RUNBOOK_ACTIONS\":[\"Pause threshold changes for this branch.\"],\"NEXT_TARGET_LAYER\":\"none\",\"NEXT_TARGET_SUBCONDITION\":\"none\",\"WHY_THIS_TARGET\":\"\",\"WHY_NOT_OTHERS\":[],\"REQUIRED_WINDOW_QUALITY\":{},\"STOP_REASON\":\"co_dominant_regime_blockers\"}}\r\n"
                )
                fh.write("exit /b 0\r\n")

            proc = subprocess.run(
                [
                    ps_exe,
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    script_path,
                    "-DurationMinutes",
                    "1",
                    "-PauseSeconds",
                    "1",
                    "-Tag",
                    "collect_triage_test",
                    "-PythonPath",
                    fake_python,
                    "-LogDir",
                    log_dir,
                    "-PrintTriage",
                    "-MaxRuns",
                    "1",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
            )

            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("VERDICT: co_dominant_overlap", proc.stdout)
            self.assertIn("STOP_REASON: co_dominant_regime_blockers", proc.stdout)
            self.assertIn("TOP_COMBINATION: stretched_from_vwap + volatility_regime_ok", proc.stdout)

            json_start = proc.stdout.find("{")
            self.assertGreaterEqual(json_start, 0, msg=proc.stdout)
            payload = json.loads(proc.stdout[json_start:])
            self.assertEqual(int(payload["max_runs"]), 1)
            self.assertEqual(int(payload["runs"]), 1)
            self.assertEqual(str(payload["summary"]["calibration_recommendation"]["ACTION_VERDICT"]), "co_dominant_overlap")

    def test_review_regime_diagnostics_handles_missing_fields_safely(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        python_exe = os.path.join(repo_root, ".venv", "Scripts", "python.exe")
        tool_path = os.path.join(repo_root, "scripts", "observation", "review_regime_diagnostics.py")

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "combined.log")
            msg = (
                "metrics={} "
                "strategy_audit_compact={'regime_filter_htf_trend_blocker_count':1,'regime_filter_vwap_stretch_blocker_count':0,'regime_filter_volatility_blocker_count':1,'regime_filter_news_blocker_count':0,'top_regime_filter_blocker':'htf_trend_ok'}"
            )
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": "2026-03-13T00:00:00+00:00", "msg": msg}, ensure_ascii=False) + "\n")

            proc = subprocess.run(
                [python_exe, tool_path, "--after", log_path],
                cwd=repo_root,
                capture_output=True,
                text=True,
            )

            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            payload = json.loads(proc.stdout)
            after = payload.get("after", {})
            self.assertIn("margins", after)
            self.assertEqual(int(after.get("margins", {}).get("htf_metric_minus_threshold", {}).get("count", -1)), 0)
            self.assertEqual(int(after.get("presence", {}).get("htf_trend_metric_used", -1)), 0)
            self.assertEqual(
                int(after.get("blocker_mix", {}).get("regime_blocker_counts", {}).get("htf_trend_ok", -1)),
                1,
            )
            blocker_mix = after.get("blocker_mix", {})
            self.assertEqual(str(blocker_mix.get("top_blocker_by_raw_coverage", "")), "htf_trend_ok")
            self.assertEqual(str(blocker_mix.get("top_blocker_by_label", "")), "htf_trend_ok")
            self.assertEqual(list(blocker_mix.get("co_dominant_blockers", [])), ["htf_trend_ok", "volatility_regime_ok"])
            self.assertEqual(str(blocker_mix.get("dominance_mode", "")), "co_dominant_raw")
            self.assertEqual(str(blocker_mix.get("action_verdict", "")), "co_dominant_overlap")
            self.assertEqual(
                str(blocker_mix.get("top_regime_filter_blocker_combination", "")),
                "htf_trend_ok + volatility_regime_ok",
            )
            self.assertAlmostEqual(
                float(blocker_mix.get("top_regime_filter_blocker_combination_share", 0.0)),
                1.0,
                places=8,
            )

    def test_review_regime_diagnostics_margin_calculations(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        python_exe = os.path.join(repo_root, ".venv", "Scripts", "python.exe")
        tool_path = os.path.join(repo_root, "scripts", "observation", "review_regime_diagnostics.py")

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "combined.log")
            line1 = {
                "ts": "2026-03-13T00:00:00+00:00",
                "msg": (
                    "diag htf_trend_metric_used=28 htf_trend_threshold_used=29 "
                    "vwap_distance_metric_used=0.0013 vwap_stretch_threshold_used=0.0012 "
                    "atr_norm=0.0015 volatility_threshold_used=0.0016 "
                    "strategy_audit_compact={'regime_filter_htf_trend_blocker_count':1,'regime_filter_vwap_stretch_blocker_count':1,'regime_filter_volatility_blocker_count':1,'regime_filter_news_blocker_count':0,'top_regime_filter_blocker':'htf_trend_ok'}"
                ),
            }
            line2 = {
                "ts": "2026-03-13T00:01:00+00:00",
                "msg": (
                    "diag htf_trend_metric_used=30 htf_trend_threshold_used=29 "
                    "vwap_distance_metric_used=0.0011 vwap_stretch_threshold_used=0.0012 "
                    "atr_norm=0.0018 volatility_threshold_used=0.0016 "
                    "strategy_audit_compact={'regime_filter_htf_trend_blocker_count':0,'regime_filter_vwap_stretch_blocker_count':1,'regime_filter_volatility_blocker_count':1,'regime_filter_news_blocker_count':0,'top_regime_filter_blocker':'volatility_regime_ok'}"
                ),
            }
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(line1, ensure_ascii=False) + "\n")
                fh.write(json.dumps(line2, ensure_ascii=False) + "\n")

            proc = subprocess.run(
                [python_exe, tool_path, "--after", log_path],
                cwd=repo_root,
                capture_output=True,
                text=True,
            )

            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            payload = json.loads(proc.stdout)
            after = payload.get("after", {})
            margins = after.get("margins", {})

            self.assertEqual(int(margins.get("htf_metric_minus_threshold", {}).get("count", 0)), 2)
            self.assertAlmostEqual(float(margins.get("htf_metric_minus_threshold", {}).get("mean", 99.0)), 0.0, places=8)
            self.assertAlmostEqual(float(margins.get("vwap_metric_minus_threshold", {}).get("mean", 99.0)), 0.0, places=8)
            self.assertAlmostEqual(float(margins.get("volatility_metric_minus_threshold", {}).get("mean", 99.0)), 0.00005, places=8)

            blocker_mix = after.get("blocker_mix", {})
            self.assertEqual(str(blocker_mix.get("top_regime_filter_blocker", "")), "htf_trend_ok")
            self.assertEqual(int(blocker_mix.get("regime_blocker_counts", {}).get("volatility_regime_ok", 0)), 2)
            self.assertEqual(str(blocker_mix.get("top_blocker_by_raw_coverage", "")), "stretched_from_vwap")
            self.assertEqual(str(blocker_mix.get("top_blocker_by_label", "")), "htf_trend_ok")
            self.assertAlmostEqual(float(blocker_mix.get("top_blocker_by_raw_coverage_share", 99.0)), 1.0, places=8)
            self.assertAlmostEqual(float(blocker_mix.get("top_blocker_by_label_share", 99.0)), 0.5, places=8)
            self.assertEqual(str(blocker_mix.get("dominance_mode", "")), "raw_vs_label_disagreement")
            self.assertIn("favor", str(blocker_mix.get("dominance_explanation", "")).lower())
            self.assertEqual(str(blocker_mix.get("action_verdict", "")), "co_dominant_overlap")
            self.assertEqual(
                dict(blocker_mix.get("regime_blocker_combination_counts", {})),
                {
                    "htf_trend_ok + stretched_from_vwap + volatility_regime_ok": 1,
                    "stretched_from_vwap + volatility_regime_ok": 1,
                },
            )
            self.assertAlmostEqual(
                float(
                    blocker_mix.get("regime_blocker_combination_share_by_failed_samples", {}).get(
                        "stretched_from_vwap + volatility_regime_ok",
                        99.0,
                    )
                ),
                0.5,
                places=8,
            )

if __name__ == "__main__":
    unittest.main()










