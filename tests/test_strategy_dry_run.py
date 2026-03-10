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

    def test_layer2_weakness_semantics_pass(self):
        df = self._build_df()
        signal_gen = SignalGenerator(SignalConfig())

        passed, layer2 = signal_gen._layer2_weakness_confirmation(df, "SHORT")

        self.assertTrue(passed)
        self.assertEqual(layer2.get("passed"), 1.0)
        self.assertEqual(layer2.get("failed_reason"), "none")
        self.assertIn("price_up_or_near_high", layer2)
        self.assertIn("obv_bearish_divergence", layer2)
        self.assertIn("cvd_bearish_divergence", layer2)
        self.assertIn("weakness_strength", layer2)

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
            "layer4_oi_blocker_count",
            "layer4_price_blocker_count",
            "layer5_vp_based_count",
            "no_signal_ratio",
            "short_signal_ratio",
        ):
            self.assertIn(key, compact)

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
        self.assertIn("strategy_audit_layer4", payload)
        self.assertIn("strategy_audit_source_quality", payload)

        compact = payload.get("strategy_audit_compact", {})
        self.assertIn("evaluations_total", compact)
        self.assertIn("layer4_price_blocker_count", compact)
        self.assertIn("layer4_oi_blocker_count", compact)

        layer4 = payload.get("strategy_audit_layer4", {})
        self.assertIn("layer4_sentiment_blocker_count", layer4)
        self.assertIn("layer4_funding_blocker_count", layer4)
        self.assertIn("layer4_lsr_blocker_count", layer4)
        self.assertIn("layer4_oi_blocker_count", layer4)
        self.assertIn("layer4_price_blocker_count", layer4)

        source_quality = payload.get("strategy_audit_source_quality", {})
        self.assertIn("regime_filter", source_quality)
        self.assertIn("layer4_fake_filter", source_quality)
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

    def test_regime_filter_pass_with_degraded_news(self):
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
        self.assertEqual(regime_diag.get("degraded_mode"), 1.0)
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

if __name__ == "__main__":
    unittest.main()























