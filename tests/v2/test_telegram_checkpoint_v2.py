from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from alerts.chart_generator import build_signal_chart
from app.main import _build_early_watch_candidate
from core.volume_profile import VolumeProfileLevels
from trading.alerts.signal_card import (
    build_early_signal_caption,
    build_signal_caption,
    build_symbol_copy_reply_markup,
)
from trading.signals.signal_types import IntentAction, StrategyIntent


class TelegramCheckpointV2Tests(unittest.TestCase):
    @staticmethod
    def _build_df() -> pd.DataFrame:
        idx = pd.date_range("2026-03-01", periods=80, freq="min", tz="UTC")
        close = np.linspace(100.0, 108.0, 80)
        close[-1] = 107.6
        close[-2] = 108.1
        df = pd.DataFrame(
            {
                "open": close - 0.3,
                "high": close + 0.7,
                "low": close - 0.8,
                "close": close,
                "ema20": pd.Series(close).ewm(span=20, adjust=False).mean().values,
                "ema50": pd.Series(close).ewm(span=50, adjust=False).mean().values,
                "vwap": close - 0.4,
                "rsi": np.linspace(54.0, 61.0, 80),
                "volume_spike": np.linspace(1.0, 1.4, 80),
                "bb_upper": close + 0.25,
                "kc_upper": close + 0.15,
                "atr": np.full(80, 1.2),
                "hist": np.linspace(0.12, 0.02, 80),
                "obv": np.linspace(100.0, 150.0, 80),
                "cvd": np.linspace(120.0, 165.0, 80),
            },
            index=idx,
        )
        df.iloc[-1, df.columns.get_loc("high")] = float(df.iloc[-1]["close"]) + 1.3
        df.iloc[-1, df.columns.get_loc("volume_spike")] = 1.18
        df.iloc[-1, df.columns.get_loc("obv")] = float(df.iloc[-2]["obv"]) - 1.5
        df.iloc[-1, df.columns.get_loc("cvd")] = float(df.iloc[-2]["cvd"]) - 1.2
        return df

    @staticmethod
    def _trace_meta() -> dict:
        return {
            "layer_trace": {
                "layers": {
                    "regime_filter": {"passed": True, "details": {}},
                    "layer1_pump_detection": {
                        "passed": False,
                        "details": {
                            "clean_pump_pct": 0.0582,
                            "clean_pump_min_pct_used": 0.05,
                            "clean_pump_ok": 1.0,
                            "rsi": 58.2,
                            "volume_spike": 1.17,
                        },
                    },
                    "layer2_weakness_confirmation": {
                        "passed": False,
                        "details": {"weakness_strength": 1.0},
                    },
                }
            },
            "layer_failed": "layer1_pump_detection",
        }

    def test_build_signal_caption_includes_pump_size_in_russian(self):
        caption = build_signal_caption(
            symbol="SOLUSDT",
            timeframe="1",
            mode="testnet",
            action_label="РАННИЙ ШОРТ",
            entry=91.469,
            tp=88.120,
            sl=93.550,
            confidence=0.72,
            reason="layered_short_entry",
            trace_meta=self._trace_meta(),
            enriched=self._build_df(),
        )
        self.assertIn("РАННИЙ ШОРТ", caption)
        self.assertIn("Pump 5.82%", caption)
        self.assertIn("Режим: testnet", caption)
        self.assertIn("Контекст по монете #SOL", caption)

    def test_build_early_signal_caption_includes_quality_grade(self):
        caption = build_early_signal_caption(
            symbol="SOLUSDT",
            timeframe="1",
            mode="testnet",
            phase_label="РАННИЙ ШОРТ: НАБЛЮДЕНИЕ",
            price=91.469,
            trace_meta=self._trace_meta(),
            watch_score=4.5,
            watch_max_score=8.0,
            quality_score=6.8,
            quality_max_score=10.0,
            quality_grade="B",
            continuation_risk=1.5,
            continuation_max_score=4.0,
            triggers=["цена у верхней зоны", "объём затухает"],
            wait_for="подтверждение входа",
            enriched=self._build_df(),
        )
        self.assertIn("РАННИЙ ШОРТ: НАБЛЮДЕНИЕ", caption)
        self.assertIn("Watch score: 4.5/8.0", caption)
        self.assertIn("Класс сетапа: B (6.8/10.0)", caption)
        self.assertIn("Риск продолжения: 1.5/4.0", caption)

    def test_build_symbol_copy_reply_markup_returns_copy_button(self):
        markup = build_symbol_copy_reply_markup("SOLUSDT")
        self.assertEqual(markup["inline_keyboard"][0][0]["text"], "SOLUSDT")
        self.assertEqual(markup["inline_keyboard"][0][0]["copy_text"]["text"], "SOLUSDT")

    def test_chart_generator_returns_png_bytes(self):
        image = build_signal_chart(
            symbol="SOLUSDT",
            df=self._build_df(),
            side="SHORT",
            entry=107.6,
            tp=105.2,
            sl=108.9,
            volume_profile=VolumeProfileLevels(poc=106.4, vah=107.9, val=104.8),
            timeframe_label="1m",
            show_trade_levels=True,
        )
        self.assertIsInstance(image, bytes)
        self.assertGreater(len(image), 1024)

    def test_early_watch_candidate_emits_for_clean_pump_context(self):
        intent = StrategyIntent(
            symbol="SOLUSDT",
            action=IntentAction.HOLD,
            reason="no_signal_layer1_pump_detection",
            metadata=self._trace_meta(),
        )
        candidate = _build_early_watch_candidate(
            symbol="SOLUSDT",
            timeframe="1",
            mode="testnet",
            enriched=self._build_df(),
            intent=intent,
        )
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["phase"], "WATCH")
        self.assertIn("РАННИЙ ШОРТ", candidate["caption"])

    def test_early_watch_candidate_allows_sub_confirmed_clean_pump_when_quality_is_good(self):
        df = self._build_df()
        trace = self._trace_meta()
        trace["layer_trace"]["layers"]["layer1_pump_detection"]["details"]["clean_pump_pct"] = 0.044
        trace["layer_trace"]["layers"]["layer1_pump_detection"]["details"]["clean_pump_min_pct_used"] = 0.05
        trace["layer_trace"]["layers"]["layer1_pump_detection"]["details"]["clean_pump_ok"] = 0.0

        intent = StrategyIntent(
            symbol="SOLUSDT",
            action=IntentAction.HOLD,
            reason="no_signal_layer1_pump_detection",
            metadata=trace,
        )
        candidate = _build_early_watch_candidate(
            symbol="SOLUSDT",
            timeframe="1",
            mode="testnet",
            enriched=df,
            intent=intent,
        )
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["phase"], "WATCH")

    def test_early_watch_candidate_skips_when_pump_is_still_accelerating(self):
        df = self._build_df()
        df.iloc[-1, df.columns.get_loc("close")] = float(df.iloc[-2]["close"]) + 1.6
        df.iloc[-1, df.columns.get_loc("high")] = float(df.iloc[-1]["close"]) + 0.2
        df.iloc[-1, df.columns.get_loc("rsi")] = float(df.iloc[-2]["rsi"]) + 3.0
        df.iloc[-1, df.columns.get_loc("hist")] = float(df.iloc[-2]["hist"]) + 0.05
        df.iloc[-1, df.columns.get_loc("volume_spike")] = float(df.iloc[-2]["volume_spike"]) + 0.2
        df.iloc[-1, df.columns.get_loc("obv")] = float(df.iloc[-2]["obv"]) + 5.0
        df.iloc[-1, df.columns.get_loc("cvd")] = float(df.iloc[-2]["cvd"]) + 5.0

        intent = StrategyIntent(
            symbol="SOLUSDT",
            action=IntentAction.HOLD,
            reason="no_signal_layer1_pump_detection",
            metadata=self._trace_meta(),
        )
        candidate = _build_early_watch_candidate(
            symbol="SOLUSDT",
            timeframe="1",
            mode="testnet",
            enriched=df,
            intent=intent,
        )
        self.assertIsNone(candidate)

    def test_early_watch_candidate_skips_when_peak_is_reclaimed_without_real_rollover(self):
        df = self._build_df()
        df.iloc[-3, df.columns.get_loc("close")] = 107.2
        df.iloc[-2, df.columns.get_loc("close")] = 107.0
        df.iloc[-1, df.columns.get_loc("open")] = 107.05
        df.iloc[-1, df.columns.get_loc("close")] = 108.05
        df.iloc[-1, df.columns.get_loc("high")] = 108.25
        df.iloc[-1, df.columns.get_loc("low")] = 106.98
        df.iloc[-1, df.columns.get_loc("rsi")] = 66.0
        df.iloc[-1, df.columns.get_loc("hist")] = 0.08
        df.iloc[-1, df.columns.get_loc("volume_spike")] = 0.92

        intent = StrategyIntent(
            symbol="SOLUSDT",
            action=IntentAction.HOLD,
            reason="no_signal_layer1_pump_detection",
            metadata=self._trace_meta(),
        )
        candidate = _build_early_watch_candidate(
            symbol="SOLUSDT",
            timeframe="1",
            mode="testnet",
            enriched=df,
            intent=intent,
        )
        self.assertIsNone(candidate)

    def test_early_setup_candidate_skips_when_layer2_is_weak_and_price_reaccelerates(self):
        df = self._build_df()
        df.iloc[-3, df.columns.get_loc("close")] = 107.4
        df.iloc[-2, df.columns.get_loc("close")] = 107.2
        df.iloc[-1, df.columns.get_loc("open")] = 107.25
        df.iloc[-1, df.columns.get_loc("close")] = 107.95
        df.iloc[-1, df.columns.get_loc("high")] = 108.30
        df.iloc[-1, df.columns.get_loc("low")] = 107.10
        df.iloc[-1, df.columns.get_loc("rsi")] = 64.0
        df.iloc[-1, df.columns.get_loc("hist")] = 0.06
        df.iloc[-1, df.columns.get_loc("volume_spike")] = 1.08

        trace = self._trace_meta()
        trace["layer_failed"] = "layer2_weakness_confirmation"
        trace["layer_trace"]["layers"]["layer1_pump_detection"]["passed"] = True
        trace["layer_trace"]["layers"]["layer2_weakness_confirmation"]["details"]["weakness_strength"] = 0.33

        intent = StrategyIntent(
            symbol="SOLUSDT",
            action=IntentAction.HOLD,
            reason="no_signal_layer2_weakness_confirmation",
            metadata=trace,
        )
        candidate = _build_early_watch_candidate(
            symbol="SOLUSDT",
            timeframe="1",
            mode="testnet",
            enriched=df,
            intent=intent,
        )
        self.assertIsNone(candidate)


if __name__ == "__main__":
    unittest.main()
