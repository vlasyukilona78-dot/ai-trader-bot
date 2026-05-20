from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd

from trading.execution.engine import ExecutionEngine
from trading.exchange.schemas import AccountSnapshot
from trading.market_data.reconciliation import ExchangeSnapshot
from trading.signals.signal_types import IntentAction
from trading.signals.strategy_interface import StrategyContext
from trading.signals.ultra_short_entry import UltraShortConfig, UltraShortEntryDetector
from trading.signals.versioning import STRATEGY_RUNTIME_VERSION
from trading.state.models import TradeState


class UltraShortEntryV2Tests(unittest.TestCase):
    def _exchange(self, symbol: str = "PUMPUSDT") -> ExchangeSnapshot:
        return ExchangeSnapshot(
            symbol=symbol,
            account=AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0),
            positions=[],
            open_orders=[],
        )

    def _trace(self, **flags: float) -> dict[str, object]:
        return {
            "layer_trace": {
                "layers": {
                    "layer2_weakness_confirmation": {
                        "details": {
                            "failed_reclaim": flags.get("failed_reclaim", 0.0),
                            "retest_failed_breakout": flags.get("retest_failed_breakout", 0.0),
                        }
                    },
                    "layer3_entry_location": {
                        "details": {
                            "acceptance_above_swing_high": flags.get("acceptance_above_high", 0.0),
                            "rejection_bar": flags.get("rejection_bar", 0.0),
                            "near_sweep_level": flags.get("near_sweep_level", 0.0),
                        }
                    },
                }
            }
        }

    def _frame(self, **last_overrides) -> pd.DataFrame:
        rows: list[dict[str, float]] = []
        close = 100.0
        for idx in range(40):
            close += 0.18
            rows.append(
                {
                    "open": close - 0.08,
                    "high": close + 0.18,
                    "low": close - 0.28,
                    "close": close,
                    "volume": 1000.0 + idx * 10,
                    "atr": 1.0,
                    "rsi": 58.0 + idx * 0.20,
                    "hist": 0.010 + idx * 0.001,
                    "volume_spike": 1.05,
                    "vwap": 104.8,
                    "vah": 109.7,
                    "poc": 105.0,
                    "val": 102.5,
                    "obv": 10_000.0 + idx * 20,
                    "cvd": 5_000.0 + idx * 10,
                    "mtf_trend_5m": 0.0,
                    "mtf_trend_15m": 0.0,
                    "mtf_rsi_5m": 52.0,
                    "mtf_rsi_15m": 51.0,
                    "clean_pump_pct": 0.044,
                }
            )
        rows[-2].update(
            {
                "open": 108.9,
                "high": 110.0,
                "low": 108.5,
                "close": 109.8,
                "rsi": 72.0,
                "hist": 0.060,
                "volume_spike": 2.20,
                "obv": 11_000.0,
                "cvd": 5_500.0,
            }
        )
        rows[-1].update(
            {
                "open": 109.8,
                "high": 110.2,
                "low": 108.2,
                "close": 109.4,
                "rsi": 68.0,
                "hist": 0.034,
                "volume_spike": 1.38,
                "obv": 10_900.0,
                "cvd": 5_420.0,
                "clean_pump_pct": 0.044,
            }
        )
        rows[-1].update(last_overrides)
        idx = pd.date_range("2026-01-01", periods=len(rows), freq="min", tz="UTC")
        return pd.DataFrame(rows, index=idx)

    def _context(self, df: pd.DataFrame, symbol: str = "PUMPUSDT") -> StrategyContext:
        return StrategyContext(
            symbol=symbol,
            market_ohlcv=df,
            mark_price=float(df.iloc[-1]["close"]),
            exchange=self._exchange(symbol),
            synced_state=TradeState.FLAT,
            timeframe="1m",
        )

    def _detector(self, **config_overrides) -> UltraShortEntryDetector:
        return UltraShortEntryDetector(UltraShortConfig(**config_overrides))

    def test_rejects_insufficient_history(self):
        df = self._frame().tail(5)
        decision = self._detector().evaluate(self._context(df))
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "insufficient_history")

    def test_rejects_weak_pump(self):
        df = self._frame(clean_pump_pct=0.012)
        decision = self._detector().evaluate(self._context(df))
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "clean_pump_below_min")

    def test_rejects_when_entry_too_far_from_high(self):
        df = self._frame(close=108.0)
        decision = self._detector().evaluate(self._context(df))
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "entry_too_far_from_high")

    def test_rejects_acceptance_above_high(self):
        decision = self._detector().evaluate(
            self._context(self._frame()),
            trace_meta=self._trace(acceptance_above_high=1.0),
        )
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "acceptance_above_high")

    def test_approves_sweep_failure_short_with_correct_metadata(self):
        decision = self._detector().evaluate(
            self._context(self._frame()),
            trace_meta=self._trace(
                failed_reclaim=1.0,
                retest_failed_breakout=1.0,
                rejection_bar=1.0,
                near_sweep_level=1.0,
            ),
            liquidation_map=SimpleNamespace(swept_above=True, downside_magnet=True, upside_risk=False),
        )
        self.assertTrue(decision.approved, decision.reason)
        self.assertEqual(decision.scenario, "sweep_failure_short")
        self.assertIn(decision.grade, {"A", "A+"})

        intent = decision.to_intent()
        self.assertEqual(intent.action, IntentAction.SHORT_ENTRY)
        self.assertEqual(intent.reason, "ultra_short_entry")
        self.assertEqual(intent.metadata["signal_profile"], "ultra")
        self.assertEqual(intent.metadata["ultra_scenario"], "sweep_failure_short")
        self.assertEqual(intent.metadata["timeframe"], "1m")
        self.assertEqual(intent.metadata["strategy_version"], STRATEGY_RUNTIME_VERSION)
        self.assertEqual(intent.metadata["setup_signature"], decision.setup_signature)
        self.assertIn("runtime_versions", intent.metadata)

    def test_approves_blowoff_rejection_short(self):
        df = self._frame(
            open=110.4,
            high=111.0,
            low=109.4,
            close=109.9,
            rsi=66.0,
            hist=0.030,
            volume_spike=1.65,
            clean_pump_pct=0.052,
        )
        decision = self._detector().evaluate(
            self._context(df),
            trace_meta=self._trace(rejection_bar=1.0, near_sweep_level=1.0),
            liquidation_map={"swept_above": True, "downside_magnet": True},
        )
        self.assertTrue(decision.approved, decision.reason)
        self.assertEqual(decision.scenario, "blowoff_rejection_short")

    def test_approves_failed_reclaim_short(self):
        df = self._frame(high=110.0, clean_pump_pct=0.038)
        df.iloc[-4, df.columns.get_loc("high")] = 110.2
        decision = self._detector().evaluate(
            self._context(df),
            trace_meta=self._trace(
                failed_reclaim=1.0,
                retest_failed_breakout=1.0,
                rejection_bar=1.0,
                near_sweep_level=1.0,
            ),
            liquidation_map={"swept_above": True, "downside_magnet": True},
        )
        self.assertTrue(decision.approved, decision.reason)
        self.assertEqual(decision.scenario, "failed_reclaim_short")

    def test_rejects_if_failed_acceptance_too_weak(self):
        df = self._frame(open=110.4, high=111.0, low=109.4, close=109.9, clean_pump_pct=0.052)
        decision = self._detector().evaluate(
            self._context(df),
            liquidation_map={"swept_above": True, "downside_magnet": True},
        )
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "failed_acceptance_too_weak")

    def test_rejects_if_continuation_risk_too_high(self):
        df = self._frame(
            open=108.8,
            high=110.2,
            low=108.6,
            close=109.9,
            rsi=71.0,
            hist=0.030,
            mtf_trend_5m=0.006,
            mtf_trend_15m=0.005,
            mtf_rsi_5m=66.0,
            mtf_rsi_15m=64.0,
        )
        decision = self._detector().evaluate(
            self._context(df),
            trace_meta=self._trace(failed_reclaim=1.0, retest_failed_breakout=1.0, rejection_bar=1.0),
            liquidation_map={"upside_risk": True, "downside_magnet": True},
        )
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "continuation_risk_too_high")

    def test_rejects_if_rr_too_low(self):
        decision = self._detector(min_rr=20.0).evaluate(
            self._context(self._frame()),
            trace_meta=self._trace(failed_reclaim=1.0, retest_failed_breakout=1.0, rejection_bar=1.0),
            liquidation_map={"swept_above": True, "downside_magnet": True},
        )
        self.assertFalse(decision.approved)
        self.assertEqual(decision.reason, "rr_below_min")

    def test_to_intent_includes_idempotency_metadata(self):
        decision = self._detector().evaluate(
            self._context(self._frame()),
            trace_meta=self._trace(failed_reclaim=1.0, retest_failed_breakout=1.0, rejection_bar=1.0),
            liquidation_map={"swept_above": True, "downside_magnet": True},
        )
        intent = decision.to_intent()
        self.assertEqual(intent.metadata["signal_profile"], "ultra")
        self.assertEqual(intent.metadata["ultra_scenario"], decision.scenario)
        self.assertEqual(intent.metadata["ultra_grade"], decision.grade)
        self.assertEqual(intent.metadata["ultra_score"], decision.score)
        self.assertEqual(intent.metadata["setup_signature"], decision.setup_signature)
        self.assertEqual(intent.metadata["timeframe"], "1m")
        self.assertEqual(intent.metadata["strategy_version"], STRATEGY_RUNTIME_VERSION)
        self.assertIn("runtime_versions", intent.metadata)

    def test_idempotency_compatibility(self):
        decision = self._detector().evaluate(
            self._context(self._frame()),
            trace_meta=self._trace(failed_reclaim=1.0, retest_failed_breakout=1.0, rejection_bar=1.0),
            liquidation_map={"swept_above": True, "downside_magnet": True},
        )
        first = decision.to_intent()
        second = replace(first, metadata={**first.metadata, "setup_signature": "different-ultra-setup"})

        first_key = ExecutionEngine._idempotency_key(first)
        self.assertEqual(first_key, ExecutionEngine._idempotency_key(first))
        self.assertNotEqual(first_key, ExecutionEngine._idempotency_key(second))


if __name__ == "__main__":
    unittest.main()
