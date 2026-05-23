from __future__ import annotations

import unittest
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from tests.v2.fakes import FakeAdapter
from trading.execution.engine import ExecutionEngine
from trading.exchange.schemas import AccountSnapshot, InstrumentRules
from trading.market_data.reconciliation import ExchangeSnapshot
from trading.risk.engine import RiskEngine
from trading.risk.limits import RiskLimits
from trading.signals.signal_types import IntentAction
from trading.signals.strategy_interface import StrategyContext
from trading.signals.ultra_short_entry import UltraShortConfig, UltraShortEntryDetector
from trading.signals.versioning import STRATEGY_RUNTIME_VERSION
from trading.state.machine import StateMachine
from trading.state.models import TradeState
from trading.state.persistence import RuntimeStore


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

    def test_downside_target_score_uses_real_targets_only(self):
        df = self._frame(vwap=115.0, poc=114.0, val=113.0)
        decision = self._detector().evaluate(
            self._context(df),
            trace_meta=self._trace(failed_reclaim=1.0, retest_failed_breakout=1.0, rejection_bar=1.0),
            liquidation_map={"swept_above": True, "downside_magnet": False, "upside_risk": 0.0},
        )

        self.assertFalse(decision.approved)
        self.assertEqual(decision.diagnostics["downside_target_score"], 0.0)
        self.assertEqual(decision.diagnostics["trade_plan"]["target_source"], "fixed_7pct_fallback")

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

    def test_ultra_intent_runs_through_risk_execution_and_persists_admission(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RuntimeStore(str(Path(tmpdir) / "runtime.db"))
            try:
                decision = self._detector().evaluate(
                    self._context(self._frame()),
                    trace_meta=self._trace(
                        failed_reclaim=1.0,
                        retest_failed_breakout=1.0,
                        rejection_bar=1.0,
                        near_sweep_level=1.0,
                    ),
                    liquidation_map=SimpleNamespace(swept_above=True, downside_magnet=True, upside_risk=0.0),
                )
                self.assertTrue(decision.approved, decision.reason)

                intent = decision.to_intent()
                intent.metadata = {
                    **intent.metadata,
                    "legacy_signal_id": decision.setup_signature,
                    "signal_side": "SHORT",
                    "entry_gate": {"approved": True, "reason": "approved", "score": decision.score},
                    "admission_status": "approved",
                    "admission_reason": "approved",
                }

                adapter = FakeAdapter(
                    rules=InstrumentRules(
                        symbol=decision.symbol,
                        tick_size=0.01,
                        qty_step=0.001,
                        min_qty=0.001,
                        min_notional=5.0,
                    )
                )
                adapter.mark_price = float(decision.entry)
                state_machine = StateMachine()
                state_machine.transition(decision.symbol, TradeState.FLAT, "init")
                risk = RiskEngine(
                    RiskLimits(
                        max_risk_per_trade_pct=0.01,
                        max_daily_loss_pct=0.05,
                        max_leverage=10.0,
                        max_concurrent_positions=2,
                        max_symbol_exposure_pct=2.0,
                        max_total_notional_pct=10.0,
                        min_liquidation_buffer_pct=0.0,
                        require_stop_loss=True,
                    ),
                    persistence=store,
                )
                execution = ExecutionEngine(
                    adapter=adapter,
                    state_machine=state_machine,
                    hedge_mode=False,
                    stop_loss_required=True,
                    require_reconciliation=True,
                    persistence=store,
                )

                risk_decision = risk.evaluate(
                    intent=intent,
                    account=adapter.get_account(),
                    existing_positions=adapter.get_positions(decision.symbol),
                    mark_price=float(decision.entry),
                    rules=adapter.get_instrument_rules(decision.symbol),
                )
                outcome = execution.execute(
                    intent=intent,
                    risk=risk_decision,
                    snapshot=ExchangeSnapshot(
                        symbol=decision.symbol,
                        account=adapter.get_account(),
                        positions=adapter.get_positions(decision.symbol),
                        open_orders=adapter.get_open_orders(decision.symbol),
                    ),
                    mark_price=float(decision.entry),
                )

                admissions = store.load_signal_admissions(limit=100)
                order_decisions = store.load_order_decisions(limit=100)

                self.assertTrue(risk_decision.approved, risk_decision.reason)
                self.assertTrue(outcome.accepted, outcome.reason)
                self.assertEqual(outcome.status, "FILLED")
                self.assertEqual(len(admissions), 1)
                self.assertEqual(admissions[0].signal_id, decision.setup_signature)
                self.assertEqual(admissions[0].raw["intent_reason"], "ultra_short_entry")
                self.assertEqual(admissions[0].raw["admission_status"], "approved")
                self.assertEqual(admissions[0].raw["execution_status"], "FILLED")
                self.assertEqual(len(order_decisions), 1)
                self.assertEqual(
                    order_decisions[0].raw["intent_context"]["metadata"]["signal_profile"],
                    "ultra",
                )
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
