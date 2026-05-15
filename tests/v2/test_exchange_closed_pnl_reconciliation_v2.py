from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from app.main import _reconcile_exchange_closed_position
from tests.v2.fakes import FakeAdapter
from trading.exchange.schemas import ClosedPnlSnapshot, PositionSide
from trading.market_data.reconciliation import ExchangeSnapshot
from trading.risk.engine import RiskEngine
from trading.risk.limits import RiskLimits
from trading.state.models import TradeState
from trading.state.persistence import RuntimeStore


class _Counter:
    def __init__(self):
        self.values: dict[str, int] = {}

    def inc(self, key: str, amount: int = 1):
        self.values[key] = self.values.get(key, 0) + int(amount)


class _TradeLearner:
    def __init__(self):
        self.exits: list[dict[str, float | str]] = []

    def record_exit(self, **kwargs):
        self.exits.append(dict(kwargs))
        return {"target_win": 0, "future_return": -0.012, "target_horizon": "exchange_closed_pnl"}


class _OnlineRetrainer:
    def __init__(self):
        self.calls = 0

    def maybe_retrain(self) -> bool:
        self.calls += 1
        return False


class ExchangeClosedPnlReconciliationV2Tests(unittest.TestCase):
    def test_exchange_closed_pnl_is_recorded_once_and_updates_learning(self):
        now = time.time()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RuntimeStore(str(Path(tmpdir) / "runtime.db"))
            risk = RiskEngine(RiskLimits(cooldown_after_stop_sec=0), persistence=store)
            adapter = FakeAdapter()
            adapter.closed_pnl_snapshots = [
                ClosedPnlSnapshot(
                    closure_id="btc-close-1",
                    symbol="BTCUSDT",
                    position_side=PositionSide.SHORT,
                    qty=0.25,
                    entry_price=101.0,
                    exit_price=99.0,
                    closed_pnl=4.2,
                    closed_ts=now - 5,
                    raw={"orderId": "btc-close-1"},
                )
            ]
            learner = _TradeLearner()
            retrainer = _OnlineRetrainer()
            counters = _Counter()
            snapshot = ExchangeSnapshot(
                symbol="BTCUSDT",
                account=adapter.get_account(),
                positions=[],
                open_orders=[],
            )

            first = _reconcile_exchange_closed_position(
                symbol="BTCUSDT",
                previous_state=TradeState.SHORT,
                previous_updated_at=now - 60,
                snapshot=snapshot,
                adapter=adapter,
                risk=risk,
                runtime_store=store,
                trade_learner=learner,
                online_retrainer=retrainer,
                counters=counters,
                mode="demo",
            )
            second = _reconcile_exchange_closed_position(
                symbol="BTCUSDT",
                previous_state=TradeState.SHORT,
                previous_updated_at=now - 60,
                snapshot=snapshot,
                adapter=adapter,
                risk=risk,
                runtime_store=store,
                trade_learner=learner,
                online_retrainer=retrainer,
                counters=counters,
                mode="demo",
            )

            self.assertTrue(first)
            self.assertFalse(second)
            self.assertAlmostEqual(risk.health_snapshot()["realized_pnl_today"], 4.2)
            self.assertEqual(len(store.load_exchange_closures(symbol="BTCUSDT")), 1)
            self.assertEqual(len(learner.exits), 1)
            self.assertEqual(learner.exits[0]["symbol"], "BTCUSDT")
            self.assertAlmostEqual(float(learner.exits[0]["realized_pnl"]), 4.2)
            self.assertEqual(retrainer.calls, 1)
            self.assertEqual(counters.values.get("exchange_closed_reconciled"), 1)
            store.close()


if __name__ == "__main__":
    unittest.main()
