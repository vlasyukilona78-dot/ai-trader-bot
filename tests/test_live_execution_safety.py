import os
import unittest

if os.getenv("ALLOW_LEGACY_RUNTIME", "false").strip().lower() not in ("1", "true", "yes"):
    raise unittest.SkipTest("legacy runtime tests are quarantined; run V2 tests instead")
import threading
import time
import unittest
from dataclasses import dataclass

from core.execution import ExecutionEngine
from core.risk_engine import RiskConfig, RiskEngine


@dataclass
class FakeSignal:
    signal_id: str
    symbol: str
    side: str
    entry: float
    tp: float
    sl: float


class FakeBybitClient:
    def __init__(
        self,
        *,
        positions: list[dict] | None = None,
        open_orders: list[dict] | None = None,
        stop_ok: bool = True,
        order_delay_sec: float = 0.0,
    ):
        self._positions = positions or []
        self._open_orders = open_orders or []
        self._stop_ok = stop_ok
        self._order_delay_sec = float(order_delay_sec)
        self.order_calls: list[dict] = []
        self.stop_calls: list[dict] = []
        self._lock = threading.Lock()

    def get_open_positions(self, symbol: str | None = None) -> list[dict]:
        if symbol is None:
            return list(self._positions)
        target = symbol.replace("/", "").upper()
        return [p for p in self._positions if str(p.get("symbol", "")).replace("/", "").upper() == target]

    def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        if symbol is None:
            return list(self._open_orders)
        target = symbol.replace("/", "").upper()
        return [o for o in self._open_orders if str(o.get("symbol", "")).replace("/", "").upper() == target]

    def place_order_market(
        self,
        symbol: str,
        side: str,
        qty: float,
        *,
        reduce_only: bool = False,
        position_idx: int | None = None,
        order_link_id: str | None = None,
        close_on_trigger: bool | None = None,
    ) -> dict:
        if self._order_delay_sec > 0:
            time.sleep(self._order_delay_sec)
        with self._lock:
            self.order_calls.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "reduce_only": reduce_only,
                    "position_idx": position_idx,
                    "order_link_id": order_link_id,
                    "close_on_trigger": close_on_trigger,
                }
            )
            oid = f"oid-{len(self.order_calls)}"
        return {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "orderId": oid,
                "orderLinkId": order_link_id,
                "avgPrice": "100",
            },
        }

    def set_trading_stop(
        self,
        symbol: str,
        *,
        stop_loss: float,
        take_profit: float | None = None,
        position_idx: int | None = None,
    ) -> dict:
        self.stop_calls.append(
            {
                "symbol": symbol,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "position_idx": position_idx,
            }
        )
        if self._stop_ok:
            return {"retCode": 0, "retMsg": "OK", "result": {}}
        return {"retCode": 10001, "retMsg": "stop failed", "result": {}}


class LiveExecutionSafetyTests(unittest.TestCase):
    def test_duplicate_order_blocked(self):
        client = FakeBybitClient()
        engine = ExecutionEngine(bybit_client=client, dry_run=False, duplicate_window_sec=60)
        signal = FakeSignal("sig-1", "BTCUSDT", "SHORT", 100.0, 98.0, 101.0)

        first = engine.execute(signal, qty=0.5)
        second = engine.execute(signal, qty=0.5)

        self.assertTrue(first.success)
        self.assertFalse(second.success)
        self.assertEqual(second.error, "duplicate_order_blocked")
        self.assertEqual(len(client.order_calls), 1)

    def test_blocks_when_exchange_already_has_position(self):
        client = FakeBybitClient(positions=[{"symbol": "BTCUSDT", "side": "LONG", "size": 0.3}])
        engine = ExecutionEngine(bybit_client=client, dry_run=False)
        signal = FakeSignal("sig-2", "BTCUSDT", "LONG", 100.0, 103.0, 99.0)

        res = engine.execute(signal, qty=0.2)

        self.assertFalse(res.success)
        self.assertEqual(res.error, "exchange_position_exists")
        self.assertEqual(len(client.order_calls), 0)

    def test_reduce_only_and_position_idx_are_correct(self):
        client = FakeBybitClient(stop_ok=False)
        engine = ExecutionEngine(bybit_client=client, dry_run=False, hedge_mode=True)
        signal = FakeSignal("sig-3", "ETHUSDT", "SHORT", 100.0, 96.0, 102.0)

        res = engine.execute(signal, qty=1.0)

        self.assertFalse(res.success)
        self.assertEqual(res.error, "stop_loss_set_failed")
        self.assertEqual(len(client.order_calls), 2)
        open_call = client.order_calls[0]
        emergency_call = client.order_calls[1]
        self.assertEqual(open_call["side"], "sell")
        self.assertFalse(open_call["reduce_only"])
        self.assertEqual(open_call["position_idx"], 2)
        self.assertEqual(emergency_call["side"], "buy")
        self.assertTrue(emergency_call["reduce_only"])
        self.assertEqual(emergency_call["position_idx"], 2)
        self.assertEqual(len(client.stop_calls), 1)
        self.assertEqual(client.stop_calls[0]["position_idx"], 2)

    def test_risk_engine_sync_and_symbol_lock(self):
        engine = RiskEngine(RiskConfig(account_equity_usdt=1000.0))
        engine.register_open_position(
            "local-1",
            symbol="XRPUSDT",
            side="LONG",
            qty=10.0,
            entry=2.0,
            sl=1.8,
            source="local",
        )
        engine.sync_exchange_positions(
            [
                {
                    "symbol": "ETHUSDT",
                    "side": "SHORT",
                    "size": 1.0,
                    "entryPrice": 2000.0,
                    "positionIdx": 2,
                }
            ]
        )

        self.assertTrue(engine.has_open_symbol("ETHUSDT"))
        self.assertGreater(engine.current_open_exposure(), 0.0)

        sizing = engine.evaluate_order(
            signal_id="sig-4",
            side="SHORT",
            entry=1990.0,
            sl=2020.0,
            symbol="ETHUSDT",
        )
        self.assertFalse(sizing.approved)
        self.assertEqual(sizing.reason, "symbol_already_open")

    def test_risk_engine_rejects_tight_stop_and_caps_size(self):
        cfg = RiskConfig(
            account_equity_usdt=1000.0,
            max_risk_per_trade=0.02,
            min_qty=0.001,
            max_qty=100.0,
            max_notional_per_trade_pct=0.05,
            max_leverage=1.0,
            min_stop_distance_pct=0.002,
            min_liquidation_buffer_pct=0.003,
        )
        engine = RiskEngine(cfg)

        tight = engine.evaluate_order(
            signal_id="sig-5",
            side="LONG",
            entry=100.0,
            sl=99.95,
            symbol="SOLUSDT",
        )
        self.assertFalse(tight.approved)
        self.assertIn(tight.reason, {"stop_too_close", "liq_buffer_too_small"})

        capped = engine.evaluate_order(
            signal_id="sig-6",
            side="LONG",
            entry=100.0,
            sl=99.0,
            symbol="SOLUSDT",
        )
        self.assertTrue(capped.approved)
        self.assertLessEqual(capped.qty * capped.expected_fill, 50.0 + 1e-9)

    def test_concurrent_execute_is_single_flight(self):
        client = FakeBybitClient(order_delay_sec=0.15)
        engine = ExecutionEngine(bybit_client=client, dry_run=False, duplicate_window_sec=60)
        signal = FakeSignal("sig-7", "ADAUSDT", "LONG", 100.0, 104.0, 98.0)

        start_evt = threading.Event()
        results = []
        lock = threading.Lock()

        def worker():
            start_evt.wait()
            res = engine.execute(signal, qty=0.4)
            with lock:
                results.append(res)

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        start_evt.set()
        t1.join(timeout=3)
        t2.join(timeout=3)

        self.assertEqual(len(results), 2)
        self.assertEqual(len(client.order_calls), 1)
        success_count = sum(1 for r in results if r.success)
        self.assertEqual(success_count, 1)
        failure_errors = {r.error for r in results if not r.success}
        self.assertTrue(
            failure_errors.issubset({"symbol_execution_inflight", "duplicate_order_blocked"}),
            failure_errors,
        )


if __name__ == "__main__":
    unittest.main()

