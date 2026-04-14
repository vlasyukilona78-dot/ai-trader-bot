from __future__ import annotations

import time
import unittest

from tests.v2.fakes import FakeAdapter
from trading.exchange.events import ExchangeEventType, NormalizedExchangeEvent
from trading.exchange.schemas import AccountSnapshot, OpenOrderSnapshot, OrderSide, PositionSide, PositionSnapshot
from trading.market_data.reconciliation import ExchangeReconciler
from trading.market_data.ws_reconciliation import ExchangeSyncService


class WebsocketReconciliationV2Tests(unittest.TestCase):
    def setUp(self):
        self.adapter = FakeAdapter()
        self.reconciler = ExchangeReconciler(self.adapter)
        self.sync = ExchangeSyncService(self.reconciler, poll_interval_sec=1, max_event_staleness_sec=2)

    def test_polling_fallback_when_ws_unavailable(self):
        self.adapter.positions = [
            PositionSnapshot(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                qty=1.0,
                entry_price=100.0,
                liq_price=0.0,
                leverage=1.0,
                position_idx=0,
            )
        ]
        snap = self.sync.snapshot("BTCUSDT")
        self.assertEqual(len(snap.positions), 1)
        health = self.sync.health()
        self.assertTrue(health.fallback_polling)

    def test_adapter_ws_metadata_marks_public_channel_fresh(self):
        now = time.time()
        self.adapter.ws_health_meta = {
            "running": True,
            "public_last_msg_ts": now,
            "private_last_msg_ts": now - 30,
        }
        self.sync.pull_adapter_events(self.adapter)
        health = self.sync.health()
        self.assertTrue(health.ws_connected)
        self.assertFalse(health.ws_stale)
        self.assertFalse(health.fallback_polling)

    def test_ws_snapshot_and_stale_order_cleanup(self):
        self.sync.handle_event(NormalizedExchangeEvent(event_type=ExchangeEventType.CONNECTED, ts=time.time()))
        self.sync.handle_event(
            NormalizedExchangeEvent(
                event_type=ExchangeEventType.ACCOUNT,
                payload={"account": AccountSnapshot(equity_usdt=1200.0, available_balance_usdt=1000.0)},
                ts=time.time(),
            )
        )

        position = PositionSnapshot(
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            qty=2.0,
            entry_price=101.0,
            liq_price=120.0,
            leverage=3.0,
            position_idx=2,
        )
        self.sync.handle_event(
            NormalizedExchangeEvent(
                event_type=ExchangeEventType.POSITION,
                symbol="BTCUSDT",
                payload={"position": position},
                ts=time.time(),
            )
        )

        order_new = OpenOrderSnapshot(
            symbol="BTCUSDT",
            order_id="1",
            order_link_id="x",
            side=OrderSide.BUY,
            qty=2.0,
            reduce_only=False,
            position_idx=2,
            status="New",
        )
        self.sync.handle_event(
            NormalizedExchangeEvent(
                event_type=ExchangeEventType.ORDER,
                symbol="BTCUSDT",
                payload={"order": order_new},
                ts=time.time(),
            )
        )

        snap = self.sync.snapshot("BTCUSDT")
        self.assertEqual(len(snap.positions), 1)
        self.assertEqual(len(snap.open_orders), 1)
        self.assertEqual(snap.positions[0].side, PositionSide.SHORT)

        order_filled = OpenOrderSnapshot(
            symbol="BTCUSDT",
            order_id="1",
            order_link_id="x",
            side=OrderSide.BUY,
            qty=2.0,
            reduce_only=False,
            position_idx=2,
            status="Filled",
        )
        self.sync.handle_event(
            NormalizedExchangeEvent(
                event_type=ExchangeEventType.ORDER,
                symbol="BTCUSDT",
                payload={"order": order_filled},
                ts=time.time(),
            )
        )
        snap2 = self.sync.snapshot("BTCUSDT")
        self.assertEqual(len(snap2.open_orders), 0)

    def test_zero_size_position_placeholder_clears_one_way_residue(self):
        self.sync.handle_event(NormalizedExchangeEvent(event_type=ExchangeEventType.CONNECTED, ts=time.time()))
        self.sync.handle_event(
            NormalizedExchangeEvent(
                event_type=ExchangeEventType.ACCOUNT,
                payload={"account": AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0)},
                ts=time.time(),
            )
        )

        self.sync.handle_event(
            NormalizedExchangeEvent(
                event_type=ExchangeEventType.POSITION,
                symbol="BTCUSDT",
                payload={
                    "position": PositionSnapshot(
                        symbol="BTCUSDT",
                        side=PositionSide.LONG,
                        qty=1.0,
                        entry_price=100.0,
                        liq_price=0.0,
                        leverage=1.0,
                        position_idx=0,
                    )
                },
                ts=time.time(),
            )
        )

        self.sync.handle_event(
            NormalizedExchangeEvent(
                event_type=ExchangeEventType.POSITION,
                symbol="BTCUSDT",
                payload={
                    "position": PositionSnapshot(
                        symbol="BTCUSDT",
                        side=PositionSide.SHORT,
                        qty=0.0,
                        entry_price=0.0,
                        liq_price=0.0,
                        leverage=0.0,
                        position_idx=0,
                    ),
                    "side_raw": "",
                },
                ts=time.time(),
            )
        )

        snap = self.sync.snapshot("BTCUSDT")
        self.assertEqual(len(snap.positions), 0)

    def test_position_list_snapshot_filters_zero_size_placeholders(self):
        self.sync.handle_event(NormalizedExchangeEvent(event_type=ExchangeEventType.CONNECTED, ts=time.time()))
        self.sync.handle_event(
            NormalizedExchangeEvent(
                event_type=ExchangeEventType.ACCOUNT,
                payload={"account": AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0)},
                ts=time.time(),
            )
        )

        self.sync.handle_event(
            NormalizedExchangeEvent(
                event_type=ExchangeEventType.POSITION,
                symbol="BTCUSDT",
                payload={
                    "positions": [
                        PositionSnapshot(
                            symbol="BTCUSDT",
                            side=PositionSide.SHORT,
                            qty=0.0,
                            entry_price=0.0,
                            liq_price=0.0,
                            leverage=0.0,
                            position_idx=0,
                        )
                    ]
                },
                ts=time.time(),
            )
        )

        snap = self.sync.snapshot("BTCUSDT")
        self.assertEqual(len(snap.positions), 0)

    def test_ws_stale_reconnect_falls_back_to_polling(self):
        self.sync.handle_event(NormalizedExchangeEvent(event_type=ExchangeEventType.CONNECTED, ts=time.time() - 20))
        self.sync.handle_event(
            NormalizedExchangeEvent(
                event_type=ExchangeEventType.ACCOUNT,
                payload={"account": AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0)},
                ts=time.time() - 20,
            )
        )
        self.adapter.positions = [
            PositionSnapshot(
                symbol="ETHUSDT",
                side=PositionSide.LONG,
                qty=1.0,
                entry_price=50.0,
                liq_price=0.0,
                leverage=1.0,
                position_idx=0,
            )
        ]

        snap = self.sync.snapshot("ETHUSDT")
        self.assertEqual(len(snap.positions), 1)
        health = self.sync.health()
        self.assertTrue(health.ws_stale)
        self.assertTrue(health.fallback_polling)

    def test_snapshot_required_forces_exchange_poll(self):
        self.sync.handle_event(NormalizedExchangeEvent(event_type=ExchangeEventType.CONNECTED, ts=time.time()))
        self.sync.handle_event(
            NormalizedExchangeEvent(
                event_type=ExchangeEventType.ACCOUNT,
                payload={"account": AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0)},
                ts=time.time(),
            )
        )
        self.sync.handle_event(
            NormalizedExchangeEvent(
                event_type=ExchangeEventType.POSITION,
                symbol="BTCUSDT",
                payload={
                    "position": PositionSnapshot(
                        symbol="BTCUSDT",
                        side=PositionSide.SHORT,
                        qty=1.0,
                        entry_price=99.0,
                        liq_price=130.0,
                        leverage=3.0,
                        position_idx=2,
                    )
                },
                ts=time.time(),
            )
        )

        self.adapter.positions = [
            PositionSnapshot(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                qty=2.0,
                entry_price=101.0,
                liq_price=80.0,
                leverage=2.0,
                position_idx=1,
            )
        ]
        self.sync.handle_event(
            NormalizedExchangeEvent(
                event_type=ExchangeEventType.SNAPSHOT_REQUIRED,
                symbol="BTCUSDT",
                payload={"reason": "ws_gap"},
                ts=time.time(),
            )
        )

        snap = self.sync.snapshot("BTCUSDT")
        self.assertEqual(len(snap.positions), 1)
        self.assertEqual(snap.positions[0].side, PositionSide.LONG)
        health = self.sync.health()
        self.assertFalse(health.snapshot_required)

    def test_reconnecting_event_triggers_fallback_polling(self):
        self.sync.handle_event(NormalizedExchangeEvent(event_type=ExchangeEventType.RECONNECTING, ts=time.time()))
        self.assertTrue(self.sync.health().fallback_polling)

    def test_stale_ws_triggers_forced_reconnect(self):
        calls: list[str] = []

        def _force_reconnect():
            calls.append("reconnect")

        self.adapter.force_ws_reconnect = _force_reconnect
        self.sync.handle_event(NormalizedExchangeEvent(event_type=ExchangeEventType.CONNECTED, ts=time.time() - 10))
        self.sync.handle_event(
            NormalizedExchangeEvent(
                event_type=ExchangeEventType.ACCOUNT,
                payload={"account": AccountSnapshot(equity_usdt=1000.0, available_balance_usdt=1000.0)},
                ts=time.time() - 10,
            )
        )

        reason = self.sync.maybe_recover_ws(self.adapter)

        self.assertEqual(reason, "stale")
        self.assertEqual(calls, ["reconnect"])

    def test_public_market_event_marks_public_ws_connected_and_fresh(self):
        self.sync.handle_event(
            NormalizedExchangeEvent(
                event_type=ExchangeEventType.MARKET,
                symbol="BTCUSDT",
                payload={"channel": "public", "mark_price": 101.0},
                ts=time.time(),
            )
        )

        health = self.sync.health()

        self.assertTrue(health.ws_connected)
        self.assertFalse(health.ws_stale)
        self.assertFalse(health.fallback_polling)


if __name__ == "__main__":
    unittest.main()
