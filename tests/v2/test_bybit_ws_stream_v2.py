from __future__ import annotations

import unittest

from trading.exchange.bybit_ws import BybitWebSocketConfig, BybitWebSocketStream
from trading.exchange.events import ExchangeEventType


class BybitWebSocketStreamV2Tests(unittest.TestCase):
    def setUp(self):
        self.stream = BybitWebSocketStream(BybitWebSocketConfig(testnet=True, symbols=["BTCUSDT"]))

    def test_normalize_public_ticker(self):
        payload = {
            "topic": "tickers.BTCUSDT",
            "type": "delta",
            "data": [{"markPrice": "101.5", "lastPrice": "101.7"}],
        }
        events = list(self.stream._normalize_message(channel="public", payload=payload))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, ExchangeEventType.MARKET)
        self.assertEqual(events[0].symbol, "BTCUSDT")
        self.assertAlmostEqual(float(events[0].payload.get("mark_price", 0.0)), 101.5, places=6)

    def test_normalize_private_order(self):
        payload = {
            "topic": "order",
            "data": [
                {
                    "symbol": "BTCUSDT",
                    "orderId": "o1",
                    "orderLinkId": "cid1",
                    "side": "Buy",
                    "leavesQty": "0.4",
                    "reduceOnly": False,
                    "positionIdx": 1,
                    "orderStatus": "PartiallyFilled",
                    "createdTime": "1700000000000",
                    "updatedTime": "1700000001000",
                }
            ],
        }
        events = list(self.stream._normalize_message(channel="private", payload=payload))
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.event_type, ExchangeEventType.ORDER)
        order = event.payload.get("order")
        self.assertIsNotNone(order)
        self.assertEqual(order.order_id, "o1")
        self.assertAlmostEqual(float(order.qty), 0.4, places=6)

    def test_auth_failure_emits_error_and_snapshot_required(self):
        payload = {"op": "auth", "success": False, "retMsg": "auth failed"}
        events = list(self.stream._normalize_message(channel="private", payload=payload))
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].event_type, ExchangeEventType.ERROR)
        self.assertEqual(events[1].event_type, ExchangeEventType.SNAPSHOT_REQUIRED)


if __name__ == "__main__":
    unittest.main()
