from __future__ import annotations

import unittest
from unittest.mock import patch

from trading.exchange.bybit_ws import BybitWebSocketConfig, BybitWebSocketStream
from trading.exchange.events import ExchangeEventType


class BybitWebSocketStreamV2Tests(unittest.TestCase):
    def setUp(self):
        self.stream = BybitWebSocketStream(BybitWebSocketConfig(testnet=True, symbols=["BTCUSDT"]))

    def test_demo_private_endpoint_uses_demo_host_and_mainnet_public(self):
        stream = BybitWebSocketStream(
            BybitWebSocketConfig(testnet=False, demo=True, symbols=["BTCUSDT"])
        )
        self.assertEqual(stream._public_endpoint(), "wss://stream.bybit.com/v5/public/linear")
        self.assertEqual(stream._private_endpoint(), "wss://stream-demo.bybit.com/v5/private")

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

    def test_connection_loop_uses_configured_timeout_and_ping_settings(self):
        stream = BybitWebSocketStream(
            BybitWebSocketConfig(
                testnet=True,
                symbols=["BTCUSDT"],
                open_timeout_sec=11.0,
                close_timeout_sec=6.0,
                ping_interval_sec=35.0,
                ping_timeout_sec=18.0,
            )
        )
        captured: dict[str, object] = {}

        class _DummyWS:
            def send(self, _payload):
                return None

            def recv(self, timeout=1):
                stream._running = False
                stream._stop_evt.set()
                raise RuntimeError("stop_loop")

        class _DummyConnect:
            def __enter__(self):
                return _DummyWS()

            def __exit__(self, exc_type, exc, tb):
                return False

        def _fake_connect(url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return _DummyConnect()

        with patch("trading.exchange.bybit_ws.ws_connect", side_effect=_fake_connect), patch(
            "trading.exchange.bybit_ws.time.sleep",
            return_value=None,
        ):
            stream._running = True
            stream._stop_evt.clear()
            stream._connection_loop(channel="public", url=stream._public_endpoint(), is_private=False)

        self.assertEqual(captured.get("open_timeout"), 11.0)
        self.assertEqual(captured.get("close_timeout"), 6.0)
        self.assertEqual(captured.get("ping_interval"), 35.0)
        self.assertEqual(captured.get("ping_timeout"), 18.0)


if __name__ == "__main__":
    unittest.main()
