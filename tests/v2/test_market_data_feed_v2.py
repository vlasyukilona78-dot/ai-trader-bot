from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import pandas as pd

from trading.market_data.feed import MarketDataFeed
from trading.signals.runtime_source_adapter import build_runtime_signal_inputs


class _FakeMarketDataClient:
    def __init__(self):
        self.derivative_calls = 0

    def close(self):
        pass

    def fetch_ohlcv(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        index = pd.date_range("2026-01-01", periods=3, freq="1min", tz="UTC")
        return pd.DataFrame(
            {
                "open": [1.0, 1.01, 1.02],
                "high": [1.02, 1.03, 1.04],
                "low": [0.99, 1.0, 1.01],
                "close": [1.01, 1.02, 1.03],
                "volume": [10.0, 12.0, 14.0],
            },
            index=index,
        )

    def fetch_ticker_meta(self, symbol: str) -> dict[str, str]:
        return {
            "markPrice": "1.031",
            "fundingRate": "0.00042",
            "turnover24h": "2500000",
            "volume24h": "150000",
            "bid1Price": "1.030",
            "ask1Price": "1.032",
        }

    def fetch_long_short_ratio(self, symbol: str) -> float:
        self.derivative_calls += 1
        return 1.37

    def fetch_open_interest(self, symbol: str) -> float:
        self.derivative_calls += 1
        return 123456.0


class MarketDataFeedV2Tests(unittest.TestCase):
    def _feed(self, client: _FakeMarketDataClient) -> MarketDataFeed:
        feed = object.__new__(MarketDataFeed)
        feed._client = client
        return feed

    def test_fetch_frame_exposes_live_ticker_runtime_payload_without_extra_derivative_calls(self):
        client = _FakeMarketDataClient()
        with patch.dict(os.environ, {"MARKETDATA_FETCH_DERIVATIVE_CONTEXT": "0"}, clear=False):
            frame = self._feed(client).fetch_frame("TESTUSDT", "1", 3)

        payload = frame.runtime_payload or {}
        self.assertEqual(frame.mark_price, 1.031)
        self.assertEqual(payload.get("funding_rate"), 0.00042)
        self.assertEqual(payload.get("funding_source"), "live:bybit:ticker")
        self.assertEqual(payload.get("funding_degraded"), False)
        self.assertEqual(payload.get("turnover24h_usdt"), 2500000.0)
        self.assertAlmostEqual(float(payload.get("spread_bps")), 19.39864209505)
        self.assertNotIn("long_short_ratio", payload)
        self.assertEqual(client.derivative_calls, 0)

    def test_fetch_frame_can_opt_into_live_derivative_context(self):
        client = _FakeMarketDataClient()
        with patch.dict(os.environ, {"MARKETDATA_FETCH_DERIVATIVE_CONTEXT": "1"}, clear=False):
            frame = self._feed(client).fetch_frame("TESTUSDT", "1", 3)

        payload = frame.runtime_payload or {}
        self.assertEqual(payload.get("long_short_ratio"), 1.37)
        self.assertEqual(payload.get("long_short_ratio_source"), "live:bybit:account-ratio")
        self.assertEqual(payload.get("open_interest_abs"), 123456.0)
        self.assertEqual(payload.get("open_interest_abs_source"), "live:bybit:open-interest")
        self.assertNotIn("open_interest", payload)
        self.assertEqual(client.derivative_calls, 2)

    def test_runtime_source_adapter_preserves_market_quality_payload_for_risk(self):
        client = _FakeMarketDataClient()
        frame = self._feed(client).fetch_frame("TESTUSDT", "1", 3)

        runtime_inputs = build_runtime_signal_inputs(frame.ohlcv, runtime_payload=frame.runtime_payload)

        self.assertEqual(runtime_inputs.get("turnover24h_usdt"), 2500000.0)
        self.assertEqual(runtime_inputs.get("volume24h"), 150000.0)
        self.assertAlmostEqual(float(runtime_inputs.get("spread_bps")), 19.39864209505)


if __name__ == "__main__":
    unittest.main()
