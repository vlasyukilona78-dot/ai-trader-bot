from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from trading.market_data.bar_contract import (
    BarContractError,
    interval_seconds,
    last_bar_times,
    retain_closed_bars,
)
from trading.market_data.feed import MarketDataFeed
from trading.market_data.mexc_client import MexcContractClient


def _bars() -> pd.DataFrame:
    index = pd.date_range("2026-01-01T10:00:00Z", periods=4, freq="h")
    frame = pd.DataFrame({"close": [10.0, 11.0, 12.0, 13.0]}, index=index)
    frame.attrs["source"] = "fixture"
    return frame


class ClosedBarContractV2Tests(unittest.TestCase):
    def test_forming_bar_is_excluded(self):
        closed = retain_closed_bars(
            _bars(), interval="Min60", as_of=pd.Timestamp("2026-01-01T12:30:00Z")
        )
        self.assertEqual(list(closed.index.hour), [10, 11])
        self.assertEqual(
            closed.attrs["candle_cutoff_ts"],
            pd.Timestamp("2026-01-01T12:00:00Z").timestamp(),
        )
        self.assertEqual(
            closed.attrs["last_bar_open_ts"],
            pd.Timestamp("2026-01-01T11:00:00Z").timestamp(),
        )
        self.assertEqual(
            closed.attrs["last_bar_close_ts"],
            pd.Timestamp("2026-01-01T12:00:00Z").timestamp(),
        )
        self.assertEqual(closed.attrs["source"], "fixture")

    def test_bar_closing_exactly_at_as_of_is_included(self):
        closed = retain_closed_bars(
            _bars(), interval="60", as_of=pd.Timestamp("2026-01-01T13:00:00Z")
        )
        self.assertEqual(list(closed.index.hour), [10, 11, 12])

    def test_generic_and_mexc_interval_names_are_equivalent(self):
        self.assertEqual(interval_seconds("60"), interval_seconds("Min60"))
        generic = retain_closed_bars(
            _bars(), interval="60", as_of=pd.Timestamp("2026-01-01T12:30:00Z")
        )
        mexc = retain_closed_bars(
            _bars(), interval="Min60", as_of=pd.Timestamp("2026-01-01T12:30:00Z")
        )
        pd.testing.assert_frame_equal(generic, mexc, check_flags=False)

    def test_naive_as_of_fails_closed(self):
        with self.assertRaises(BarContractError):
            retain_closed_bars(_bars(), interval="Min60", as_of=pd.Timestamp("2026-01-01"))

    def test_non_datetime_index_fails_closed(self):
        with self.assertRaises(BarContractError):
            retain_closed_bars(
                pd.DataFrame({"close": [1.0]}),
                interval="Min60",
                as_of=pd.Timestamp("2026-01-01T12:00:00Z"),
            )

    def test_naive_datetime_index_fails_closed(self):
        frame = _bars()
        frame.index = frame.index.tz_localize(None)
        with self.assertRaises(BarContractError):
            retain_closed_bars(
                frame,
                interval="Min60",
                as_of=pd.Timestamp("2026-01-01T12:00:00Z"),
            )

    def test_calendar_month_interval_is_rejected(self):
        with self.assertRaises(BarContractError):
            interval_seconds("Month1")

    def test_last_bar_times_use_the_fixed_interval(self):
        open_ts, close_ts = last_bar_times(_bars().iloc[:2], interval="Min60")
        self.assertEqual(open_ts, pd.Timestamp("2026-01-01T11:00:00Z").timestamp())
        self.assertEqual(close_ts, pd.Timestamp("2026-01-01T12:00:00Z").timestamp())


class _Client:
    def __init__(self):
        self.requests = []
        self.ticker_requests = []

    def fetch_ohlcv(self, **kwargs):
        self.requests.append(kwargs)
        return _bars()

    def fetch_ticker_meta(self, symbol: str):
        self.ticker_requests.append(symbol)
        return {"lastPrice": "123.5"}

    def close(self):
        return None


class ClosedMarketFrameV2Tests(unittest.TestCase):
    def test_feed_requests_extra_bar_and_exposes_cutoff_metadata(self):
        client = _Client()
        frame = MarketDataFeed(client=client).fetch_closed_frame(
            "BTCUSDT",
            "Min60",
            2,
            as_of=pd.Timestamp("2026-01-01T13:30:00Z"),
        )
        self.assertEqual(client.requests[0]["limit"], 3)
        self.assertEqual(list(frame.ohlcv.index.hour), [11, 12])
        self.assertEqual(frame.mark_price, float(frame.ohlcv.iloc[-1]["close"]))
        self.assertEqual(client.ticker_requests, [])
        self.assertEqual(
            frame.candle_cutoff_ts,
            pd.Timestamp("2026-01-01T13:00:00Z").timestamp(),
        )
        self.assertEqual(
            frame.last_bar_close_ts,
            pd.Timestamp("2026-01-01T13:00:00Z").timestamp(),
        )

    def test_empty_mexc_response_keeps_timezone_aware_bar_index(self):
        client = MexcContractClient()
        with patch.object(client, "_request_public", return_value={"data": {}}):
            frame = client.fetch_ohlcv("BTCUSDT", "60", 10)
        client.close()

        self.assertTrue(frame.empty)
        self.assertIsInstance(frame.index, pd.DatetimeIndex)
        self.assertIsNotNone(frame.index.tz)


if __name__ == "__main__":
    unittest.main()
