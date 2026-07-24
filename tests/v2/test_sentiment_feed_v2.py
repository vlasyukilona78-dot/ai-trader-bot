from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from trading.market_data.sentiment import SentimentFeed, SentimentFeedConfig


def _mock_response(value: str):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"data": [{"value": value}]})
    return resp


class SentimentFeedV2Tests(unittest.TestCase):
    def test_live_fetch_returns_value_and_caches(self):
        feed = SentimentFeed(SentimentFeedConfig(refresh_sec=60))
        with patch.object(feed._session, "get", return_value=_mock_response("22")) as mock_get:
            value, source = feed.get_sentiment(now=1000.0)
            self.assertEqual(value, 22.0)
            self.assertEqual(source, "live_alternative_me")
            self.assertEqual(mock_get.call_count, 1)

            # second call within refresh window must not hit the network
            value2, source2 = feed.get_sentiment(now=1010.0)
            self.assertEqual(value2, 22.0)
            self.assertEqual(source2, "cache_fresh")
            self.assertEqual(mock_get.call_count, 1)

    def test_refetches_after_refresh_window(self):
        feed = SentimentFeed(SentimentFeedConfig(refresh_sec=60))
        with patch.object(feed._session, "get", side_effect=[_mock_response("22"), _mock_response("40")]) as mock_get:
            feed.get_sentiment(now=1000.0)
            value, source = feed.get_sentiment(now=1070.0)
            self.assertEqual(value, 40.0)
            self.assertEqual(source, "live_alternative_me")
            self.assertEqual(mock_get.call_count, 2)

    def test_network_failure_with_stale_but_usable_cache(self):
        feed = SentimentFeed(SentimentFeedConfig(refresh_sec=60, max_age_sec=300))
        with patch.object(feed._session, "get", return_value=_mock_response("15")):
            feed.get_sentiment(now=1000.0)

        with patch.object(feed._session, "get", side_effect=RuntimeError("network down")):
            value, source = feed.get_sentiment(now=1200.0)
            self.assertEqual(value, 15.0)
            self.assertEqual(source, "fallback_stale_cache")

    def test_network_failure_with_no_cache_falls_back_to_neutral(self):
        feed = SentimentFeed(SentimentFeedConfig())
        with patch.object(feed._session, "get", side_effect=RuntimeError("network down")):
            value, source = feed.get_sentiment(now=1000.0)
            self.assertEqual(value, 50.0)
            self.assertEqual(source, "fallback_neutral_50")

    def test_network_failure_with_expired_cache_falls_back_to_neutral(self):
        feed = SentimentFeed(SentimentFeedConfig(refresh_sec=60, max_age_sec=100))
        with patch.object(feed._session, "get", return_value=_mock_response("15")):
            feed.get_sentiment(now=1000.0)

        with patch.object(feed._session, "get", side_effect=RuntimeError("network down")):
            value, source = feed.get_sentiment(now=1200.0)
            self.assertEqual(value, 50.0)
            self.assertEqual(source, "fallback_neutral_50")

    def test_get_sentiment_never_raises_on_bad_payload(self):
        feed = SentimentFeed(SentimentFeedConfig())
        bad_resp = MagicMock()
        bad_resp.raise_for_status = MagicMock()
        bad_resp.json = MagicMock(return_value={"data": []})
        with patch.object(feed._session, "get", return_value=bad_resp):
            value, source = feed.get_sentiment(now=1000.0)
            self.assertEqual(value, 50.0)
            self.assertEqual(source, "fallback_neutral_50")


if __name__ == "__main__":
    unittest.main()
