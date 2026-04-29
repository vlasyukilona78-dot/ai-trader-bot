from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from core.coinglass_liquidation import (
    CoinglassLiquidationClient,
    CoinglassLiquidationConfig,
    parse_coinglass_heatmap_bands,
)
from core.liquidation_map import build_liquidation_map


class CoinglassLiquidationV2Tests(unittest.TestCase):
    def test_parse_coinglass_heatmap_bands_keeps_strong_levels(self):
        payload = {
            "data": {
                "y": ["95", "100", "105", "110"],
                "prices": [
                    [1777000000, 100, 102, 99, 101],
                    [1777014400, 101, 104, 100, 103],
                    [1777028800, 103, 106, 102, 104],
                ],
                "liq": [
                    [0, 0, 12],
                    [1, 2, 50],
                    [2, 2, 90],
                    [2, 3, 8],
                ],
            }
        }

        bands = parse_coinglass_heatmap_bands(payload, current_price=102.0, min_intensity=0.20)

        self.assertTrue(any(row["source"] == "coinglass" for row in bands))
        self.assertTrue(any(row["side"] == "above" and float(row["level"]) == 105.0 for row in bands))
        self.assertFalse(any(float(row["level"]) == 110.0 for row in bands))
        strong_band = next(row for row in bands if float(row["level"]) == 105.0)
        self.assertEqual(float(strong_band["notional_usdt"]), 140.0)

    def test_liquidation_map_uses_coinglass_bands_from_frame_attrs(self):
        idx = pd.date_range("2026-03-01", periods=80, freq="4h", tz="UTC")
        close = np.linspace(100.0, 103.0, len(idx))
        frame = pd.DataFrame(
            {
                "open": close - 0.2,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": np.linspace(1000.0, 1400.0, len(idx)),
                "atr": np.full(len(idx), 1.0),
            },
            index=idx,
        )
        frame.attrs["coinglass_liquidation_bands"] = [
            {
                "level": 104.5,
                "weight": 4.7,
                "side": "above",
                "source": "coinglass",
                "start_ts": int(idx[-12].timestamp()),
                "end_ts": int(idx[-1].timestamp()),
                "notional_usdt": 300000.0,
                "margin_usdt": 42500.0,
            }
        ]

        liq_map = build_liquidation_map(frame)

        self.assertTrue(any(band.source in {"coinglass", "feed"} for band in liq_map.bands))
        self.assertTrue(any(band.notional_usdt >= 300000.0 for band in liq_map.bands))
        self.assertTrue(any(band.margin_usdt >= 42500.0 for band in liq_map.bands))
        self.assertIsNotNone(liq_map.nearest_above_distance_pct)
        self.assertGreater(liq_map.upside_risk, 0.0)

    def test_client_applies_global_cooldown_after_rate_limit(self):
        class FakeResponse:
            status_code = 429
            headers = {"Retry-After": "120"}

            def raise_for_status(self):
                raise AssertionError("429 responses must not call raise_for_status")

        class FakeSession:
            def __init__(self):
                self.calls: list[str] = []
                self.trust_env = False
                self.headers: dict[str, str] = {}
                self.proxies: dict[str, str] = {}

            def get(self, _url, *, params, timeout):
                self.calls.append(str(params["symbol"]))
                return FakeResponse()

            def close(self):
                pass

        cfg = CoinglassLiquidationConfig(
            enabled=True,
            api_key="test",
            rate_limit_cooldown_sec=900.0,
            ttl_sec=30.0,
        )
        client = CoinglassLiquidationClient(cfg)
        fake_session = FakeSession()
        client._session = fake_session

        first = client.fetch_heatmap_bands("BTCUSDT", current_price=100.0)
        second = client.fetch_heatmap_bands("ETHUSDT", current_price=100.0)

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(fake_session.calls, ["BTCUSDT"])


if __name__ == "__main__":
    unittest.main()
