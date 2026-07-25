from __future__ import annotations

import unittest

from trading.market_data.universe import SymbolUniverse, UniverseConfig


class FakeClient:
    def __init__(self, tickers, details=None):
        self._tickers = tickers
        self._details = details or {}
        self.detail_calls = 0

    def fetch_all_tickers(self, force: bool = False):
        return list(self._tickers)

    def fetch_contract_details(self, force: bool = False):
        self.detail_calls += 1
        return dict(self._details)


def _ticker(symbol, turnover, change=0.0, price=1.0, funding=0.0):
    return {
        "symbol": symbol,
        "amount24": turnover,
        "riseFallRate": change,
        "lastPrice": price,
        "fundingRate": funding,
        "holdVol": 1000,
    }


class UniverseFilteringV2Tests(unittest.TestCase):
    def test_turnover_band_is_enforced(self):
        client = FakeClient([
            _ticker("TINY_USDT", 10_000),
            _ticker("GOOD_USDT", 1_000_000),
            _ticker("HUGE_USDT", 500_000_000),
        ])
        uni = SymbolUniverse(client, UniverseConfig(min_turnover_24h_usdt=400_000,
                                                   max_turnover_24h_usdt=100_000_000))
        self.assertEqual(uni.symbols(), ["GOODUSDT"])

    def test_non_usdt_and_index_products_excluded(self):
        client = FakeClient([
            _ticker("XRP_USD", 1_000_000),
            _ticker("NAS100_USDT", 1_000_000),
            _ticker("REAL_USDT", 1_000_000),
        ])
        uni = SymbolUniverse(client, UniverseConfig(min_turnover_24h_usdt=400_000))
        self.assertEqual(uni.symbols(), ["REALUSDT"])

    def test_entries_are_ordered_most_pumped_first(self):
        client = FakeClient([
            _ticker("A_USDT", 1_000_000, change=0.05),
            _ticker("B_USDT", 1_000_000, change=0.40),
            _ticker("C_USDT", 1_000_000, change=0.20),
        ])
        uni = SymbolUniverse(client, UniverseConfig(min_turnover_24h_usdt=400_000))
        self.assertEqual(uni.symbols(), ["BUSDT", "CUSDT", "AUSDT"])

    def test_min_notional_filter_drops_oversized_lots(self):
        client = FakeClient(
            [_ticker("BIGLOT_USDT", 1_000_000, price=100.0),
             _ticker("SMALLLOT_USDT", 1_000_000, price=1.0)],
            details={
                # 10 contracts x $100 = $1000 minimum ticket
                "BIGLOT_USDT": {"contractSize": 10, "minVol": 1, "maxLeverage": 50},
                "SMALLLOT_USDT": {"contractSize": 1, "minVol": 1, "maxLeverage": 100},
            },
        )
        uni = SymbolUniverse(client, UniverseConfig(min_turnover_24h_usdt=400_000,
                                                   max_min_notional_usdt=20.0))
        self.assertEqual(uni.symbols(), ["SMALLLOTUSDT"])
        entry = uni.snapshot.entries[0]
        self.assertAlmostEqual(entry.min_notional_usdt, 1.0)
        self.assertEqual(entry.max_leverage, 100)

    def test_specs_not_fetched_when_filter_disabled(self):
        client = FakeClient([_ticker("A_USDT", 1_000_000)], details={})
        uni = SymbolUniverse(client, UniverseConfig(min_turnover_24h_usdt=400_000,
                                                   max_min_notional_usdt=0.0))
        uni.refresh()
        self.assertEqual(client.detail_calls, 0)

    def test_previous_snapshot_survives_an_empty_response(self):
        client = FakeClient([_ticker("A_USDT", 1_000_000)])
        uni = SymbolUniverse(client, UniverseConfig(min_turnover_24h_usdt=400_000))
        self.assertEqual(uni.symbols(), ["AUSDT"])
        client._tickers = []
        self.assertEqual(uni.refresh(force=True).symbols, ["AUSDT"])

    def test_default_turnover_floor_matches_the_measured_liquidity_gate(self):
        self.assertEqual(UniverseConfig().min_turnover_24h_usdt, 400_000.0)


if __name__ == "__main__":
    unittest.main()
