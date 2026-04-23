from __future__ import annotations

import unittest

from trading.exchange.bybit_adapter import BybitAdapter, InstrumentMetadataError
from trading.exchange.schemas import InstrumentRules, PositionSide


class BybitAdapterV2Tests(unittest.TestCase):
    def test_position_idx_logic(self):
        self.assertEqual(BybitAdapter.position_idx_for_side(PositionSide.LONG, hedge_mode=False), 0)
        self.assertEqual(BybitAdapter.position_idx_for_side(PositionSide.SHORT, hedge_mode=False), 0)
        self.assertEqual(BybitAdapter.position_idx_for_side(PositionSide.LONG, hedge_mode=True), 1)
        self.assertEqual(BybitAdapter.position_idx_for_side(PositionSide.SHORT, hedge_mode=True), 2)

    def test_extract_instrument_rules(self):
        payload = {
            "result": {
                "list": [
                    {
                        "lotSizeFilter": {"qtyStep": "0.01", "minOrderQty": "0.1", "minNotionalValue": "5"},
                        "priceFilter": {"tickSize": "0.1"},
                    }
                ]
            }
        }
        rules = BybitAdapter._extract_instrument_rules("BTCUSDT", payload)
        self.assertEqual(rules.symbol, "BTCUSDT")
        self.assertAlmostEqual(rules.qty_step, 0.01)
        self.assertAlmostEqual(rules.min_qty, 0.1)
        self.assertAlmostEqual(rules.tick_size, 0.1)

    def test_extract_instrument_rules_prefers_stricter_market_qty_cap(self):
        payload = {
            "result": {
                "list": [
                    {
                        "lotSizeFilter": {
                            "qtyStep": "1",
                            "minOrderQty": "1",
                            "minNotionalValue": "5",
                            "maxOrderQty": "1000000",
                            "maxMktOrderQty": "250000",
                        },
                        "priceFilter": {"tickSize": "0.0001"},
                    }
                ]
            }
        }
        rules = BybitAdapter._extract_instrument_rules("ENJUSDT", payload)
        self.assertEqual(rules.symbol, "ENJUSDT")
        self.assertAlmostEqual(rules.max_qty, 250000.0)

    def test_round_qty(self):
        self.assertEqual(BybitAdapter.round_qty(1.239, 0.01), 1.23)
        self.assertEqual(BybitAdapter.round_qty(0.009, 0.01), 0.0)
        self.assertEqual(BybitAdapter.round_qty(0.30000000000000004, 0.1), 0.3)

    def test_order_side_parsing_accepts_long_short_aliases(self):
        self.assertEqual(BybitAdapter._parse_order_side("LONG").value, "BUY")
        self.assertEqual(BybitAdapter._parse_order_side("SHORT").value, "SELL")

    def test_invalid_instrument_rules_rejected(self):
        with self.assertRaises(InstrumentMetadataError):
            BybitAdapter._validate_rules(InstrumentRules(symbol="BTCUSDT", tick_size=0.0, qty_step=0.01, min_qty=0.01, min_notional=5.0))

    def test_extract_account_snapshot_prefers_positive_totals(self):
        adapter = object.__new__(BybitAdapter)
        payload = {
            "result": {
                "list": [
                    {
                        "totalEquity": "125.5",
                        "totalAvailableBalance": "111.25",
                        "coin": [
                            {
                                "coin": "USDT",
                                "equity": "0",
                                "availableToWithdraw": "",
                                "walletBalance": "0",
                            }
                        ],
                    }
                ]
            }
        }
        snapshot = BybitAdapter._extract_account_snapshot(adapter, payload)
        self.assertAlmostEqual(snapshot.equity_usdt, 125.5)
        self.assertAlmostEqual(snapshot.available_balance_usdt, 111.25)

    def test_get_account_demo_auto_funds_and_retries_wallet_balance(self):
        class FakeClient:
            def __init__(self):
                self.calls = 0
                self.private_auth_invalid = False
                self.private_auth_invalid_reason = ""

            def request_private(self, method, path, params=None):
                self.calls += 1
                if self.calls == 1:
                    return {
                        "result": {
                            "list": [
                                {
                                    "totalEquity": "0",
                                    "totalAvailableBalance": "0",
                                    "coin": [{"coin": "USDT", "equity": "0", "walletBalance": "0"}],
                                }
                            ]
                        }
                    }
                return {
                    "result": {
                        "list": [
                            {
                                "totalEquity": "100000",
                                "totalAvailableBalance": "99950",
                                "coin": [{"coin": "USDT", "equity": "100000", "walletBalance": "100000"}],
                            }
                        ]
                    }
                }

            def apply_demo_funds(self, *, usdt_amount="100000"):
                return {"retCode": 0, "retMsg": "OK"}

        adapter = object.__new__(BybitAdapter)
        adapter.config = type("Cfg", (), {"dry_run": False, "demo": True})()
        adapter.client = FakeClient()
        adapter._demo_auto_fund_attempted = False
        snapshot = BybitAdapter.get_account(adapter)
        self.assertGreater(snapshot.equity_usdt, 0.0)
        self.assertGreater(snapshot.available_balance_usdt, 0.0)
        self.assertTrue(adapter._demo_auto_fund_attempted)

    def test_ensure_position_leverage_records_success(self):
        class FakeClient:
            def set_position_leverage(self, **kwargs):
                return {"retCode": 0, "retMsg": "OK", "result": kwargs}

        adapter = object.__new__(BybitAdapter)
        adapter.client = FakeClient()
        adapter._applied_leverage_cache = {}
        ok = BybitAdapter.ensure_position_leverage(adapter, "BTCUSDT", 3.0)
        self.assertTrue(ok)
        self.assertEqual(adapter._applied_leverage_cache["BTCUSDT"], 3.0)


if __name__ == "__main__":
    unittest.main()

