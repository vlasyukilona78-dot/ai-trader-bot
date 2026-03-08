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

    def test_round_qty(self):
        self.assertEqual(BybitAdapter.round_qty(1.239, 0.01), 1.23)
        self.assertEqual(BybitAdapter.round_qty(0.009, 0.01), 0.0)

    def test_order_side_parsing_accepts_long_short_aliases(self):
        self.assertEqual(BybitAdapter._parse_order_side("LONG").value, "BUY")
        self.assertEqual(BybitAdapter._parse_order_side("SHORT").value, "SELL")

    def test_invalid_instrument_rules_rejected(self):
        with self.assertRaises(InstrumentMetadataError):
            BybitAdapter._validate_rules(InstrumentRules(symbol="BTCUSDT", tick_size=0.0, qty_step=0.01, min_qty=0.01, min_notional=5.0))


if __name__ == "__main__":
    unittest.main()

