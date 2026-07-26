from __future__ import annotations

import unittest

import pandas as pd

from core.signal_generator import SignalConfig, SignalGenerator


def _frame(close: float, volume: float, turnover: float | None, n: int = 20) -> pd.DataFrame:
    data = {
        "open": [close] * n,
        "high": [close * 1.001] * n,
        "low": [close * 0.999] * n,
        "close": [close] * n,
        "volume": [volume] * n,
        "atr": [close * 0.08] * n,
    }
    if turnover is not None:
        data["turnover"] = [turnover] * n
    return pd.DataFrame(data)


class TurnoverUnitsV2Tests(unittest.TestCase):
    """A MEXC kline's `volume` is a contract count, so close*volume is wrong by
    the contract size and the error differs per symbol: on BTC (contractSize
    0.0001) it overstates turnover roughly 10,000x, on CHILLGUY (contractSize 10)
    it understates it 10x. The gate must read the exchange's own `amount`."""

    def _gen(self) -> SignalGenerator:
        return SignalGenerator(SignalConfig(min_hourly_usd_volume=100_000.0, min_atr_pct=0.0))

    def test_turnover_column_is_used_when_present(self):
        gen = self._gen()
        # 12 bars x 20_000 turnover = 240k, comfortably over the floor
        ok, d = gen._layer1b_quality_gate(_frame(close=1.0, volume=1.0, turnover=20_000.0))
        self.assertTrue(ok)
        self.assertAlmostEqual(d["usd_volume_recent"], 240_000.0)

    def test_contract_count_no_longer_inflates_a_thin_symbol(self):
        """BTC-like: tiny contract size means close*volume is enormous while the
        real turnover is small. The gate must reject on the real number."""
        gen = self._gen()
        ok, d = gen._layer1b_quality_gate(_frame(close=64_000.0, volume=100.0, turnover=640.0))
        self.assertFalse(ok)
        self.assertAlmostEqual(d["usd_volume_recent"], 640.0 * 12)
        self.assertEqual(d["liquidity_ok"], 0.0)

    def test_contract_count_no_longer_understates_a_liquid_symbol(self):
        """CHILLGUY-like: contract size 10 means close*volume is a tenth of the
        real turnover, and the old maths would have rejected a tradeable coin."""
        gen = self._gen()
        naive = 0.012 * 100_000  # what close*volume would have given: 1_200 per bar
        real = naive * 10
        ok, d = gen._layer1b_quality_gate(_frame(close=0.012, volume=100_000.0, turnover=real))
        self.assertTrue(ok)
        self.assertGreater(d["usd_volume_recent"], 100_000.0)

    def test_falls_back_to_price_times_volume_when_turnover_is_absent(self):
        gen = self._gen()
        _, d = gen._layer1b_quality_gate(_frame(close=10.0, volume=2_000.0, turnover=None))
        self.assertAlmostEqual(d["usd_volume_recent"], 10.0 * 2_000.0 * 12)


class _StubClient:
    @staticmethod
    def denormalize_symbol(symbol: str) -> str:
        return symbol.replace("_", "")


class HistoryCacheSchemaV2Tests(unittest.TestCase):
    def test_cache_without_turnover_is_discarded(self):
        """Old caches predate turnover capture and cannot be repaired from what
        they hold; serving them would leave half the history incomparable."""
        import os
        import tempfile

        from trading.market_data.history import HistoryCollector, HistoryConfig

        with tempfile.TemporaryDirectory() as tmp:
            collector = HistoryCollector(client=_StubClient(), config=HistoryConfig(cache_dir=tmp))
            path = os.path.join(tmp, "TESTUSDT_Min60.csv")
            pd.DataFrame({"time": [1, 2], "open": [1.0, 1.0], "high": [1.0, 1.0],
                          "low": [1.0, 1.0], "close": [1.0, 1.0], "volume": [1.0, 1.0]}).to_csv(path, index=False)
            self.assertTrue(collector._read_cache("TESTUSDT", "Min60").empty)

    def test_cache_with_turnover_is_served(self):
        import os
        import tempfile

        from trading.market_data.history import HistoryCollector, HistoryConfig

        with tempfile.TemporaryDirectory() as tmp:
            collector = HistoryCollector(client=_StubClient(), config=HistoryConfig(cache_dir=tmp))
            path = os.path.join(tmp, "TESTUSDT_Min60.csv")
            pd.DataFrame({"time": [1, 2], "open": [1.0, 1.0], "high": [1.0, 1.0],
                          "low": [1.0, 1.0], "close": [1.0, 1.0], "volume": [1.0, 1.0],
                          "turnover": [10.0, 10.0]}).to_csv(path, index=False)
            self.assertEqual(len(collector._read_cache("TESTUSDT", "Min60")), 2)


if __name__ == "__main__":
    unittest.main()
