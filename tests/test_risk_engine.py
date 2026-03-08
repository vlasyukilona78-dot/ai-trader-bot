import os
import unittest

if os.getenv("ALLOW_LEGACY_RUNTIME", "false").strip().lower() not in ("1", "true", "yes"):
    raise unittest.SkipTest("legacy runtime tests are quarantined; run V2 tests instead")

from engine.risk import RiskConfig, RiskEngine


class RiskEngineTests(unittest.TestCase):
    def test_recommend_qty_positive(self):
        engine = RiskEngine(RiskConfig(account_equity_usdt=1000, risk_per_trade=0.01, min_qty=0.001, max_qty=10))
        qty = engine.recommend_qty(entry=100.0, sl=101.0)
        self.assertGreater(qty, 0.0)
        self.assertLessEqual(qty, 10.0)

    def test_circuit_breaker_on_consecutive_losses(self):
        cfg = RiskConfig(max_consecutive_losses=2, circuit_breaker_minutes=5)
        engine = RiskEngine(cfg)
        engine.on_trade_closed(-10)
        ok, _ = engine.can_open_trade()
        self.assertTrue(ok)
        engine.on_trade_closed(-10)
        ok, reason = engine.can_open_trade()
        self.assertFalse(ok)
        self.assertIn("max_consecutive_losses", reason)


if __name__ == "__main__":
    unittest.main()

