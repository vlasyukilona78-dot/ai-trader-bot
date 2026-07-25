from __future__ import annotations

import unittest

from trading.signals.volatility_context import VolatilityContext, VolatilityContextConfig


class VolatilityContextV2Tests(unittest.TestCase):
    def _ctx(self, **over) -> VolatilityContext:
        cfg = {"percentile": 0.8, "min_observations": 5, "fallback_floor": 0.046}
        cfg.update(over)
        return VolatilityContext(VolatilityContextConfig(**cfg))

    def test_cold_start_uses_the_fixed_fallback(self):
        ctx = self._ctx()
        ctx.observe("A", 0.01, now=100.0)
        self.assertEqual(ctx.floor(now=100.0), 0.046)

    def test_floor_tracks_the_percentile_once_populated(self):
        ctx = self._ctx()
        for i, v in enumerate([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]):
            ctx.observe(f"S{i}", v, now=100.0)
        # 80th percentile of ten sorted values -> index 7 -> 0.08
        self.assertAlmostEqual(ctx.floor(now=100.0), 0.08)

    def test_floor_adapts_when_the_whole_market_calms_down(self):
        """The March failure: a fixed floor admitted almost nothing in a quiet
        period, while a percentile keeps selecting the top slice."""
        hot = self._ctx()
        for i, v in enumerate([0.05, 0.06, 0.07, 0.08, 0.09, 0.10]):
            hot.observe(f"S{i}", v, now=100.0)
        calm = self._ctx()
        for i, v in enumerate([0.01, 0.015, 0.02, 0.025, 0.03, 0.035]):
            calm.observe(f"S{i}", v, now=100.0)

        self.assertGreater(hot.floor(now=100.0), calm.floor(now=100.0))
        # in the calm regime the floor drops below the old hardcoded number
        self.assertLess(calm.floor(now=100.0), 0.046)

    def test_one_symbol_cannot_dominate_the_distribution(self):
        ctx = self._ctx()
        for _ in range(50):
            ctx.observe("SPAM", 0.99, now=100.0)
        for i, v in enumerate([0.01, 0.02, 0.03, 0.04, 0.05]):
            ctx.observe(f"S{i}", v, now=100.0)
        self.assertEqual(ctx.observed_symbols, 6)
        self.assertLess(ctx.floor(now=100.0), 0.99)

    def test_stale_observations_expire(self):
        ctx = self._ctx(max_age_sec=60.0)
        for i, v in enumerate([0.01, 0.02, 0.03, 0.04, 0.05, 0.06]):
            ctx.observe(f"S{i}", v, now=100.0)
        self.assertEqual(ctx.observed_symbols, 6)
        # everything ages out -> back to the fallback
        self.assertEqual(ctx.floor(now=1000.0), 0.046)
        self.assertEqual(ctx.observed_symbols, 0)

    def test_invalid_observations_are_ignored(self):
        ctx = self._ctx()
        ctx.observe("A", 0.0, now=100.0)
        ctx.observe("B", -1.0, now=100.0)
        ctx.observe("C", float("nan"), now=100.0)
        self.assertEqual(ctx.observed_symbols, 0)


if __name__ == "__main__":
    unittest.main()
