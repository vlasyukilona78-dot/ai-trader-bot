from __future__ import annotations

import unittest

import pandas as pd

from backtesting.replay import ExecutionCosts, ReplayConfig, replay_short, summarise

FREE = ExecutionCosts(taker_fee=0.0, maker_fee=0.0, half_spread=0.0, slippage=0.0)


def _bars(rows) -> pd.DataFrame:
    """rows: list of (high, low, close)."""
    return pd.DataFrame(rows, columns=["high", "low", "close"])


class ReplayMechanicsV2Tests(unittest.TestCase):
    def test_immediate_target_pays_the_target(self):
        cfg = ReplayConfig(target_pct=0.03, costs=FREE, max_loss_on_deployed=None)
        r = replay_short(_bars([(100.0, 96.0, 97.0)]), 100.0, cfg)
        self.assertEqual(r.exit_reason, "target")
        self.assertEqual(r.legs, 1)
        self.assertAlmostEqual(r.pnl_on_initial, 0.03, places=9)

    def test_costs_reduce_the_same_trade(self):
        bars = _bars([(100.0, 96.0, 97.0)])
        free = replay_short(bars, 100.0, ReplayConfig(costs=FREE, max_loss_on_deployed=None))
        paid = replay_short(bars, 100.0, ReplayConfig(max_loss_on_deployed=None))
        self.assertLess(paid.pnl_on_initial, free.pnl_on_initial)
        self.assertGreater(paid.fees_paid_on_initial, 0.0)

    def test_adverse_extreme_is_taken_before_the_exit_within_a_bar(self):
        """One bar that both triggers an add and reaches the target must add first."""
        cfg = ReplayConfig(target_pct=0.03, dca_step_pct=0.08, costs=FREE, max_loss_on_deployed=None)
        r = replay_short(_bars([(109.0, 90.0, 95.0)]), 100.0, cfg)
        self.assertEqual(r.legs, 2)

    def test_stop_wins_a_same_bar_tie_with_the_target(self):
        cfg = ReplayConfig(target_pct=0.03, stop_pct_from_blended=0.02,
                           costs=FREE, max_loss_on_deployed=None, max_adds=0)
        r = replay_short(_bars([(103.0, 96.0, 97.0)]), 100.0, cfg)
        self.assertEqual(r.exit_reason, "stop")
        self.assertLess(r.pnl_on_initial, 0.0)

    def test_horizon_marks_out_at_the_last_close(self):
        cfg = ReplayConfig(target_pct=0.30, costs=FREE, max_loss_on_deployed=None, max_adds=0)
        r = replay_short(_bars([(101.0, 99.0, 100.0), (101.0, 99.0, 99.0)]), 100.0, cfg)
        self.assertEqual(r.exit_reason, "horizon")
        self.assertAlmostEqual(r.pnl_on_initial, 0.01, places=6)

    def test_stop_fires_before_a_higher_averaging_level(self):
        """With a stop at 105 and an add at 108, a bar reaching 110 traded through
        the stop first. Filling the add and staying in would keep a position the
        market had already closed."""
        cfg = ReplayConfig(costs=FREE, dca_step_pct=0.08, stop_pct_from_blended=0.05,
                           max_loss_on_deployed=None)
        r = replay_short(_bars([(110.0, 104.0, 106.0)]), 100.0, cfg)
        self.assertEqual(r.exit_reason, "stop")
        self.assertEqual(r.legs, 1)  # no averaging leg was added

    def test_stopped_loss_is_bounded_when_price_does_not_gap(self):
        cfg = ReplayConfig(costs=FREE, max_loss_on_deployed=1.0, max_adds=0)
        # the bar trades through the stop rather than opening beyond it
        r = replay_short(_bars([(260.0, 90.0, 250.0)]), 100.0, cfg)
        self.assertEqual(r.exit_reason, "stop")
        self.assertAlmostEqual(r.pnl_on_deployed, -1.0, delta=0.05)

    def test_a_gap_through_the_stop_costs_more_than_the_stop_price(self):
        """The stop is not a guaranteed fill. If the bar's whole range sits above
        it, the exit happens at the worst price actually available, and the loss
        exceeds what the stop nominally risked."""
        cfg = ReplayConfig(costs=FREE, max_loss_on_deployed=1.0, max_adds=0)
        r = replay_short(_bars([(400.0, 300.0, 350.0)]), 100.0, cfg)
        self.assertEqual(r.exit_reason, "stop")
        self.assertLess(r.pnl_on_deployed, -1.0)


class LegBlendingV2Tests(unittest.TestCase):
    """Equal-notional legs buy fewer units at higher prices, so the blend is
    notional-weighted, not the arithmetic mean of the leg prices."""

    def test_equal_notional_blend_is_below_the_arithmetic_mean(self):
        cfg = ReplayConfig(target_pct=0.99, dca_step_pct=0.10, max_adds=1,
                           equal_notional_legs=True, costs=FREE, max_loss_on_deployed=None)
        r = replay_short(_bars([(110.0, 109.0, 110.0), (110.0, 109.0, 110.0)]), 100.0, cfg)
        self.assertEqual(r.legs, 2)
        arithmetic = (100.0 + 110.0) / 2
        self.assertLess(r.blended_entry, arithmetic)
        self.assertAlmostEqual(r.blended_entry, 2 / (1 / 100.0 + 1 / 110.0), places=6)

    def test_equal_quantity_blend_is_the_arithmetic_mean(self):
        cfg = ReplayConfig(target_pct=0.99, dca_step_pct=0.10, max_adds=1,
                           equal_notional_legs=False, costs=FREE, max_loss_on_deployed=None)
        r = replay_short(_bars([(110.0, 109.0, 110.0), (110.0, 109.0, 110.0)]), 100.0, cfg)
        self.assertAlmostEqual(r.blended_entry, 105.0, places=6)

    def test_deployed_capital_grows_with_each_leg(self):
        cfg = ReplayConfig(target_pct=0.99, dca_step_pct=0.08, max_adds=3,
                           costs=FREE, max_loss_on_deployed=None)
        r = replay_short(_bars([(130.0, 129.0, 130.0)]), 100.0, cfg)
        self.assertEqual(r.legs, 4)
        self.assertAlmostEqual(r.max_deployed, 4.0, places=6)


class DrawdownConventionV2Tests(unittest.TestCase):
    def test_drawdown_is_measured_against_the_initial_leg(self):
        """Normalising by the moving average entry understates the loss; this is
        the convention error the earlier score had."""
        cfg = ReplayConfig(target_pct=0.99, dca_step_pct=0.08, max_adds=2,
                           costs=FREE, max_loss_on_deployed=None)
        r = replay_short(_bars([(120.0, 119.0, 120.0)]), 100.0, cfg)
        naive = (120.0 - r.blended_entry) / r.blended_entry * r.legs
        self.assertGreater(r.worst_drawdown_on_initial, naive)


class TargetConsistencyV2Tests(unittest.TestCase):
    def test_target_is_applied_during_the_replay_not_after(self):
        """The earlier score credited one target while the labels used another.
        Here a bigger target simply resolves less often."""
        bars = _bars([(100.0, 97.5, 98.0), (100.0, 97.5, 98.0)])
        near = replay_short(bars, 100.0, ReplayConfig(target_pct=0.02, costs=FREE,
                                                     max_loss_on_deployed=None, max_adds=0))
        far = replay_short(bars, 100.0, ReplayConfig(target_pct=0.10, costs=FREE,
                                                    max_loss_on_deployed=None, max_adds=0))
        self.assertEqual(near.exit_reason, "target")
        self.assertEqual(far.exit_reason, "horizon")


class SummariseV2Tests(unittest.TestCase):
    def test_reports_capital_and_tail_alongside_the_average(self):
        cfg = ReplayConfig(costs=FREE, max_loss_on_deployed=None, max_adds=0)
        winners = [replay_short(_bars([(100.0, 96.0, 97.0)]), 100.0, cfg) for _ in range(3)]
        loser = replay_short(_bars([(140.0, 139.0, 140.0)]), 100.0,
                             ReplayConfig(costs=FREE, max_loss_on_deployed=0.2, max_adds=0))
        s = summarise(winners + [loser])
        self.assertEqual(s["trades"], 4.0)
        self.assertAlmostEqual(s["win_rate"], 0.75)
        self.assertLess(s["worst_trade"], 0.0)
        self.assertIn("mean_pnl_on_deployed", s)
        self.assertIn("worst_drawdown_on_initial", s)

    def test_empty_input_is_handled(self):
        self.assertEqual(summarise([]), {})


if __name__ == "__main__":
    unittest.main()
