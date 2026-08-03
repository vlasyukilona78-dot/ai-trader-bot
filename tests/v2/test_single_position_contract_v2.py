from __future__ import annotations

import unittest

import pandas as pd

from backtesting.single_position import (
    EntryPlan,
    ExecutionCosts,
    FundingPayment,
    ScoredCandidate,
    SinglePositionContract,
    SinglePositionContractError,
    SizingRules,
    replay_single_short,
    select_single_position,
)


FREE = ExecutionCosts(
    entry_fee_rate=0.0,
    exit_fee_rate=0.0,
    half_spread=0.0,
    entry_slippage=0.0,
    exit_slippage=0.0,
)


def _contract(*, costs: ExecutionCosts = FREE, horizon_bars: int = 2) -> SinglePositionContract:
    return SinglePositionContract(
        costs=costs,
        sizing=SizingRules(
            equity_quote=1000.0,
            risk_fraction=0.01,
            max_notional_quote=1000.0,
            max_leverage=1.0,
            quantity_step=0.001,
            min_quantity=0.001,
            min_notional_quote=5.0,
        ),
        bar_interval_seconds=300,
        max_holding_seconds=300 * horizon_bars,
    )


_DECISION_TS = 900.0
_ACTIONABLE_TS = 950.0
# the first 300s boundary strictly after the decision became actionable
_ENTRY_BAR_OPEN_TS = 1200.0
_COHORT = "cycle-1"


def _plan(
    symbol: str = "AAAUSDT",
    *,
    cohort_id: str = _COHORT,
    decision_ts: float = _DECISION_TS,
    actionable_ts: float = _ACTIONABLE_TS,
    entry_bar_open_ts: float = _ENTRY_BAR_OPEN_TS,
) -> EntryPlan:
    return EntryPlan(
        symbol=symbol,
        cohort_id=cohort_id,
        decision_ts=decision_ts,
        actionable_ts=actionable_ts,
        entry_bar_open_ts=entry_bar_open_ts,
        decision_price=100.0,
        stop_price=105.0,
        take_profit_price=95.0,
    )


def _bars(rows, start_ts: float = _ENTRY_BAR_OPEN_TS) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "time": start_ts + index * 300,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
            }
            for index, (open_price, high, low, close) in enumerate(rows)
        ]
    )


class SinglePositionReplayV2Tests(unittest.TestCase):
    def test_take_profit_is_one_entry_and_one_exit(self):
        result = replay_single_short(
            _bars([(100.0, 101.0, 94.0, 96.0), (96.0, 97.0, 94.0, 95.0)]),
            plan=_plan(),
            contract=_contract(),
        )

        self.assertTrue(result.filled)
        self.assertEqual(result.exit_reason, "take_profit")
        self.assertAlmostEqual(result.quantity, 2.0)
        self.assertAlmostEqual(result.net_pnl_quote, 10.0)
        self.assertAlmostEqual(result.return_on_risk, 1.0)
        self.assertEqual(result.bars_held, 1)

    def test_stop_wins_when_stop_and_target_share_a_bar(self):
        result = replay_single_short(
            _bars([(100.0, 106.0, 94.0, 100.0), (100.0, 101.0, 99.0, 100.0)]),
            plan=_plan(),
            contract=_contract(),
        )

        self.assertEqual(result.exit_reason, "stop")
        self.assertAlmostEqual(result.net_pnl_quote, -10.0)
        self.assertAlmostEqual(result.return_on_risk, -1.0)

    def test_gap_through_stop_uses_worse_open(self):
        result = replay_single_short(
            _bars([(100.0, 101.0, 99.0, 100.0), (110.0, 112.0, 109.0, 111.0)]),
            plan=_plan(),
            contract=_contract(),
        )

        self.assertEqual(result.exit_reason, "stop_gap")
        self.assertEqual(result.exit_reference_price, 110.0)
        self.assertLess(result.return_on_risk, -1.0)

    def test_setup_invalidated_before_entry_is_not_credited(self):
        result = replay_single_short(
            _bars([(94.0, 96.0, 93.0, 95.0), (95.0, 96.0, 94.0, 95.0)]),
            plan=_plan(),
            contract=_contract(),
        )

        self.assertFalse(result.filled)
        self.assertEqual(result.exit_reason, "entry_invalidated_by_target_gap")
        self.assertEqual(result.net_pnl_quote, 0.0)

    def test_costs_reduce_the_same_trade(self):
        paid = ExecutionCosts(
            entry_fee_rate=0.0008,
            exit_fee_rate=0.0008,
            half_spread=0.000145,
            entry_slippage=0.00014,
            exit_slippage=0.00014,
        )
        bars = _bars([(100.0, 101.0, 94.0, 96.0), (96.0, 97.0, 94.0, 95.0)])
        free_result = replay_single_short(bars, plan=_plan(), contract=_contract())
        paid_result = replay_single_short(bars, plan=_plan(), contract=_contract(costs=paid))

        self.assertGreater(paid_result.fees_quote, 0.0)
        self.assertLess(paid_result.net_pnl_quote, free_result.net_pnl_quote)

    def test_positive_funding_rate_pays_the_short(self):
        bars = _bars([(100.0, 101.0, 99.0, 100.0), (100.0, 101.0, 99.0, 100.0)])
        without = replay_single_short(bars, plan=_plan(), contract=_contract())
        with_funding = replay_single_short(
            bars,
            plan=_plan(),
            contract=_contract(),
            funding_payments=(FundingPayment(timestamp=1300.0, rate=0.001, mark_price=100.0),),
        )

        self.assertEqual(with_funding.funding_events_applied, 1)
        self.assertGreater(with_funding.net_pnl_quote, without.net_pnl_quote)

    def test_intrabar_exit_does_not_assume_later_funding_was_received(self):
        result = replay_single_short(
            _bars([(100.0, 101.0, 94.0, 96.0), (96.0, 97.0, 94.0, 95.0)]),
            plan=_plan(),
            contract=_contract(),
            funding_payments=(FundingPayment(timestamp=1200.0, rate=0.01, mark_price=100.0),),
        )

        self.assertEqual(result.exit_reason, "take_profit")
        self.assertEqual(result.funding_events_applied, 0)
        self.assertEqual(result.funding_pnl_quote, 0.0)

    def test_incomplete_or_gapped_horizon_fails_closed(self):
        with self.assertRaisesRegex(SinglePositionContractError, "incomplete_horizon"):
            replay_single_short(
                _bars([(100.0, 101.0, 99.0, 100.0)]),
                plan=_plan(),
                contract=_contract(horizon_bars=2),
            )

        gapped = _bars([(100.0, 101.0, 99.0, 100.0), (100.0, 101.0, 99.0, 100.0)])
        gapped.loc[1, "time"] += 300
        with self.assertRaisesRegex(SinglePositionContractError, "incomplete_horizon|bar_cadence_gap"):
            replay_single_short(gapped, plan=_plan(), contract=_contract(horizon_bars=2))

    def test_contract_rejects_any_concurrency_other_than_one(self):
        with self.assertRaisesRegex(SinglePositionContractError, "concurrency_must_equal_one"):
            SinglePositionContract(
                costs=FREE,
                sizing=_contract().sizing,
                bar_interval_seconds=300,
                max_holding_seconds=600,
                max_concurrent_positions=2,
            )


class SinglePositionSelectionV2Tests(unittest.TestCase):
    def _scored(
        self,
        score: float,
        symbol: str,
        exit_ts: float,
        *,
        cohort_id: str = _COHORT,
        entry_bar_open_ts: float = _ENTRY_BAR_OPEN_TS,
        decision_ts: float = _DECISION_TS,
    ) -> ScoredCandidate:
        plan = _plan(
            symbol=symbol,
            cohort_id=cohort_id,
            decision_ts=decision_ts,
            actionable_ts=entry_bar_open_ts - 250.0,
            entry_bar_open_ts=entry_bar_open_ts,
        )
        result = replay_single_short(
            _bars(
                [(100.0, 101.0, 99.0, 100.0), (100.0, 101.0, 99.0, 100.0)],
                start_ts=entry_bar_open_ts,
            ),
            plan=plan,
            contract=_contract(),
        )
        return ScoredCandidate(
            score, plan, result.__class__(**{**result.__dict__, "exit_ts": exit_ts})
        )

    def test_highest_score_wins_cohort_and_book_stays_single(self):
        a = self._scored(0.6, "AUSDT", 2000.0)
        b = self._scored(0.9, "BUSDT", 2000.0)
        busy = self._scored(
            0.95, "CUSDT", 2100.0, cohort_id="cycle-2", entry_bar_open_ts=1800.0
        )
        later = self._scored(
            0.7, "DUSDT", 2600.0, cohort_id="cycle-3", entry_bar_open_ts=2400.0
        )

        selection = select_single_position([a, later, busy, b], minimum_score=0.5)

        self.assertEqual([item.result.symbol for item in selection.selected], ["BUSDT", "DUSDT"])
        self.assertEqual(selection.skipped_busy, 2)

    def test_threshold_and_unfilled_are_counted_without_selection(self):
        below = self._scored(0.4, "AUSDT", 1600.0)
        invalid_plan = _plan(symbol="BUSDT")
        invalid = replay_single_short(
            _bars([(106.0, 107.0, 105.5, 106.0), (106.0, 107.0, 105.0, 106.0)]),
            plan=invalid_plan,
            contract=_contract(),
        )
        selection = select_single_position(
            [below, ScoredCandidate(0.8, invalid_plan, invalid)],
            minimum_score=0.5,
        )

        self.assertEqual(selection.selected, ())
        self.assertEqual(selection.skipped_below_threshold, 1)
        self.assertEqual(selection.skipped_unfilled, 1)

    def test_unfilled_top_score_does_not_promote_runner_up_with_hindsight(self):
        top_plan = _plan(symbol="TOPUSDT")
        invalid = replay_single_short(
            _bars([(106.0, 107.0, 105.5, 106.0), (106.0, 107.0, 105.0, 106.0)]),
            plan=top_plan,
            contract=_contract(),
        )
        runner_up = self._scored(0.8, "RUNNERUSDT", 1600.0)

        selection = select_single_position(
            [ScoredCandidate(0.9, top_plan, invalid), runner_up],
            minimum_score=0.5,
        )

        self.assertEqual(selection.selected, ())
        self.assertEqual(selection.skipped_unfilled, 1)
        self.assertEqual(selection.skipped_busy, 1)


if __name__ == "__main__":
    unittest.main()
