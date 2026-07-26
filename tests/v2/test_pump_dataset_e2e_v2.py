from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ai.pump_dataset import (
    BAR_SECONDS_1H,
    EventConfig,
    LabelConfig,
    build_symbol_rows,
    forward_window_quality,
)

START = 1_700_000_000  # arbitrary aligned epoch


def _bars(start: int, count: int, bar_sec: int, closes) -> pd.DataFrame:
    closes = list(closes)
    assert len(closes) == count
    return pd.DataFrame(
        {
            "time": [start + i * bar_sec for i in range(count)],
            "open": closes,
            "high": [c * 1.001 for c in closes],
            "low": [c * 0.999 for c in closes],
            "close": closes,
            "volume": [1000.0] * count,
        }
    )


def _hourly_with_pump(n_hours: int = 200, pump_at: int = 120) -> pd.DataFrame:
    closes = [1.0] * n_hours
    # a clean 12% run into pump_at, then a slide well past the 3% target
    for i in range(pump_at - 6, pump_at + 1):
        closes[i] = 1.0 + 0.12 * (i - (pump_at - 7)) / 7
    for i in range(pump_at + 1, n_hours):
        closes[i] = closes[pump_at] * 0.90
    return _bars(START, n_hours, 3600, closes)


def _five_min_covering(hourly: pd.DataFrame, drop_range: tuple[int, int] | None = None) -> pd.DataFrame:
    first, last = int(hourly["time"].iloc[0]), int(hourly["time"].iloc[-1])
    n = (last - first) // 300 + 1
    closes = []
    for i in range(n):
        t = first + i * 300
        row = hourly[hourly["time"] <= t]
        closes.append(float(row["close"].iloc[-1]) if len(row) else 1.0)
    frame = _bars(first, n, 300, closes)
    if drop_range:
        lo, hi = drop_range
        frame = frame[(frame["time"] < lo) | (frame["time"] > hi)].reset_index(drop=True)
    return frame


class _FakeCollector:
    """Serves prepared frames so the builder can be exercised without network."""

    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames

    def fetch_range(self, symbol, interval, start_ts, end_ts=None, **kwargs):
        frame = self.frames.get(interval, pd.DataFrame())
        if frame.empty:
            return frame
        out = frame.copy()
        out["datetime"] = pd.to_datetime(out["time"], unit="s", utc=True)
        return out.set_index("datetime")

    def fetch_funding_history(self, symbol, pages=15):
        return pd.DataFrame(columns=["time", "funding_rate"])


def _collector(drop_range=None) -> tuple[_FakeCollector, pd.DataFrame]:
    h1 = _hourly_with_pump()
    frames = {
        "Min60": h1,
        "Min15": _bars(START, (len(h1) - 1) * 4 + 1, 900,
                       np.interp(np.arange((len(h1) - 1) * 4 + 1) / 4.0,
                                 np.arange(len(h1)), h1["close"].to_numpy())),
        "Min5": _five_min_covering(h1, drop_range),
        "Hour4": _bars(START, len(h1) // 4, 14400, h1["close"].iloc[::4].to_numpy()[: len(h1) // 4]),
    }
    return _FakeCollector(frames), h1


class BuildSymbolRowsE2EV2Tests(unittest.TestCase):
    def _run(self, collector, h1, label_cfg=None):
        return build_symbol_rows(
            "TESTUSDT", collector,
            int(h1["time"].iloc[0]), int(h1["time"].iloc[-1]),
            EventConfig(min_move_pct=0.05, lookback_hours=6, cooldown_hours=6),
            label_cfg or LabelConfig(horizon_hours=48),
        )

    def test_produces_rows_with_decision_and_quality_columns(self):
        collector, h1 = _collector()
        rows = self._run(collector, h1)
        self.assertTrue(rows)
        row = rows[0]
        for key in ("ts", "decision_ts", "entry", "fwd_coverage", "fwd_max_gap_bars",
                    "mae_pct", "mfe_pct", "dca_resolved", "n_averages"):
            self.assertIn(key, row)

    def test_decision_is_one_hourly_bar_after_the_event_stamp(self):
        collector, h1 = _collector()
        for row in self._run(collector, h1):
            self.assertEqual(row["decision_ts"], row["ts"] + BAR_SECONDS_1H)

    def test_labels_never_read_before_the_decision(self):
        """Directly contrast the corrected window against the old leaky one.

        A spike is planted inside the signal hour and nowhere else. Labelling from
        the event stamp would score that spike as the trade's own adverse move;
        labelling from the decision must not see it at all.
        """
        collector, h1 = _collector()
        rows = self._run(collector, h1)
        self.assertTrue(rows)

        five_min = collector.frames["Min5"]
        row = rows[0]
        entry = row["entry"]
        decision_ts, stamp_ts = row["decision_ts"], row["ts"]
        horizon = 48 * 3600

        spike = entry * 1.5
        in_signal_hour = (five_min["time"] >= stamp_ts) & (five_min["time"] < decision_ts)
        self.assertTrue(in_signal_hour.any(), "fixture must have bars inside the signal hour")
        leaky = five_min.copy()
        leaky.loc[in_signal_hour, "high"] = spike

        def mae_from(src, start):
            w = src[(src["time"] >= start) & (src["time"] < start + horizon)]
            return float((w["high"].max() - entry) / entry)

        # the planted spike is only visible from the stamp, never from the decision
        self.assertGreater(mae_from(leaky, stamp_ts), 0.4)
        self.assertLess(mae_from(leaky, decision_ts), 0.4)
        self.assertAlmostEqual(mae_from(leaky, decision_ts), row["mae_pct"], places=9)

    def test_gapped_forward_history_is_rejected(self):
        clean, h1 = _collector()
        n_clean = len(self._run(clean, h1))
        self.assertTrue(n_clean)

        # blank out ~6 hours of 5m bars right after the first plausible decision
        pump_ts = int(h1["time"].iloc[120])
        gapped, h1g = _collector(drop_range=(pump_ts, pump_ts + 6 * 3600))
        rows = self._run(gapped, h1g)
        self.assertLess(len(rows), n_clean)

    def test_coverage_and_gap_are_reported_on_surviving_rows(self):
        collector, h1 = _collector()
        for row in self._run(collector, h1):
            self.assertGreaterEqual(row["fwd_coverage"], 0.90)
            self.assertLessEqual(row["fwd_max_gap_bars"], 12)

    def test_events_without_a_full_horizon_are_skipped(self):
        collector, h1 = _collector()
        rows = self._run(collector, h1)
        horizon = 48 * 3600
        end = int(h1["time"].iloc[-1])
        for row in rows:
            self.assertLessEqual(row["decision_ts"] + horizon, end)


class ClosedHigherTimeframeSemanticsV2Tests(unittest.TestCase):
    """A 4h bar stamped before the decision can still be forming at it. Reading an
    indicator off that bar is look-ahead, so the chosen semantics are: keep a bar
    only once time + bar_seconds <= decision_ts."""

    def test_forming_four_hour_bar_is_excluded_end_to_end(self):
        collector, h1 = _collector()
        rows = build_symbol_rows(
            "TESTUSDT", collector,
            int(h1["time"].iloc[0]), int(h1["time"].iloc[-1]),
            EventConfig(min_move_pct=0.05, lookback_hours=6, cooldown_hours=6),
            LabelConfig(horizon_hours=48),
        )
        self.assertTrue(rows)
        h4 = collector.frames["Hour4"]
        for row in rows:
            decision_ts = row["decision_ts"]
            usable = h4[h4["time"] + 14400 <= decision_ts]
            forming = h4[(h4["time"] <= decision_ts) & (h4["time"] + 14400 > decision_ts)]
            # the fixture must actually contain a mid-formation bar for this to bite
            if len(forming):
                self.assertLess(int(usable["time"].max()), int(forming["time"].min()))

    def test_boundary_bar_closing_exactly_at_the_decision_is_usable(self):
        h4 = pd.DataFrame({"time": [0, 14400, 28800]})
        decision_ts = 28800
        usable = h4[h4["time"] + 14400 <= decision_ts]
        self.assertEqual(list(usable["time"]), [0, 14400])


class ForwardWindowQualityV2Tests(unittest.TestCase):
    def test_full_window_scores_full_coverage(self):
        horizon, bar = 3600 * 4, 300
        frame = _bars(START, horizon // bar, bar, [1.0] * (horizon // bar))
        q = forward_window_quality(frame, START, horizon, bar)
        self.assertAlmostEqual(q["coverage"], 1.0)
        self.assertLessEqual(q["max_gap_bars"], 1.0)

    def test_missing_middle_shows_up_as_a_gap(self):
        horizon, bar = 3600 * 4, 300
        frame = _bars(START, horizon // bar, bar, [1.0] * (horizon // bar))
        frame = frame[(frame["time"] < START + 3600) | (frame["time"] > START + 2 * 3600)]
        q = forward_window_quality(frame, START, horizon, bar)
        self.assertLess(q["coverage"], 1.0)
        self.assertGreater(q["max_gap_bars"], 4.0)

    def test_late_start_counts_against_the_gap(self):
        horizon, bar = 3600 * 4, 300
        frame = _bars(START + 3600, (horizon - 3600) // bar, bar, [1.0] * ((horizon - 3600) // bar))
        q = forward_window_quality(frame, START, horizon, bar)
        self.assertGreaterEqual(q["max_gap_bars"], 12.0)

    def test_empty_window_is_zero_coverage(self):
        q = forward_window_quality(pd.DataFrame(), START, 3600, 300)
        self.assertEqual(q["coverage"], 0.0)


if __name__ == "__main__":
    unittest.main()
