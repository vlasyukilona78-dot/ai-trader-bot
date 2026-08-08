from __future__ import annotations

import json
import sys
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from unittest.mock import patch

import pandas as pd

from app.scan import describe, parse_args, scan_once
from trading.market_data.bar_contract import interval_seconds, last_bar_times
from trading.market_data.mexc_client import _RateLimiter
from trading.market_data.universe import UniverseEntry, UniverseSnapshot
from trading.signals.signal_types import IntentAction, StrategyIntent


class RateLimiterV2Tests(unittest.TestCase):
    """Concurrency without pacing silently lost symbols: at 8 workers MEXC
    dropped 13 of 60 requests and the client returned empty frames that are
    indistinguishable from 'no data'."""

    def test_requests_are_paced_to_the_configured_rate(self):
        limiter = _RateLimiter(rate_per_sec=20.0)
        start = time.monotonic()
        for _ in range(30):
            limiter.acquire()
        elapsed = time.monotonic() - start
        # 30 tokens at 20/s with a full bucket: at least the overflow must wait
        self.assertGreater(elapsed, 0.3)

    def test_limiter_is_thread_safe(self):
        limiter = _RateLimiter(rate_per_sec=50.0)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: limiter.acquire(), range(40)))
        self.assertTrue(True)  # no deadlock or exception


class _FakeFrame:
    def __init__(self, ohlcv, *, timeframe, cutoff, mark_price=999.0):
        self.ohlcv = ohlcv
        # Deliberately different from the last close: the scanner must not use
        # the live ticker in a closed-bar decision.
        self.mark_price = mark_price
        self.candle_cutoff_ts = float(cutoff)
        if ohlcv.empty:
            self.last_bar_open_ts = None
            self.last_bar_close_ts = None
        else:
            self.last_bar_open_ts, self.last_bar_close_ts = last_bar_times(
                ohlcv, interval=timeframe
            )


@dataclass(frozen=True)
class _InvalidContract:
    kind: str = "future_bar"


@dataclass(frozen=True)
class _StaleData:
    frame: pd.DataFrame


def _align_last_close(frame: pd.DataFrame, *, timeframe: str, cutoff: float) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    aligned = frame.copy()
    target_open = pd.Timestamp(float(cutoff), unit="s", tz="UTC") - pd.Timedelta(
        seconds=interval_seconds(timeframe)
    )
    aligned.index = aligned.index + (target_open - aligned.index[-1])
    return aligned


class _FakeFeed:
    def __init__(self, frames: dict):
        self.frames = frames
        self.closed_requests = []

    def fetch_frame(self, symbol, timeframe, candles):
        raise AssertionError("scanner must not call the non-causal fetch_frame")

    def fetch_closed_frame(self, symbol, timeframe, candles, *, as_of):
        self.closed_requests.append((symbol, timeframe, candles, float(as_of)))
        got = self.frames.get(symbol)
        if isinstance(got, Exception):
            raise got
        if isinstance(got, _InvalidContract):
            frame = _FakeFrame(
                _align_last_close(_ohlcv(), timeframe=timeframe, cutoff=as_of),
                timeframe=timeframe,
                cutoff=as_of,
            )
            frame.last_bar_close_ts = float(as_of) + 1.0
            return frame
        if isinstance(got, _StaleData):
            return _FakeFrame(got.frame, timeframe=timeframe, cutoff=as_of)
        ohlcv = got if got is not None else _ohlcv(0)
        if isinstance(ohlcv.index, pd.DatetimeIndex):
            ohlcv = _align_last_close(ohlcv, timeframe=timeframe, cutoff=as_of)
        frame = _FakeFrame(
            ohlcv,
            timeframe=timeframe,
            cutoff=as_of,
        )
        return frame


class _FakeUniverse:
    def __init__(self, symbols, *, funding_rate=None, open_interest=None):
        self._symbols = symbols
        self._funding_rate = funding_rate
        self._open_interest = open_interest

    def refresh(self):
        return UniverseSnapshot(
            entries=[
                UniverseEntry(
                    symbol=symbol,
                    mexc_symbol=symbol.replace("USDT", "_USDT"),
                    turnover_24h_usdt=1_000_000.0 + index,
                    change_24h=0.1 + index / 100.0,
                    funding_rate=self._funding_rate,
                    open_interest=self._open_interest,
                )
                for index, symbol in enumerate(self._symbols)
            ],
            total_contracts=len(self._symbols),
            refreshed_at=1_700_000_000.0,
        )


class _FakeStrategy:
    def __init__(self, action=IntentAction.HOLD, *, errors=()):
        self.action = action
        self.benchmark = "unset"
        self.errors = set(errors)
        self.contexts = []

    def set_benchmark(self, frame):
        self.benchmark = frame

    def generate(self, ctx):
        self.contexts.append(ctx)
        if ctx.symbol in self.errors:
            raise RuntimeError("SENSITIVE_STRATEGY_EXCEPTION_TEXT")
        return StrategyIntent(
            symbol=ctx.symbol,
            action=self.action,
            reason="test",
            stop_loss=1.02,
            take_profit=0.97,
            confidence=0.7,
            metadata={
                "legacy_signal_id": f"wall-clock-{time.time_ns()}",
                "layer_trace": {"layers": {"layer5_tp_sl": {"details": {
                    "entry": float(ctx.mark_price), "sl": 1.02, "tp": 0.97,
                }}}},
            },
        )


class _CaptureJournal:
    def __init__(self):
        self.cycles = []
        self.envelopes = []

    def append_cycle(self, records, *, envelope):
        self.cycles.append(list(records))
        self.envelopes.append(envelope)


class _DuplicateJournal(_CaptureJournal):
    enabled = True

    def append_cycle(self, records, *, envelope):
        super().append_cycle(records, envelope=envelope)
        return False


class _CaptureTracker:
    def __init__(self):
        self.updates = []
        self.records = []

    def update_frame(self, symbol, frame, *, observed_at=None):
        self.updates.append((symbol, frame, observed_at))

    def record_short(self, **kwargs):
        self.records.append(kwargs)

    def expire_stale(self):
        return 0


class _Logger:
    def __init__(self):
        self.warnings = []

    def info(self, *a, **k):
        pass

    def debug(self, *a, **k):
        pass

    def warning(self, msg, *a, **k):
        self.warnings.append(msg)

    def error(self, msg, *a, **k):
        self.warnings.append(msg)


def _ohlcv(n=100):
    frame = pd.DataFrame({
        "open": [1.0] * n, "high": [1.01] * n, "low": [0.99] * n,
        "close": [1.0] * n, "volume": [100.0] * n,
    }, index=pd.date_range("2026-01-01T00:00:00Z", periods=n, freq="h"))
    return frame


class ScanOnceV2Tests(unittest.TestCase):
    def _run(self, frames, symbols, action=IntentAction.HOLD):
        logger = _Logger()
        strategy = _FakeStrategy(action)
        signals = scan_once(universe=_FakeUniverse(symbols), feed=_FakeFeed(frames),
                            strategy=strategy, logger=logger, timeframe="60",
                            candles=320, workers=2)
        return signals, logger, strategy

    def test_returns_only_entry_intents(self):
        frames = {s: _ohlcv() for s in ["BTCUSDT", "AUSDT", "BUSDT"]}
        signals, _, _ = self._run(frames, ["AUSDT", "BUSDT"], IntentAction.SHORT_ENTRY)
        self.assertEqual(len(signals), 2)
        holds, _, _ = self._run(frames, ["AUSDT", "BUSDT"], IntentAction.HOLD)
        self.assertEqual(holds, [])

    def test_short_history_is_skipped_not_evaluated(self):
        frames = {"BTCUSDT": _ohlcv(), "AUSDT": _ohlcv(10)}
        signals, logger, _ = self._run(frames, ["AUSDT"], IntentAction.SHORT_ENTRY)
        self.assertEqual(signals, [])
        self.assertTrue(logger.warnings)  # coverage warning fired

    def test_low_coverage_raises_a_warning(self):
        frames = {"BTCUSDT": _ohlcv(), "AUSDT": _ohlcv()}
        frames["BUSDT"] = pd.DataFrame()
        _, logger, _ = self._run(frames, ["AUSDT", "BUSDT"], IntentAction.SHORT_ENTRY)
        self.assertTrue(any("coverage_low" in str(w) for w in logger.warnings))

    def test_benchmark_failure_does_not_stop_the_scan(self):
        frames = {"BTCUSDT": RuntimeError("down"), "AUSDT": _ohlcv()}
        signals, logger, strategy = self._run(frames, ["AUSDT"], IntentAction.SHORT_ENTRY)
        self.assertEqual(len(signals), 1)
        self.assertIsNone(strategy.benchmark)

    def test_empty_universe_is_reported(self):
        _, logger, _ = self._run({}, [])
        self.assertTrue(any("empty_universe" in str(w) for w in logger.warnings))

    def test_one_closed_cutoff_drives_benchmark_symbols_and_strategy_context(self):
        frames = {s: _ohlcv() for s in ["BTCUSDT", "AUSDT", "BUSDT"]}
        feed = _FakeFeed(frames)
        strategy = _FakeStrategy(IntentAction.HOLD)
        journal = _CaptureJournal()

        signals = scan_once(
            universe=_FakeUniverse(["AUSDT", "BUSDT"]),
            feed=feed,
            strategy=strategy,
            logger=_Logger(),
            timeframe="60",
            candles=320,
            workers=2,
            population_journal=journal,
        )

        self.assertEqual(signals, [])
        cutoffs = {request[3] for request in feed.closed_requests}
        self.assertEqual(len(cutoffs), 1)
        cutoff = cutoffs.pop()
        self.assertEqual([request[0] for request in feed.closed_requests], [
            "BTCUSDT", "AUSDT", "BUSDT",
        ])
        self.assertEqual(len(strategy.contexts), 2)
        for context in strategy.contexts:
            self.assertEqual(context.candle_cutoff_ts, cutoff)
            self.assertEqual(context.mark_price, 1.0)
        self.assertIsNotNone(strategy.benchmark)
        self.assertEqual(float(strategy.benchmark.iloc[-1]["close"]), 1.0)

        records = journal.cycles[0]
        self.assertEqual([record.symbol for record in records], ["AUSDT", "BUSDT"])
        self.assertTrue(all(record.candle_cutoff_ts == cutoff for record in records))
        self.assertTrue(all(record.action == "HOLD" for record in records))
        self.assertEqual([record.cycle_ordinal for record in records], [0, 1])
        self.assertTrue(all(record.cycle_size == 2 for record in records))

    def test_frozen_universe_funding_reaches_strategy_and_feature_snapshot(self):
        frames = {s: _ohlcv() for s in ["BTCUSDT", "AUSDT"]}
        strategy = _FakeStrategy(IntentAction.HOLD)
        journal = _CaptureJournal()

        scan_once(
            universe=_FakeUniverse(
                ["AUSDT"],
                funding_rate=0.00125,
                open_interest=987_654.0,
            ),
            feed=_FakeFeed(frames),
            strategy=strategy,
            logger=_Logger(),
            timeframe="60",
            candles=320,
            workers=1,
            population_journal=journal,
        )

        self.assertEqual(strategy.contexts[0].funding_rate, 0.00125)
        feature_snapshot = journal.cycles[0][0].metadata["feature_snapshot"]
        self.assertEqual(feature_snapshot["values"]["funding_rate"], 0.00125)
        self.assertEqual(feature_snapshot["observed"]["funding_rate"], 1)
        self.assertEqual(feature_snapshot["values"]["open_interest"], 987_654.0)
        self.assertEqual(
            feature_snapshot["source_times"]["bar_cutoff_ts"],
            journal.cycles[0][0].candle_cutoff_ts,
        )
        # The snapshot is the market's identity and carries no wall clock. When
        # the universe answered is real provenance, but it lives beside the
        # snapshot so a slower scan cannot produce a different "market".
        self.assertNotIn("universe_refreshed_at", feature_snapshot["source_times"])
        self.assertNotIn("universe_received_at", feature_snapshot["source_times"])
        feature_provenance = journal.cycles[0][0].metadata["feature_provenance"]
        self.assertEqual(
            feature_provenance["universe_received_at"],
            journal.cycles[0][0].universe_received_at,
        )
        self.assertRegex(feature_provenance["envelope_hash"], r"^[0-9a-f]{64}$")
        self.assertRegex(
            feature_provenance["market_feature_hash"], r"^[0-9a-f]{64}$"
        )
        provenance = journal.cycles[0][0].metadata["provenance"]
        self.assertRegex(provenance["strategy_config_hash"], r"^[0-9a-f]{64}$")
        self.assertRegex(provenance["universe_policy_hash"], r"^[0-9a-f]{64}$")

    def test_population_journal_covers_every_symbol_and_safe_failure_status(self):
        symbols = ["ENTRYUSDT", "HOLDUSDT", "EMPTYUSDT", "SHORTUSDT",
                   "BADBARUSDT", "FETCHUSDT", "ERRORUSDT"]
        frames = {
            "BTCUSDT": _ohlcv(),
            "ENTRYUSDT": _ohlcv(),
            "HOLDUSDT": _ohlcv(),
            "EMPTYUSDT": _ohlcv(0),
            "SHORTUSDT": _ohlcv(10),
            "BADBARUSDT": _InvalidContract(),
            "FETCHUSDT": RuntimeError("SENSITIVE_FETCH_EXCEPTION_TEXT"),
            "ERRORUSDT": _ohlcv(),
        }

        class PerSymbolStrategy(_FakeStrategy):
            def generate(self, ctx):
                self.contexts.append(ctx)
                if ctx.symbol in self.errors:
                    raise RuntimeError("SENSITIVE_STRATEGY_EXCEPTION_TEXT")
                action = (
                    IntentAction.HOLD
                    if ctx.symbol == "HOLDUSDT"
                    else IntentAction.SHORT_ENTRY
                )
                return StrategyIntent(
                    symbol=ctx.symbol,
                    action=action,
                    reason="test",
                    stop_loss=1.02,
                    take_profit=0.97,
                    confidence=0.7,
                )

        journal = _CaptureJournal()
        scan_once(
            universe=_FakeUniverse(symbols),
            feed=_FakeFeed(frames),
            strategy=PerSymbolStrategy(errors={"ERRORUSDT"}),
            logger=_Logger(),
            timeframe="60",
            candles=320,
            workers=4,
            population_journal=journal,
        )

        records = journal.cycles[0]
        self.assertEqual([record.symbol for record in records], symbols)
        by_symbol = {record.symbol: record for record in records}
        self.assertEqual(by_symbol["ENTRYUSDT"].status, "evaluated")
        self.assertEqual(by_symbol["ENTRYUSDT"].action, "SHORT_ENTRY")
        self.assertEqual(by_symbol["HOLDUSDT"].status, "evaluated")
        self.assertEqual(by_symbol["HOLDUSDT"].action, "HOLD")
        self.assertEqual(by_symbol["EMPTYUSDT"].status, "no_data")
        self.assertEqual(by_symbol["SHORTUSDT"].status, "short_history")
        self.assertEqual(by_symbol["BADBARUSDT"].status, "invalid_bar_contract")
        self.assertEqual(by_symbol["BADBARUSDT"].error_code, "BarContractError")
        self.assertEqual(by_symbol["FETCHUSDT"].status, "data_error")
        self.assertEqual(by_symbol["FETCHUSDT"].error_code, "RuntimeError")
        self.assertEqual(by_symbol["ERRORUSDT"].status, "strategy_error")
        self.assertEqual(by_symbol["ERRORUSDT"].error_code, "RuntimeError")

        encoded = json.dumps(
            [record.as_dict() for record in records],
            ensure_ascii=False,
            sort_keys=True,
        )
        self.assertNotIn("SENSITIVE_FETCH_EXCEPTION_TEXT", encoded)
        self.assertNotIn("SENSITIVE_STRATEGY_EXCEPTION_TEXT", encoded)

    def test_causal_ids_are_stable_and_records_keep_universe_order(self):
        frames = {s: _ohlcv() for s in ["BTCUSDT", "AUSDT", "BUSDT"]}
        journals = []
        for _ in range(2):
            journal = _CaptureJournal()
            with patch("app.scan.time.time", return_value=1_800_000_123.0):
                scan_once(
                    universe=_FakeUniverse(["BUSDT", "AUSDT"]),
                    feed=_FakeFeed(frames),
                    strategy=_FakeStrategy(IntentAction.SHORT_ENTRY),
                    logger=_Logger(),
                    timeframe="60",
                    candles=320,
                    workers=2,
                    population_journal=journal,
                )
            journals.append(journal.cycles[0])

        first, second = journals
        self.assertEqual([record.symbol for record in first], ["BUSDT", "AUSDT"])
        self.assertEqual(
            [record.snapshot_id for record in first],
            [record.snapshot_id for record in second],
        )
        self.assertTrue(all(
            "legacy_signal_id" not in record.metadata.get("strategy", {})
            for record in first
        ))

    def test_tracker_uses_decision_time_closed_bar_open_and_snapshot_id(self):
        frames = {"BTCUSDT": _ohlcv(), "AUSDT": _ohlcv()}
        journal = _CaptureJournal()
        tracker = _CaptureTracker()
        with patch("app.scan.time.time", return_value=1_800_000_123.0):
            scan_once(
                universe=_FakeUniverse(["AUSDT"]),
                feed=_FakeFeed(frames),
                strategy=_FakeStrategy(IntentAction.SHORT_ENTRY),
                logger=_Logger(),
                timeframe="60",
                candles=320,
                workers=1,
                population_journal=journal,
                tracker=tracker,
                alerters=(object(),),
            )

        record = journal.cycles[0][0]
        observation = tracker.records[0]
        self.assertEqual(observation["signal_id"], record.snapshot_id)
        self.assertEqual(observation["signal_ts"], record.decision_ts)
        self.assertEqual(observation["signal_bar_ts"], record.base_bar_open_ts)
        self.assertFalse(observation["delivered"])

    def test_stale_and_gapped_frames_fail_closed_as_data_quality_errors(self):
        stale = _StaleData(_ohlcv())
        gapped = _ohlcv().drop(_ohlcv().index[-2])
        journal = _CaptureJournal()

        scan_once(
            universe=_FakeUniverse(["STALEUSDT", "GAPUSDT"]),
            feed=_FakeFeed({"BTCUSDT": _ohlcv(), "STALEUSDT": stale, "GAPUSDT": gapped}),
            strategy=_FakeStrategy(IntentAction.SHORT_ENTRY),
            logger=_Logger(),
            timeframe="60",
            candles=320,
            workers=2,
            population_journal=journal,
        )

        by_symbol = {record.symbol: record for record in journal.cycles[0]}
        self.assertEqual(by_symbol["STALEUSDT"].status, "data_quality_error")
        self.assertEqual(by_symbol["GAPUSDT"].status, "data_quality_error")
        self.assertIsNone(by_symbol["STALEUSDT"].base_bar_open_ts)

    def test_duplicate_population_cycle_suppresses_repeated_signal(self):
        frames = {"BTCUSDT": _ohlcv(), "AUSDT": _ohlcv()}
        journal = _DuplicateJournal()

        signals = scan_once(
            universe=_FakeUniverse(["AUSDT"]),
            feed=_FakeFeed(frames),
            strategy=_FakeStrategy(IntentAction.SHORT_ENTRY),
            logger=_Logger(),
            timeframe="60",
            candles=320,
            workers=1,
            population_journal=journal,
        )

        self.assertEqual(signals, [])
        self.assertEqual(len(journal.cycles), 1)

    def test_long_entry_is_returned_but_not_recorded_as_short_observation(self):
        frames = {"BTCUSDT": _ohlcv(), "AUSDT": _ohlcv()}
        tracker = _CaptureTracker()
        signals = scan_once(
            universe=_FakeUniverse(["AUSDT"]),
            feed=_FakeFeed(frames),
            strategy=_FakeStrategy(IntentAction.LONG_ENTRY),
            logger=_Logger(),
            timeframe="60",
            candles=320,
            workers=1,
            tracker=tracker,
        )
        self.assertEqual(len(signals), 1)
        self.assertEqual(tracker.records, [])


class ScanCliV2Tests(unittest.TestCase):
    def test_population_journal_defaults_and_disable_switch(self):
        with patch.object(sys, "argv", ["scan"]):
            defaults = parse_args()
        self.assertEqual(
            defaults.population_journal,
            "data/runtime/mexc_population_decisions_v4.jsonl",
        )
        self.assertFalse(defaults.disable_population_journal)

        with patch.object(sys, "argv", ["scan", "--disable-population-journal"]):
            disabled = parse_args()
        self.assertTrue(disabled.disable_population_journal)


class DescribeV2Tests(unittest.TestCase):
    def test_includes_margin_terms_and_safe_leverage(self):
        intent = StrategyIntent(
            symbol="XUSDT", action=IntentAction.SHORT_ENTRY, reason="t",
            metadata={"layer_trace": {"layers": {"layer5_tp_sl": {"details": {
                "entry": 1.0, "sl": 1.02, "tp": 0.97,
                "stop_pct_of_margin": 200.0, "target_pct_of_margin": 300.0,
                "max_safe_leverage": 50.0,
            }}}}},
        )
        text = describe("XUSDT", intent)
        self.assertIn("SHORT_ENTRY XUSDT", text)
        self.assertIn("% of margin", text)
        self.assertIn("max safe leverage 50x", text)

    def test_missing_metadata_does_not_raise(self):
        intent = StrategyIntent(symbol="XUSDT", action=IntentAction.SHORT_ENTRY, reason="t")
        self.assertIn("XUSDT", describe("XUSDT", intent))


if __name__ == "__main__":
    unittest.main()
