"""Signals-only MEXC scanner: public data in, alerts out, no order placement.

The main runtime still carries the full execution stack - adapter, state machine,
order lifecycle - which exists to trade an account by itself. That is not how
these signals are used: the bot finds the setup and the trade is entered by hand.
Running the execution machinery for that would mean private API keys and an order
path that nothing needs, so this entry point deliberately has neither. It reads
public MEXC data, applies the same strategy, and reports.

Nothing here can place, modify or cancel an order.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal, Mapping

import pandas as pd

from ai.reversal.feature_contract import build_runtime_feature_snapshot, configuration_hash
from core.indicators import compute_indicators
from trading.market_data.bar_contract import (
    BarContractError,
    closed_boundary_ts,
    interval_seconds,
    last_bar_times,
)
from trading.market_data.feed import MarketDataFeed
from trading.market_data.mexc_client import MexcContractClient
from trading.market_data.source_timing import SourceTiming
from trading.market_data.timeframe_cache import HigherTimeframeCache
from trading.market_data.universe import SymbolUniverse, UniverseConfig
from trading.metrics.cycle_envelope import CycleEnvelope
from trading.metrics.logging import setup_logging
from trading.metrics.population_journal import (
    PopulationDecision,
    PopulationJournal,
    make_cycle_id,
    safe_error_code,
)
from trading.signals.layered_strategy import LayeredPumpStrategy
from trading.signals.signal_types import IntentAction, StrategyIntent
from trading.signals.strategy_interface import StrategyContext
from trading.state.models import TradeState
from trading.state.signal_observation_tracker import SignalObservationTracker


_MIN_HISTORY_BARS = 80
_REQUIRED_OHLCV_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})
_PopulationStatus = Literal[
    "evaluated",
    "no_data",
    "short_history",
    "invalid_bar_contract",
    "data_error",
    "data_quality_error",
    "strategy_error",
]


class InvalidStrategyIntentError(ValueError):
    """Raised when a strategy result cannot be journalled safely."""


class MarketDataQualityError(ValueError):
    """Raised when structurally valid closed bars are stale, gapped, or invalid."""


@dataclass(frozen=True)
class _ScanEvaluation:
    symbol: str
    status: _PopulationStatus
    decision_ts: float
    base_bar_open_ts: float | None
    base_bar_close_ts: float | None
    bar_count: int = 0
    mark_price: float = 0.0
    intent: StrategyIntent | None = None
    frame: pd.DataFrame | None = None
    stage: str = ""
    error_code: str | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MEXC signal scanner (no execution)")
    p.add_argument("--timeframe", default="60", help="Bar size for the entry frame")
    p.add_argument("--candles", type=int, default=320)
    p.add_argument("--min-turnover", type=float, default=400_000.0)
    p.add_argument("--max-turnover", type=float, default=100_000_000.0)
    p.add_argument("--max-symbols", type=int, default=0, help="0 = whole filtered universe")
    p.add_argument("--max-min-notional", type=float, default=0.0,
                   help="Skip contracts whose minimum lot exceeds this notional")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--interval-sec", type=int, default=300)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--universe-refresh-sec", type=int, default=900)
    p.add_argument("--observations", default="data/runtime/observations.json",
                   help="Where forward outcomes of delivered signals are recorded")
    p.add_argument("--observe-minutes", type=int, default=120,
                   help="How long each signal is followed after delivery")
    p.add_argument(
        "--population-journal",
        default="data/runtime/mexc_population_decisions.jsonl",
        help="Append one causal decision row for every point-in-time universe symbol",
    )
    p.add_argument(
        "--disable-population-journal",
        action="store_true",
        help="Disable the runtime-population JSONL journal",
    )
    return p.parse_args()


def _finite_number(value, *, field: str) -> float:
    if isinstance(value, bool):
        raise BarContractError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BarContractError(f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise BarContractError(f"{field} must be a finite number")
    return number


def _fetch_closed_frame(*, feed, symbol: str, timeframe: str, candles: int, cutoff: float):
    fetch = getattr(feed, "fetch_closed_frame", None)
    if not callable(fetch):
        raise BarContractError("feed does not implement fetch_closed_frame")
    frame = fetch(symbol=symbol, timeframe=timeframe, candles=candles, as_of=cutoff)
    ohlcv = getattr(frame, "ohlcv", None)
    if not isinstance(ohlcv, pd.DataFrame):
        raise BarContractError("closed frame must contain a pandas OHLCV frame")

    reported_cutoff = _finite_number(
        getattr(frame, "candle_cutoff_ts", None), field="candle_cutoff_ts"
    )
    if not math.isclose(reported_cutoff, cutoff, rel_tol=0.0, abs_tol=1e-6):
        raise BarContractError("closed frame cutoff differs from the scan cutoff")

    if ohlcv.empty:
        if getattr(frame, "last_bar_open_ts", None) is not None:
            raise BarContractError("empty closed frame exposes a last bar open")
        if getattr(frame, "last_bar_close_ts", None) is not None:
            raise BarContractError("empty closed frame exposes a last bar close")
        return ohlcv, None, None, 0.0

    missing = _REQUIRED_OHLCV_COLUMNS.difference(ohlcv.columns)
    if missing:
        raise BarContractError("closed frame is missing required OHLCV columns")

    actual_open_ts, actual_close_ts = last_bar_times(ohlcv, interval=timeframe)
    reported_open_ts = _finite_number(
        getattr(frame, "last_bar_open_ts", None), field="last_bar_open_ts"
    )
    reported_close_ts = _finite_number(
        getattr(frame, "last_bar_close_ts", None), field="last_bar_close_ts"
    )
    if not math.isclose(reported_open_ts, actual_open_ts, rel_tol=0.0, abs_tol=1e-6):
        raise BarContractError("last bar open metadata differs from the OHLCV frame")
    if not math.isclose(reported_close_ts, actual_close_ts, rel_tol=0.0, abs_tol=1e-6):
        raise BarContractError("last bar close metadata differs from the OHLCV frame")
    if actual_close_ts > cutoff:
        raise BarContractError("closed frame contains a bar beyond the scan cutoff")
    if not math.isclose(actual_close_ts, cutoff, rel_tol=0.0, abs_tol=1e-6):
        raise MarketDataQualityError("last closed bar is stale at the scan cutoff")

    numeric = ohlcv.loc[:, sorted(_REQUIRED_OHLCV_COLUMNS)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    finite = numeric.map(lambda value: math.isfinite(float(value)))
    if not bool(finite.to_numpy().all()):
        raise MarketDataQualityError("OHLCV contains non-finite values")
    if bool((numeric["volume"] < 0).any()):
        raise MarketDataQualityError("OHLCV volume must not be negative")
    if bool((numeric["high"] < numeric[["open", "close"]].max(axis=1)).any()):
        raise MarketDataQualityError("OHLCV high is below open or close")
    if bool((numeric["low"] > numeric[["open", "close"]].min(axis=1)).any()):
        raise MarketDataQualityError("OHLCV low is above open or close")
    if bool((numeric["high"] < numeric["low"]).any()):
        raise MarketDataQualityError("OHLCV high is below low")

    expected_step = interval_seconds(timeframe)
    if len(ohlcv.index) > 1:
        index_ns = ohlcv.index.tz_convert("UTC").asi8
        deltas = (index_ns[1:] - index_ns[:-1]) / 1_000_000_000
        if any(not math.isclose(float(delta), expected_step, rel_tol=0.0, abs_tol=1e-6) for delta in deltas):
            raise MarketDataQualityError("OHLCV cadence contains a gap or overlap")

    mark_price = _finite_number(ohlcv.iloc[-1]["close"], field="last closed close")
    if mark_price <= 0:
        raise BarContractError("last closed close must be positive")
    return ohlcv, actual_open_ts, actual_close_ts, mark_price


def _validate_intent(intent, *, symbol: str) -> StrategyIntent:
    if not isinstance(intent, StrategyIntent):
        raise InvalidStrategyIntentError("strategy must return StrategyIntent")
    if intent.symbol != symbol:
        raise InvalidStrategyIntentError("strategy intent symbol differs from its input")
    if not isinstance(intent.action, IntentAction):
        raise InvalidStrategyIntentError("strategy intent action is invalid")
    if not isinstance(intent.reason, str) or not intent.reason:
        raise InvalidStrategyIntentError("strategy intent reason is invalid")
    try:
        confidence = float(intent.confidence)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidStrategyIntentError("strategy confidence is invalid") from exc
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise InvalidStrategyIntentError("strategy confidence is invalid")
    if not isinstance(intent.metadata, Mapping):
        raise InvalidStrategyIntentError("strategy metadata is invalid")
    return intent


def _optional_finite(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


_UNSAFE_JOURNAL_VALUE = object()
_SENSITIVE_KEY_PARTS = ("token", "secret", "password", "authorization", "cookie", "proxy")


def _safe_journal_value(value, *, depth: int = 0):
    """Copy a bounded JSON-safe subset without serializing arbitrary objects."""

    if depth > 7:
        return _UNSAFE_JOURNAL_VALUE
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else _UNSAFE_JOURNAL_VALUE
    if isinstance(value, str):
        return value[:2_048]
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for raw_key, item in list(value.items())[:256]:
            if not isinstance(raw_key, str) or not raw_key:
                continue
            key = raw_key[:128]
            lowered = key.casefold()
            if key == "legacy_signal_id" or any(part in lowered for part in _SENSITIVE_KEY_PARTS):
                continue
            safe_item = _safe_journal_value(item, depth=depth + 1)
            if safe_item is not _UNSAFE_JOURNAL_VALUE:
                result[key] = safe_item
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in list(value)[:256]:
            safe_item = _safe_journal_value(item, depth=depth + 1)
            if safe_item is not _UNSAFE_JOURNAL_VALUE:
                result.append(safe_item)
        return result
    return _UNSAFE_JOURNAL_VALUE


def _universe_timing(snapshot, *, fallback: float) -> SourceTiming:
    """Timing of the universe response, never its pre-request TTL anchor.

    ``refreshed_at`` is read before the request is sent, so treating it as the
    moment the data was known would date every ticker value earlier than the
    process could possibly have held it. A snapshot without real response timing
    (an injected stub, or one predating this contract) is dated at the scan
    itself: when the true instant is unknown the safe assumption is later, not
    earlier.
    """

    started = _optional_finite(getattr(snapshot, "request_started_at", None))
    received = _optional_finite(getattr(snapshot, "received_at", None))
    if started is None or received is None or received <= 0.0 or started <= 0.0:
        started = received = fallback
    return SourceTiming(
        source="universe",
        request_started_at=min(started, received),
        received_at=received,
    )


def _timed_source(
    source: str,
    *,
    request_started_at: float,
    received_at: float,
    source_as_of: float | None,
    error: BaseException | None = None,
) -> SourceTiming:
    if error is None:
        return SourceTiming(
            source=source,
            request_started_at=request_started_at,
            received_at=received_at,
            source_as_of=source_as_of,
        )
    return SourceTiming(
        source=source,
        request_started_at=request_started_at,
        received_at=received_at,
        status="error",
        source_as_of=source_as_of,
        error_code=safe_error_code(error),
    )


def _universe_metadata(snapshot) -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {}
    for entry in getattr(snapshot, "entries", ()):
        symbol = str(getattr(entry, "symbol", "") or "")
        if not symbol:
            continue
        metadata[symbol] = {
            "mexc_symbol": str(getattr(entry, "mexc_symbol", "") or ""),
            "turnover_24h_usdt": _optional_finite(getattr(entry, "turnover_24h_usdt", None)),
            "change_24h": _optional_finite(getattr(entry, "change_24h", None)),
            "funding_rate": _optional_finite(getattr(entry, "funding_rate", None)),
            "open_interest": _optional_finite(getattr(entry, "open_interest", None)),
            "last_price": _optional_finite(getattr(entry, "last_price", None)),
            "min_notional_usdt": _optional_finite(getattr(entry, "min_notional_usdt", None)),
            "max_leverage": _optional_finite(getattr(entry, "max_leverage", None)),
        }
    return metadata


def _population_record(
    *,
    result: _ScanEvaluation,
    envelope: CycleEnvelope,
    universe_refreshed_at: float,
    scan_observed_at: float,
    universe_meta: Mapping[str, object],
    benchmark_status: str,
    cycle_ordinal: int,
    cycle_size: int,
) -> PopulationDecision:
    metadata: dict[str, object] = {
        "universe": dict(universe_meta),
        "base": {
            "bar_count": result.bar_count,
            "mark_price": result.mark_price if result.mark_price > 0 else None,
        },
        "benchmark_status": benchmark_status,
        "provenance": {
            "strategy_config_hash": envelope.strategy_config_hash,
            "universe_policy_hash": envelope.universe_policy_hash,
        },
    }
    if result.stage:
        metadata["stage"] = result.stage

    if result.status == "evaluated" and result.intent is not None:
        # The legacy signal ID contains wall-clock milliseconds. Persisting it
        # would make causal input/snapshot IDs change for identical bar inputs.
        layer_trace = _safe_journal_value(result.intent.metadata.get("layer_trace"))
        if isinstance(layer_trace, Mapping):
            metadata["strategy"] = {"layer_trace": layer_trace}
        metadata["stop_loss"] = _optional_finite(result.intent.stop_loss)
        metadata["take_profit"] = _optional_finite(result.intent.take_profit)
        action = result.intent.action.value
        reason = result.intent.reason
        confidence = float(result.intent.confidence)
    else:
        action = IntentAction.HOLD.value
        reason = result.status
        confidence = 0.0

    # One versioned extractor is shared by runtime capture and future dataset /
    # inference code.  Missing observations remain explicit nulls plus masks;
    # zero is never used as a silent substitute for an unavailable source.
    # The envelope is written once per cycle as a header record. Copying it here
    # made the journal quadratic in universe size and pushed the ordered symbol
    # list past the bounds that keep arbitrary per-row metadata safe.
    metadata["feature_snapshot"] = build_runtime_feature_snapshot(
        metadata,
        bar_cutoff_ts=envelope.candle_cutoff_ts,
        universe_refreshed_at=universe_refreshed_at,
    )

    return PopulationDecision.create(
        cycle_id=envelope.cycle_id,
        universe_refreshed_at=universe_refreshed_at,
        universe_request_started_at=envelope.universe_timing.request_started_at,
        universe_received_at=envelope.universe_timing.received_at,
        scan_observed_at=scan_observed_at,
        candle_cutoff_ts=envelope.candle_cutoff_ts,
        decision_ts=result.decision_ts,
        ranking_ready_ts=envelope.ranking_ready_ts,
        cycle_completed_ts=envelope.cycle_completed_ts,
        actionable_ts=envelope.actionable_ts,
        entry_eligible_ts=envelope.entry_eligible_ts,
        entry_bar_open_ts=envelope.entry_bar_open_ts,
        symbol=result.symbol,
        timeframe=envelope.timeframe,
        status=result.status,
        base_bar_open_ts=result.base_bar_open_ts,
        base_bar_close_ts=result.base_bar_close_ts,
        action=action,
        reason=reason,
        confidence=confidence,
        metadata=metadata,
        cycle_ordinal=cycle_ordinal,
        cycle_size=cycle_size,
        error_code=result.error_code,
    )


def scan_once(*, universe, feed, strategy, logger, timeframe, candles, workers,
              tracker=None, alerters=(), population_journal=None) -> list:
    cycle_started_at = time.time()
    # Freeze the causal cutoff before the universe request. Deriving it afterwards
    # let a refresh that happened to cross a bar boundary produce a cutoff later
    # than the cycle's own start, which is both a false provenance claim and an
    # envelope-invariant crash.
    candle_cutoff_ts = closed_boundary_ts(cycle_started_at, timeframe)
    snapshot = universe.refresh()
    scan_observed_at = max(time.time(), cycle_started_at)
    symbols = list(snapshot.symbols)
    universe_timing = _universe_timing(snapshot, fallback=scan_observed_at)

    if not symbols:
        # An empty cycle is still a cycle. Returning silently would leave a hole
        # that no later completeness check could distinguish from a scan that
        # never ran.
        empty_completed_at = time.time()
        empty_envelope = CycleEnvelope.build(
            cycle_id=make_cycle_id(
                timeframe=timeframe,
                candle_cutoff_ts=candle_cutoff_ts,
                universe_received_at=universe_timing.received_at,
                universe_symbols=(),
            ),
            timeframe=timeframe,
            cycle_started_at=cycle_started_at,
            candle_cutoff_ts=candle_cutoff_ts,
            universe_symbols=(),
            universe_timing=universe_timing,
            source_timings=(universe_timing,),
            strategy_config_hash=configuration_hash(
                strategy.configuration_snapshot()
                if hasattr(strategy, "configuration_snapshot")
                else getattr(strategy, "config", None),
                component="mexc_signal_strategy",
            ),
            universe_policy_hash=configuration_hash(
                getattr(universe, "config", None), component="mexc_universe_policy"
            ),
            ranking_ready_ts=scan_observed_at,
            cycle_completed_ts=empty_completed_at,
            status="empty_universe",
        )
        # Durable evidence, not just a log line: a gap in the journal cannot be
        # told apart from a scan that never ran.
        if population_journal is not None and getattr(population_journal, "enabled", True):
            try:
                population_journal.append_cycle((), envelope=empty_envelope)
            except Exception as exc:
                logger.warning(
                    "empty_cycle_journal_failed=%s",
                    safe_error_code(exc),
                    extra={"event": "scan"},
                )
        logger.warning(
            "empty_universe cycle=%s entry_bar_open_ts=%.0f",
            empty_envelope.cycle_id,
            empty_envelope.entry_bar_open_ts,
            extra={"event": "scan"},
        )
        return []

    universe_refreshed_at = _finite_number(
        getattr(snapshot, "refreshed_at", None), field="universe_refreshed_at"
    )
    cycle_id = make_cycle_id(
        timeframe=timeframe,
        candle_cutoff_ts=candle_cutoff_ts,
        universe_received_at=universe_timing.received_at,
        universe_symbols=symbols,
    )
    universe_metadata = _universe_metadata(snapshot)
    strategy_config = (
        strategy.configuration_snapshot()
        if hasattr(strategy, "configuration_snapshot")
        else getattr(strategy, "config", None)
    )
    strategy_config_hash = configuration_hash(
        strategy_config,
        component="mexc_signal_strategy",
    )
    universe_policy_hash = configuration_hash(
        getattr(universe, "config", None),
        component="mexc_universe_policy",
    )

    # Freeze the volatility distribution before evaluating anyone, so a
    # candidate's fate does not depend on its position in the scan order.
    if hasattr(strategy, "begin_sweep"):
        strategy.begin_sweep()

    benchmark_started_at = time.time()
    try:
        btc, _, _, _ = _fetch_closed_frame(
            feed=feed,
            symbol="BTCUSDT",
            timeframe=timeframe,
            candles=candles,
            cutoff=candle_cutoff_ts,
        )
        strategy.set_benchmark(btc if not btc.empty else None)
        benchmark_status = "available" if not btc.empty else "no_data"
        benchmark_timing = _timed_source(
            "benchmark",
            request_started_at=benchmark_started_at,
            received_at=time.time(),
            source_as_of=candle_cutoff_ts,
        )
    except Exception as exc:
        strategy.set_benchmark(None)
        benchmark_status = f"error:{safe_error_code(exc)}"
        # A source that failed still consumed time the cycle waited for.
        benchmark_timing = _timed_source(
            "benchmark",
            request_started_at=benchmark_started_at,
            received_at=time.time(),
            source_as_of=candle_cutoff_ts,
            error=exc,
        )
        logger.warning(
            "benchmark_unavailable=%s",
            safe_error_code(exc),
            extra={"event": "scan"},
        )

    def decision_time() -> float:
        # A clock correction must not place the recorded decision before the
        # already-observed causal cutoff.
        return max(time.time(), scan_observed_at, candle_cutoff_ts)

    def evaluate(symbol: str) -> _ScanEvaluation:
        try:
            raw, bar_open_ts, bar_close_ts, mark_price = _fetch_closed_frame(
                feed=feed,
                symbol=symbol,
                timeframe=timeframe,
                candles=candles,
                cutoff=candle_cutoff_ts,
            )
        except MarketDataQualityError as exc:
            return _ScanEvaluation(
                symbol=symbol,
                status="data_quality_error",
                decision_ts=decision_time(),
                base_bar_open_ts=None,
                base_bar_close_ts=None,
                stage="market_data",
                error_code=safe_error_code(exc),
            )
        except BarContractError as exc:
            return _ScanEvaluation(
                symbol=symbol,
                status="invalid_bar_contract",
                decision_ts=decision_time(),
                base_bar_open_ts=None,
                base_bar_close_ts=None,
                stage="market_data",
                error_code=safe_error_code(exc),
            )
        except Exception as exc:
            return _ScanEvaluation(
                symbol=symbol,
                status="data_error",
                decision_ts=decision_time(),
                base_bar_open_ts=None,
                base_bar_close_ts=None,
                stage="market_data",
                error_code=safe_error_code(exc),
            )

        if raw.empty:
            return _ScanEvaluation(
                symbol=symbol,
                status="no_data",
                decision_ts=decision_time(),
                base_bar_open_ts=bar_open_ts,
                base_bar_close_ts=bar_close_ts,
                stage="market_data",
            )
        if len(raw) < _MIN_HISTORY_BARS:
            return _ScanEvaluation(
                symbol=symbol,
                status="short_history",
                decision_ts=decision_time(),
                base_bar_open_ts=bar_open_ts,
                base_bar_close_ts=bar_close_ts,
                bar_count=len(raw),
                mark_price=mark_price,
                frame=raw,
                stage="market_data",
            )

        try:
            enriched = compute_indicators(raw)
            symbol_universe_meta = universe_metadata.get(symbol, {})
            funding_rate = _optional_finite(symbol_universe_meta.get("funding_rate"))
            intent = _validate_intent(
                strategy.generate(
                    StrategyContext(
                        symbol=symbol,
                        market_ohlcv=enriched,
                        # Signals are decided at the closed-bar cutoff. A live
                        # ticker would reintroduce data that did not exist at it.
                        mark_price=mark_price,
                        exchange=None,
                        # No position is held by the bot; every symbol is evaluated flat.
                        synced_state=TradeState.FLAT,
                        sentiment_index=50.0,
                        sentiment_source="fallback_neutral_50",
                        # This is the same frozen point-in-time snapshot already
                        # journalled for the cycle, not a later per-symbol fetch.
                        funding_rate=funding_rate,
                        candle_cutoff_ts=candle_cutoff_ts,
                    )
                ),
                symbol=symbol,
            )
            generated_at = decision_time()
            return _ScanEvaluation(
                symbol=symbol,
                status="evaluated",
                decision_ts=generated_at,
                base_bar_open_ts=bar_open_ts,
                base_bar_close_ts=bar_close_ts,
                bar_count=len(enriched),
                mark_price=mark_price,
                intent=intent,
                frame=enriched,
                stage="strategy",
            )
        except Exception as exc:
            return _ScanEvaluation(
                symbol=symbol,
                status="strategy_error",
                decision_ts=decision_time(),
                base_bar_open_ts=bar_open_ts,
                base_bar_close_ts=bar_close_ts,
                bar_count=len(raw),
                mark_price=mark_price,
                frame=raw,
                stage="strategy",
                error_code=safe_error_code(exc),
            )

    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(evaluate, symbols))
    # Base OHLCV and the higher-timeframe cache are both read inside `evaluate`,
    # so this span bounds every per-symbol market request in the cycle.
    market_data_timing = _timed_source(
        "market_data",
        request_started_at=started,
        received_at=time.time(),
        source_as_of=candle_cutoff_ts,
    )

    # The cycle becomes comparable only once its last symbol has been decided,
    # and reachable only once that decision set is sealed. Neither depends on
    # which worker finished first.
    # Bounded below by the last per-symbol decision and by the moment the last
    # market request answered: a cohort is not comparable while either is still
    # outstanding, and which of the two lands later depends on scheduling.
    ranking_ready_ts = max(
        max(result.decision_ts for result in results),
        market_data_timing.received_at,
    )
    cycle_completed_ts = max(time.time(), ranking_ready_ts)
    envelope = CycleEnvelope.build(
        cycle_id=cycle_id,
        timeframe=timeframe,
        cycle_started_at=cycle_started_at,
        candle_cutoff_ts=candle_cutoff_ts,
        universe_symbols=symbols,
        universe_timing=universe_timing,
        source_timings=(universe_timing, benchmark_timing, market_data_timing),
        strategy_config_hash=strategy_config_hash,
        universe_policy_hash=universe_policy_hash,
        ranking_ready_ts=ranking_ready_ts,
        cycle_completed_ts=cycle_completed_ts,
    )

    cycle_size = len(results)
    records = [
        _population_record(
            result=result,
            envelope=envelope,
            universe_refreshed_at=universe_refreshed_at,
            scan_observed_at=scan_observed_at,
            universe_meta=universe_metadata.get(result.symbol, {}),
            benchmark_status=benchmark_status,
            cycle_ordinal=ordinal,
            cycle_size=cycle_size,
        )
        for ordinal, result in enumerate(results)
    ]
    duplicate_cycle = False
    if population_journal is not None and getattr(population_journal, "enabled", True):
        duplicate_cycle = population_journal.append_cycle(records, envelope=envelope) is False
        if duplicate_cycle:
            logger.info(
                "duplicate_population_cycle_suppressed=%s",
                cycle_id,
                extra={"event": "scan"},
            )

    records_by_symbol = {record.symbol: record for record in records}
    results_by_symbol = {result.symbol: result for result in results}
    for result in results:
        if result.intent is None or not isinstance(result.intent.metadata, dict):
            continue
        record = records_by_symbol[result.symbol]
        result.intent.metadata["population_tracking"] = {
            "snapshot_id": record.snapshot_id,
            "cohort_id": record.cycle_id,
            "decision_ts": record.decision_ts,
            "base_bar_open_ts": record.base_bar_open_ts,
            "actionable_ts": record.actionable_ts,
            "entry_bar_open_ts": record.entry_bar_open_ts,
        }

    signals = [
        (result.symbol, result.intent, result.frame)
        for result in results
        if result.status == "evaluated"
        and result.intent is not None
        and result.frame is not None
        and result.intent.action in (IntentAction.SHORT_ENTRY, IntentAction.LONG_ENTRY)
        and not duplicate_cycle
    ]

    if tracker is not None:
        # Feed every available closed frame forward, not just the ones that
        # fired. An observation opened earlier still needs subsequent bars.
        for result in results:
            if result.frame is None or result.frame.empty:
                continue
            try:
                tracker.update_frame(
                    result.symbol,
                    result.frame,
                    observed_at=result.decision_ts,
                )
            except Exception as exc:
                logger.debug(
                    "tracker_update_failed=%s err=%s",
                    result.symbol,
                    safe_error_code(exc),
                    extra={"event": "scan"},
                )

        for sym, intent, _ in signals:
            if intent.action is not IntentAction.SHORT_ENTRY:
                continue
            meta = intent.metadata if isinstance(intent.metadata, dict) else {}
            layer5 = (meta.get("layer_trace", {}).get("layers", {})
                      .get("layer5_tp_sl", {}).get("details", {}))
            result = results_by_symbol[sym]
            record = records_by_symbol[sym]
            try:
                tracker.record_short(
                    signal_id=record.snapshot_id,
                    symbol=sym,
                    phase=intent.action.value,
                    entry=float(layer5.get("entry") or result.mark_price),
                    take_profit=float(layer5.get("tp") or intent.take_profit or 0.0),
                    stop_loss=float(layer5.get("sl") or intent.stop_loss or 0.0),
                    signal_ts=result.decision_ts,
                    signal_bar_ts=result.base_bar_open_ts,
                    # Delivery happens after scan_once returns. Main calls
                    # record_short again with the same snapshot ID only after a
                    # channel confirms success.
                    delivered=False,
                    candidate_source="mexc_scan",
                )
            except Exception as exc:
                logger.warning(
                    "tracker_record_failed=%s err=%s",
                    sym,
                    safe_error_code(exc),
                    extra={"event": "scan"},
                )

        try:
            expired = tracker.expire_stale()
            if expired:
                logger.info("observations_expired=%d", expired, extra={"event": "scan"})
        except Exception as exc:
            logger.debug(
                "tracker_expire_failed=%s",
                safe_error_code(exc),
                extra={"event": "scan"},
            )

    evaluated_count = sum(result.status == "evaluated" for result in results)
    skipped = {
        status: sum(result.status == status for result in results)
        for status in (
            "no_data",
            "short_history",
            "invalid_bar_contract",
            "data_error",
            "data_quality_error",
            "strategy_error",
        )
    }

    # Dropped symbols are reported, not swallowed: a scan that quietly covers
    # half the board is choosing a different universe than the one configured.
    logger.info(
        "scan symbols=%d evaluated=%d signals=%d skipped=%s elapsed=%.1fs",
        len(symbols), evaluated_count, len(signals), skipped, time.time() - started,
        extra={"event": "scan"},
    )
    if evaluated_count < len(symbols) * 0.9:
        logger.warning("scan_coverage_low evaluated=%d of %d skipped=%s",
                       evaluated_count, len(symbols), skipped, extra={"event": "scan"})
    return signals


def describe(symbol: str, intent) -> str:
    meta = intent.metadata if isinstance(intent.metadata, dict) else {}
    layer5 = (meta.get("layer_trace", {}).get("layers", {}).get("layer5_tp_sl", {}).get("details", {}))
    stop_margin = layer5.get("stop_pct_of_margin")
    target_margin = layer5.get("target_pct_of_margin")
    safe_lev = layer5.get("max_safe_leverage")

    parts = [f"{intent.action.value} {symbol}",
             f"entry~{layer5.get('entry', 0):.6g}",
             f"stop {layer5.get('sl', 0):.6g}",
             f"target {layer5.get('tp', 0):.6g}"]
    if stop_margin is not None:
        parts.append(f"stop {stop_margin:.0f}% / target {target_margin:.0f}% of margin")
    if safe_lev is not None:
        parts.append(f"max safe leverage {safe_lev:.0f}x")
    return " | ".join(parts)


def load_env() -> None:
    """Read .env if one is present.

    The credentials live in .env, but nothing in the V2 stack loads it, so a
    scanner started straight from the shell found no token and reported
    alerts=0 while looking like it had started correctly.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def build_alerters():
    """Telegram only when both credentials are present; silence otherwise.

    Observation must be able to run with no alerting configured at all, so a
    missing token is a quiet no-op rather than a startup failure.
    """
    token = os.getenv("TELEGRAM_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID", "")
    if not (token and chat_id):
        return []
    from trading.alerts.telegram import TelegramAlerter

    return [TelegramAlerter(token=token, chat_id=chat_id)]


def main() -> int:
    args = parse_args()
    load_env()
    logger = setup_logging("INFO")

    client = MexcContractClient()
    universe = SymbolUniverse(
        client,
        UniverseConfig(
            min_turnover_24h_usdt=args.min_turnover,
            max_turnover_24h_usdt=args.max_turnover,
            max_symbols=args.max_symbols,
            max_min_notional_usdt=args.max_min_notional,
            refresh_sec=args.universe_refresh_sec,
        ),
    )
    feed = MarketDataFeed(client=client)
    strategy = LayeredPumpStrategy()
    strategy.set_htf_cache(HigherTimeframeCache(feed))

    alerters = build_alerters()
    tracker = SignalObservationTracker(args.observations, horizon_minutes=args.observe_minutes)
    population_journal = PopulationJournal(
        args.population_journal,
        enabled=not args.disable_population_journal,
    )

    logger.info(
        "scanner_start venue=mexc execution=disabled timeframe=%s alerts=%d "
        "observations=%s population_journal=%s population_journal_enabled=%s",
        args.timeframe,
        len(alerters),
        args.observations,
        args.population_journal,
        population_journal.enabled,
        extra={"event": "startup"},
    )
    if not alerters:
        # Otherwise a scanner with no delivery path looks identical to a healthy
        # one until the first signal is found and silently goes nowhere.
        logger.warning(
            "no_alert_channel: set TELEGRAM_TOKEN and CHAT_ID (.env or environment); "
            "signals will only be written to the log",
            extra={"event": "startup"},
        )

    try:
        while True:
            signals = scan_once(universe=universe, feed=feed, strategy=strategy, logger=logger,
                                timeframe=args.timeframe, candles=args.candles,
                                workers=args.workers, tracker=tracker, alerters=alerters,
                                population_journal=population_journal)
            for symbol, intent, frame in signals:
                text = describe(symbol, intent)
                logger.info("%s", text, extra={"event": "signal"})
                delivered = False
                for alerter in alerters:
                    try:
                        channel_delivered = bool(alerter.send(text))
                        delivered = channel_delivered or delivered
                        if not channel_delivered:
                            logger.warning(
                                "alert_not_delivered=%s",
                                type(alerter).__name__,
                                extra={"event": "signal"},
                            )
                    except Exception as exc:
                        logger.warning(
                            "alert_failed=%s",
                            safe_error_code(exc),
                            extra={"event": "signal"},
                        )
                if tracker is not None and intent.action is IntentAction.SHORT_ENTRY:
                    meta = intent.metadata if isinstance(intent.metadata, dict) else {}
                    tracking = meta.get("population_tracking", {})
                    layer5 = (
                        meta.get("layer_trace", {})
                        .get("layers", {})
                        .get("layer5_tp_sl", {})
                        .get("details", {})
                    )
                    try:
                        tracker.record_short(
                            signal_id=str(tracking.get("snapshot_id") or ""),
                            symbol=symbol,
                            phase=intent.action.value,
                            entry=float(layer5.get("entry") or frame.iloc[-1]["close"]),
                            take_profit=float(layer5.get("tp") or intent.take_profit or 0.0),
                            stop_loss=float(layer5.get("sl") or intent.stop_loss or 0.0),
                            signal_ts=float(tracking.get("decision_ts") or time.time()),
                            signal_bar_ts=tracking.get("base_bar_open_ts"),
                            delivered=delivered,
                            candidate_source="mexc_scan",
                        )
                    except Exception as exc:
                        logger.warning(
                            "tracker_delivery_update_failed=%s err=%s",
                            symbol,
                            safe_error_code(exc),
                            extra={"event": "signal"},
                        )
            if not args.loop:
                break
            time.sleep(max(30, args.interval_sec))
    finally:
        feed.close()
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
