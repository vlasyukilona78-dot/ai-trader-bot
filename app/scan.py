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
import time
from concurrent.futures import ThreadPoolExecutor

from core.indicators import compute_indicators
from trading.market_data.feed import MarketDataFeed
from trading.market_data.mexc_client import MexcContractClient
from trading.market_data.timeframe_cache import HigherTimeframeCache
from trading.market_data.universe import SymbolUniverse, UniverseConfig
from trading.metrics.logging import setup_logging
from trading.signals.layered_strategy import LayeredPumpStrategy
from trading.signals.signal_types import IntentAction
from trading.signals.strategy_interface import StrategyContext
from trading.state.models import TradeState


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
    return p.parse_args()


def scan_once(*, universe, feed, strategy, logger, timeframe, candles, workers) -> list:
    snapshot = universe.refresh()
    symbols = snapshot.symbols
    if not symbols:
        logger.warning("empty_universe", extra={"event": "scan"})
        return []

    # Freeze the volatility distribution before evaluating anyone, so a
    # candidate's fate does not depend on its position in the scan order.
    if hasattr(strategy, "begin_sweep"):
        strategy.begin_sweep()

    try:
        btc = feed.fetch_frame(symbol="BTCUSDT", timeframe=timeframe, candles=candles).ohlcv
        strategy.set_benchmark(btc)
    except Exception as exc:
        strategy.set_benchmark(None)
        logger.warning("benchmark_unavailable=%s", exc, extra={"event": "scan"})

    skipped = {"no_data": 0, "short_history": 0, "error": 0}

    def evaluate(symbol: str):
        try:
            frame = feed.fetch_frame(symbol=symbol, timeframe=timeframe, candles=candles)
            if frame.ohlcv.empty:
                skipped["no_data"] += 1
                return None
            if len(frame.ohlcv) < 80:
                skipped["short_history"] += 1
                return None
            enriched = compute_indicators(frame.ohlcv)
            intent = strategy.generate(
                StrategyContext(
                    symbol=symbol,
                    market_ohlcv=enriched,
                    mark_price=frame.mark_price or float(enriched.iloc[-1]["close"]),
                    exchange=None,
                    # No position is held by the bot; every symbol is evaluated flat.
                    synced_state=TradeState.FLAT,
                    sentiment_index=50.0,
                    sentiment_source="fallback_neutral_50",
                )
            )
            return symbol, intent, enriched
        except Exception as exc:
            skipped["error"] += 1
            logger.debug("symbol_failed=%s err=%s", symbol, exc, extra={"event": "scan"})
            return None

    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        evaluated = [r for r in pool.map(evaluate, symbols) if r is not None]

    signals = [
        (sym, intent, frame)
        for sym, intent, frame in evaluated
        if intent.action in (IntentAction.SHORT_ENTRY, IntentAction.LONG_ENTRY)
    ]

    # Dropped symbols are reported, not swallowed: a scan that quietly covers
    # half the board is choosing a different universe than the one configured.
    logger.info(
        "scan symbols=%d evaluated=%d signals=%d skipped=%s elapsed=%.1fs",
        len(symbols), len(evaluated), len(signals), skipped, time.time() - started,
        extra={"event": "scan"},
    )
    if len(evaluated) < len(symbols) * 0.9:
        logger.warning("scan_coverage_low evaluated=%d of %d skipped=%s",
                       len(evaluated), len(symbols), skipped, extra={"event": "scan"})
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


def main() -> int:
    args = parse_args()
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

    logger.info("scanner_start venue=mexc execution=disabled timeframe=%s",
                args.timeframe, extra={"event": "startup"})

    try:
        while True:
            signals = scan_once(universe=universe, feed=feed, strategy=strategy, logger=logger,
                                timeframe=args.timeframe, candles=args.candles, workers=args.workers)
            for symbol, intent, _ in signals:
                logger.info("%s", describe(symbol, intent), extra={"event": "signal"})
            if not args.loop:
                break
            time.sleep(max(30, args.interval_sec))
    finally:
        feed.close()
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
