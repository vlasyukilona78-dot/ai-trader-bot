"""Build the runtime research dataset across symbols, one process per core.

Profiling the single-process build showed where the time goes: about 47% inside
`_layer1_pump_window`, which rebuilds pandas Series over the whole pump window on
every bar, then the forward-frame slice, the volume profile and the regime read.
All of those live in behaviour-locked strategy code whose exact numbers are
pinned by golden vectors, so making them faster means editing the most
safety-critical path in the project for a small constant factor.

Symbols are independent, so the honest lever is the other one: run them in
parallel processes. That sidesteps the GIL entirely and scales with cores without
touching a single line of strategy logic.

Two properties matter as much as the speed:

- output order follows the input symbol list, never worker completion order, so
  the dataset is identical whatever the scheduler does;
- each symbol writes its own shard, so a crash costs that symbol rather than the
  whole multi-hour run.

This is offline research tooling. It reads cached history, never the exchange.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

import pandas as pd

from ai.pump_dataset import LabelConfig, PumpEvent, forward_window_quality, label_event
from ai.runtime_dataset import calibration_config, replay_runtime_signals

DEFAULT_CACHE = Path("data/history")
DEFAULT_OUTPUT = Path("data/processed/runtime_dataset")


@dataclass(frozen=True)
class SymbolResult:
    symbol: str
    rows: int
    events: int
    error: str | None = None


def _read(cache: Path, symbol: str, interval: str) -> pd.DataFrame:
    path = cache / f"{symbol}_{interval}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def complete_symbols(cache: Path) -> list[str]:
    """Symbols with every timeframe the replay and its labels need."""

    found: list[str] = []
    for path in sorted(cache.glob("*_Min60.csv")):
        symbol = path.name.replace("_Min60.csv", "")
        if (cache / f"{symbol}_Hour4.csv").exists() and (cache / f"{symbol}_Min5.csv").exists():
            found.append(symbol)
    return found


def build_symbol_rows(
    symbol: str,
    *,
    cache: Path,
    label_cfg: LabelConfig,
    benchmark: pd.DataFrame | None,
) -> list[dict]:
    """Replay one symbol and label whatever the strategy would have emitted."""

    hourly = _read(cache, symbol, "Min60")
    if hourly.empty or len(hourly) < 200:
        return []
    higher = _read(cache, symbol, "Hour4")
    minute5 = _read(cache, symbol, "Min5")
    if higher.empty or minute5.empty:
        return []

    end_ts = int(minute5["time"].max())
    horizon = label_cfg.horizon_hours * 3600
    forward_time = pd.to_numeric(minute5["time"], errors="coerce")

    rows: list[dict] = []
    for event in replay_runtime_signals(symbol, hourly, higher, benchmark, calibration_config()):
        if event.decision_ts + horizon > end_ts:
            continue  # the forward window would run past the cached data
        window = minute5[
            (forward_time >= event.decision_ts)
            & (forward_time < event.decision_ts + horizon)
        ]
        if len(window) < 10:
            continue
        quality = forward_window_quality(window, event.decision_ts, horizon, 300)
        if (
            quality["coverage"] < label_cfg.min_forward_coverage
            or quality["max_gap_bars"] > label_cfg.max_forward_gap_bars
        ):
            continue
        labels = label_event(
            PumpEvent(symbol=symbol, ts=event.ts, entry=event.entry, move_pct=0.0, run_up_bars=0),
            window,
            label_cfg,
            decision_ts=event.decision_ts,
        )
        if not labels:
            continue
        rows.append(
            {
                "symbol": symbol,
                "ts": event.ts,
                "decision_ts": event.decision_ts,
                "entry": event.entry,
                "side": event.side,
                "stop": event.stop,
                "target": event.target,
                "fwd_coverage": quality["coverage"],
                "fwd_max_gap_bars": quality["max_gap_bars"],
                **event.diagnostics,
                **labels,
            }
        )
    return rows


def _shard_path(output: Path, symbol: str) -> Path:
    return output / f"{symbol}.jsonl"


def _worker(payload: tuple[str, str, str, int]) -> SymbolResult:
    """Run in a spawned process: rebuild everything from plain arguments.

    Nothing but strings and ints crosses the process boundary, so a pickling
    difference cannot silently change the configuration a worker used.
    """

    symbol, cache_text, output_text, horizon_hours = payload
    cache = Path(cache_text)
    output = Path(output_text)
    try:
        benchmark = _read(cache, "BTCUSDT", "Min60")
        rows = build_symbol_rows(
            symbol,
            cache=cache,
            label_cfg=LabelConfig(horizon_hours=horizon_hours),
            benchmark=benchmark if not benchmark.empty else None,
        )
    except Exception as exc:  # one bad symbol must not end a multi-hour run
        return SymbolResult(symbol=symbol, rows=0, events=0, error=type(exc).__name__)

    shard = _shard_path(output, symbol)
    tmp = shard.with_suffix(".jsonl.partial")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    # Rename last: a shard file exists only once it is complete, so resuming
    # cannot mistake a half-written symbol for a finished one.
    tmp.replace(shard)
    return SymbolResult(symbol=symbol, rows=len(rows), events=len(rows))


def build_dataset(
    symbols: list[str],
    *,
    cache: Path = DEFAULT_CACHE,
    output: Path = DEFAULT_OUTPUT,
    workers: int | None = None,
    horizon_hours: int = 48,
    resume: bool = True,
    progress=None,
) -> list[SymbolResult]:
    """Build every symbol, then concatenate the shards in input order."""

    output.mkdir(parents=True, exist_ok=True)
    pending = [
        symbol
        for symbol in symbols
        if not (resume and _shard_path(output, symbol).exists())
    ]
    results: dict[str, SymbolResult] = {}
    for symbol in symbols:
        if symbol not in pending:
            existing = _shard_path(output, symbol)
            count = sum(1 for line in existing.open(encoding="utf-8") if line.strip())
            results[symbol] = SymbolResult(symbol=symbol, rows=count, events=count)

    if pending:
        chosen = workers if workers is not None else max(1, (os.cpu_count() or 2) - 2)
        payloads = [(symbol, str(cache), str(output), horizon_hours) for symbol in pending]

        if chosen <= 1:
            # One worker means no pool at all. Spawning a process to run a single
            # job buys nothing and hides tracebacks behind a pickling boundary,
            # which is exactly what you do not want while debugging a symbol.
            for payload in payloads:
                result = _worker(payload)
                results[result.symbol] = result
                if progress is not None:
                    progress(result, len(results), len(symbols))
        else:
            with ProcessPoolExecutor(max_workers=chosen) as pool:
                futures = {pool.submit(_worker, payload): payload[0] for payload in payloads}
                for done in as_completed(futures):
                    result = done.result()
                    results[result.symbol] = result
                    if progress is not None:
                        progress(result, len(results), len(symbols))

    # Input order, never completion order.
    return [results[symbol] for symbol in symbols if symbol in results]


def concatenate(symbols: list[str], *, output: Path, destination: Path) -> int:
    """Join the shards into one CSV, preserving the requested symbol order."""

    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        shard = _shard_path(output, symbol)
        if not shard.exists():
            continue
        rows = [json.loads(line) for line in shard.open(encoding="utf-8") if line.strip()]
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return 0
    combined = pd.concat(frames, ignore_index=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination, index=False)
    return len(combined)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel runtime research dataset builder")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--shards", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--out", type=Path, default=Path("data/processed/runtime_dataset.csv"))
    parser.add_argument("--workers", type=int, default=0, help="0 = cores minus two")
    parser.add_argument("--limit", type=int, default=0, help="0 = every complete symbol")
    parser.add_argument("--horizon-hours", type=int, default=48)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbols = complete_symbols(args.cache)
    if args.limit:
        symbols = symbols[: args.limit]
    if not symbols:
        print("no symbol has the full Min60/Hour4/Min5 history this build needs")
        return 1

    started = time.perf_counter()

    def report(result: SymbolResult, done: int, total: int) -> None:
        elapsed = time.perf_counter() - started
        rate = done / elapsed if elapsed else 0.0
        remaining = (total - done) / rate if rate else 0.0
        note = f" error={result.error}" if result.error else ""
        print(
            f"[{done}/{total}] {result.symbol}: {result.rows} rows{note} "
            f"| {elapsed / 60:.1f}m elapsed, ~{remaining / 60:.1f}m left",
            flush=True,
        )

    results = build_dataset(
        symbols,
        cache=args.cache,
        output=args.shards,
        workers=args.workers or None,
        horizon_hours=args.horizon_hours,
        resume=not args.no_resume,
        progress=report,
    )

    failed = [r for r in results if r.error]
    total_rows = concatenate(symbols, output=args.shards, destination=args.out)
    print(
        f"\n{total_rows} rows from {len(results) - len(failed)} symbols "
        f"in {(time.perf_counter() - started) / 60:.1f} minutes -> {args.out}"
    )
    if failed:
        print(f"{len(failed)} symbols failed: " + ", ".join(f"{r.symbol}({r.error})" for r in failed[:10]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
