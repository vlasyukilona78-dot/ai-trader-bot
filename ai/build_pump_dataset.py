"""CLI: build the labelled pump dataset across the MEXC alt universe.

Example:
    python -m ai.build_pump_dataset --days 100 --symbols 120 --out data/processed/pump_dataset.csv
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from ai.pump_dataset import EventConfig, LabelConfig, build_symbol_rows
from trading.market_data.history import HistoryCollector, HistoryConfig
from trading.market_data.mexc_client import MexcContractClient
from trading.market_data.universe import SymbolUniverse, UniverseConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build labelled pump dataset from MEXC history")
    p.add_argument("--days", type=int, default=100, help="History depth in days")
    p.add_argument("--symbols", type=int, default=120, help="How many symbols from the universe")
    p.add_argument("--min-turnover", type=float, default=250_000.0)
    p.add_argument("--max-turnover", type=float, default=100_000_000.0)
    p.add_argument("--min-move", type=float, default=0.05, help="Minimum run-up to count as an event")
    p.add_argument("--lookback-hours", type=int, default=6)
    p.add_argument("--horizon-hours", type=int, default=48)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cache-dir", default="data/history")
    p.add_argument("--out", default="data/processed/pump_dataset.csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    client = MexcContractClient()
    universe = SymbolUniverse(
        client,
        UniverseConfig(
            min_turnover_24h_usdt=args.min_turnover,
            max_turnover_24h_usdt=args.max_turnover,
            max_symbols=args.symbols,
        ),
    )
    entries = universe.refresh().entries
    print(f"universe: {len(entries)} symbols (of {universe.snapshot.total_contracts} contracts)")

    now = int(time.time())
    start = now - args.days * 86400
    event_cfg = EventConfig(min_move_pct=args.min_move, lookback_hours=args.lookback_hours)
    label_cfg = LabelConfig(horizon_hours=args.horizon_hours)

    # BTC serves as the market benchmark, so a single-coin pump can be told apart
    # from the whole board moving together.
    benchmark = HistoryCollector(client, HistoryConfig(cache_dir=args.cache_dir)).fetch_range(
        "BTCUSDT", "Min60", start, now
    ).reset_index()
    print(f"benchmark BTC bars: {len(benchmark)}")

    def work(symbol: str):
        collector = HistoryCollector(MexcContractClient(), HistoryConfig(cache_dir=args.cache_dir))
        try:
            return build_symbol_rows(symbol, collector, start, now, event_cfg, label_cfg,
                                     benchmark_1h=benchmark)
        except Exception as exc:  # one bad symbol must not kill the run
            print(f"  {symbol}: failed ({exc})")
            return []

    rows: list[dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, e.symbol): e.symbol for e in entries}
        for done, fut in enumerate(as_completed(futures), 1):
            got = fut.result()
            rows.extend(got)
            if done % 10 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)} symbols, {len(rows)} events, {time.time()-t0:.0f}s")

    if not rows:
        print("no events collected")
        return 1

    df = pd.DataFrame(rows).sort_values(["ts", "symbol"]).reset_index(drop=True)
    out_path = args.out
    import os

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\nwrote {len(df)} events from {df.symbol.nunique()} symbols -> {out_path}")
    print(f"  MAE   median {df.mae_pct.median()*100:.2f}%  mean {df.mae_pct.mean()*100:.2f}%")
    print(f"  MFE   median {df.mfe_pct.median()*100:.2f}%")
    print(f"  averages needed: median {df.n_averages.median():.0f}  mean {df.n_averages.mean():.2f}")
    print(f"  resolved into profit: {df.dca_resolved.mean()*100:.1f}%")
    for col in [c for c in df.columns if c.startswith("good_mae_")]:
        print(f"  {col}: {df[col].mean()*100:.1f}% of events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
