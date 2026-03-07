#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from core.market_data import MarketDataClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OHLCV from Bybit and save CSV")
    parser.add_argument("--symbol", default="BTC/USDT", help="Symbol, e.g. BTC/USDT")
    parser.add_argument("--interval", default="1", help="Bybit interval, e.g. 1, 5, 15, 60, D")
    parser.add_argument("--limit", type=int, default=1000, help="Candle count (max depends on exchange)")
    parser.add_argument("--output", default="data/raw/BTCUSDT_1m.csv", help="Output CSV path")
    parser.add_argument("--base-url", default="https://api.bybit.com", help="Bybit API base URL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = MarketDataClient(base_url=args.base_url)
    try:
        df = client.fetch_ohlcv(symbol=args.symbol, interval=str(args.interval), limit=int(args.limit))
    finally:
        client.close()

    if df.empty:
        raise SystemExit("No OHLCV rows downloaded. Check internet/API availability and symbol.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_df = df.reset_index()
    preferred = ["datetime", "time", "open", "high", "low", "close", "volume"]
    cols = [c for c in preferred if c in out_df.columns]
    out_df = out_df[cols]
    out_df.to_csv(out_path, index=False)

    print(f"saved={out_path}")
    print(f"rows={len(out_df)}")
    print(f"from={out_df['datetime'].iloc[0]} to={out_df['datetime'].iloc[-1]}")


if __name__ == "__main__":
    main()
