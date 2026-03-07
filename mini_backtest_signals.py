import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from config import LOG_PATH, TIMEFRAME

BYBIT_BASE_URL = "https://api.bybit.com"


@dataclass
class CostConfig:
    fee_bps: float = 5.0
    slippage_bps: float = 2.0
    spread_bps: float = 1.0
    funding_bps_per_8h: float = 1.0


@dataclass
class EvalResult:
    outcome: str
    hit_ts: int | None
    hit_price: float | None
    gross_return_pct: float | None
    net_return_pct: float | None
    cost_pct: float | None
    resolution: str


def to_ms(ts_value) -> int | None:
    try:
        dt = pd.to_datetime(ts_value, utc=True)
        if pd.isna(dt):
            return None
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def interval_to_minutes(interval: str) -> int:
    value = str(interval).strip().upper()
    if value.isdigit():
        return int(value)
    mapping = {"D": 24 * 60, "W": 7 * 24 * 60}
    return mapping.get(value, 1)


def fetch_forward_klines(symbol: str, interval: str, start_ms: int, limit: int) -> pd.DataFrame:
    normalized = symbol.replace("/", "").upper()
    params = {
        "category": "linear",
        "symbol": normalized,
        "interval": str(interval),
        "start": int(start_ms),
        "limit": int(limit),
    }
    response = requests.get(f"{BYBIT_BASE_URL}/v5/market/kline", params=params, timeout=12)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("result", {}).get("list", [])
    if not rows:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "turnover"])

    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume", "turnover"])
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time").reset_index(drop=True)
    return df


def _calc_cost_pct(cost: CostConfig, hold_minutes: float) -> float:
    round_trip_bps = (2.0 * cost.fee_bps) + (2.0 * cost.slippage_bps) + cost.spread_bps
    funding_bps = cost.funding_bps_per_8h * max(0.0, hold_minutes) / (8.0 * 60.0)
    return (round_trip_bps + funding_bps) / 10000.0


def _calc_return_pct(direction: str, entry: float, exit_price: float) -> float:
    if entry <= 0:
        return 0.0
    if direction == "SHORT":
        return (entry - exit_price) / entry
    return (exit_price - entry) / entry


def _hit_flags(direction: str, tp: float, sl: float, high: float, low: float) -> tuple[bool, bool]:
    if direction == "SHORT":
        return low <= tp, high >= sl
    return high >= tp, low <= sl


def resolve_ambiguous_with_lower_tf(
    symbol: str,
    coarse_ts: int,
    coarse_interval: str,
    resolve_interval: str,
    direction: str,
    tp: float,
    sl: float,
) -> tuple[str | None, int | None, float | None, str]:
    if str(resolve_interval) == str(coarse_interval):
        return None, None, None, "same_interval"

    coarse_minutes = interval_to_minutes(coarse_interval)
    lower_minutes = interval_to_minutes(resolve_interval)
    if lower_minutes >= coarse_minutes:
        return None, None, None, "not_lower_tf"

    lower_limit = max(5, int(coarse_minutes / max(lower_minutes, 1)) + 2)
    candles = fetch_forward_klines(symbol, resolve_interval, coarse_ts, lower_limit)
    if candles.empty:
        return None, None, None, "lower_tf_empty"

    for _, row in candles.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        ts = int(row["time"])
        hit_tp, hit_sl = _hit_flags(direction, tp, sl, high, low)
        if hit_tp and hit_sl:
            continue
        if hit_tp:
            return "win", ts, float(tp), "resolved_lower_tf"
        if hit_sl:
            return "loss", ts, float(sl), "resolved_lower_tf"

    return None, None, None, "unresolved_lower_tf"


def evaluate_path(
    symbol: str,
    signal_ts_ms: int,
    direction: str,
    entry: float,
    tp: float,
    sl: float,
    candles: pd.DataFrame,
    interval: str,
    resolve_ambiguous_interval: str,
    cost: CostConfig,
) -> EvalResult:
    for _, row in candles.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        ts = int(row["time"])

        hit_tp, hit_sl = _hit_flags(direction, tp, sl, high, low)
        outcome = None
        exit_price = None
        hit_ts = ts
        resolution = "direct"

        if hit_tp and hit_sl:
            resolved_outcome, resolved_ts, resolved_price, reason = resolve_ambiguous_with_lower_tf(
                symbol=symbol,
                coarse_ts=ts,
                coarse_interval=interval,
                resolve_interval=resolve_ambiguous_interval,
                direction=direction,
                tp=tp,
                sl=sl,
            )
            if resolved_outcome is not None:
                outcome = resolved_outcome
                hit_ts = resolved_ts
                exit_price = resolved_price
                resolution = reason
            else:
                # Conservative fallback for ambiguous bars: count as loss.
                outcome = "loss"
                exit_price = sl if direction == "LONG" else sl
                resolution = f"ambiguous->{reason}->loss"
        elif hit_tp:
            outcome = "win"
            exit_price = tp
        elif hit_sl:
            outcome = "loss"
            exit_price = sl

        if outcome is not None and exit_price is not None:
            hold_minutes = max(0.0, (hit_ts - signal_ts_ms) / 60000.0)
            gross = _calc_return_pct(direction, entry, exit_price)
            cost_pct = _calc_cost_pct(cost, hold_minutes)
            net = gross - cost_pct
            return EvalResult(outcome, hit_ts, exit_price, gross, net, cost_pct, resolution)

    return EvalResult("no_hit", None, None, None, None, None, "none")


def summarize(results_df: pd.DataFrame) -> dict:
    total = len(results_df)
    if total == 0:
        return {
            "total_signals": 0,
            "evaluated": 0,
            "wins": 0,
            "losses": 0,
            "no_hit": 0,
            "win_rate": 0.0,
            "avg_net_return_pct": 0.0,
            "total_net_return_pct": 0.0,
            "signals_per_day": 0.0,
        }

    evaluated = int((results_df["outcome"].isin(["win", "loss"])).sum())
    wins = int((results_df["outcome"] == "win").sum())
    losses = int((results_df["outcome"] == "loss").sum())
    no_hit = int((results_df["outcome"] == "no_hit").sum())
    win_rate = float(wins / (wins + losses)) if (wins + losses) > 0 else 0.0

    net_series = pd.to_numeric(results_df.get("net_return_pct"), errors="coerce")
    avg_net = float(net_series.mean()) if net_series.notna().any() else 0.0
    total_net = float(net_series.fillna(0.0).sum())

    ts_series = pd.to_datetime(results_df["signal_time"], errors="coerce", utc=True).dropna()
    if ts_series.empty:
        signals_per_day = 0.0
    else:
        span_days = max((ts_series.max() - ts_series.min()).total_seconds() / 86400.0, 1.0)
        signals_per_day = float(total / span_days)

    return {
        "total_signals": total,
        "evaluated": evaluated,
        "wins": wins,
        "losses": losses,
        "no_hit": no_hit,
        "win_rate": win_rate,
        "avg_net_return_pct": avg_net,
        "total_net_return_pct": total_net,
        "signals_per_day": signals_per_day,
    }


def run_backtest(
    log_path: Path,
    strategy: str,
    interval: str,
    limit: int,
    forward_candles: int,
    resolve_ambiguous_interval: str,
    cost: CostConfig,
    output: Path,
):
    if not log_path.exists():
        raise FileNotFoundError(f"Signal log not found: {log_path}")

    df = pd.read_csv(log_path)
    required = {"time", "symbol", "direction", "entry", "tp", "sl"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {log_path}: {sorted(missing)}")

    if strategy:
        if "strategy" not in df.columns:
            raise ValueError("Strategy filter requested, but 'strategy' column is missing in log")
        df = df[df["strategy"] == strategy]

    if df.empty:
        raise ValueError("No rows left after filtering")

    if limit > 0:
        df = df.tail(limit)

    rows = []
    for _, row in df.iterrows():
        signal_ts_ms = to_ms(row["time"])
        if signal_ts_ms is None:
            rows.append(
                {
                    "signal_time": row["time"],
                    "symbol": row["symbol"],
                    "direction": row["direction"],
                    "entry": row["entry"],
                    "tp": row["tp"],
                    "sl": row["sl"],
                    "outcome": "invalid_time",
                    "hit_time": None,
                    "hit_price": None,
                    "gross_return_pct": None,
                    "net_return_pct": None,
                    "cost_pct": None,
                    "resolution": "none",
                }
            )
            continue

        try:
            direction = str(row["direction"]).upper().strip()
            candles = fetch_forward_klines(str(row["symbol"]), interval, signal_ts_ms, forward_candles)
            result = evaluate_path(
                symbol=str(row["symbol"]),
                signal_ts_ms=signal_ts_ms,
                direction=direction,
                entry=float(row["entry"]),
                tp=float(row["tp"]),
                sl=float(row["sl"]),
                candles=candles,
                interval=interval,
                resolve_ambiguous_interval=resolve_ambiguous_interval,
                cost=cost,
            )
            hit_time = (
                datetime.fromtimestamp(result.hit_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                if result.hit_ts
                else None
            )
            rows.append(
                {
                    "signal_time": row["time"],
                    "symbol": row["symbol"],
                    "direction": direction,
                    "entry": float(row["entry"]),
                    "tp": float(row["tp"]),
                    "sl": float(row["sl"]),
                    "outcome": result.outcome,
                    "hit_time": hit_time,
                    "hit_price": result.hit_price,
                    "gross_return_pct": result.gross_return_pct,
                    "net_return_pct": result.net_return_pct,
                    "cost_pct": result.cost_pct,
                    "resolution": result.resolution,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "signal_time": row["time"],
                    "symbol": row["symbol"],
                    "direction": row["direction"],
                    "entry": float(row["entry"]),
                    "tp": float(row["tp"]),
                    "sl": float(row["sl"]),
                    "outcome": f"error:{type(exc).__name__}",
                    "hit_time": None,
                    "hit_price": None,
                    "gross_return_pct": None,
                    "net_return_pct": None,
                    "cost_pct": None,
                    "resolution": "error",
                }
            )

    results_df = pd.DataFrame(rows)
    stats = summarize(results_df)

    output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output, index=False)

    print("Mini backtest complete")
    print(f"Signals total: {stats['total_signals']}")
    print(f"Evaluated (win/loss): {stats['evaluated']}")
    print(f"Wins: {stats['wins']}")
    print(f"Losses: {stats['losses']}")
    print(f"No hit: {stats['no_hit']}")
    print(f"Win rate: {stats['win_rate'] * 100:.2f}%")
    print(f"Avg net return: {stats['avg_net_return_pct'] * 100:.4f}%")
    print(f"Total net return: {stats['total_net_return_pct'] * 100:.4f}%")
    print(f"Signals per day: {stats['signals_per_day']:.2f}")
    print(f"Detailed report: {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Mini backtest for signal log using Bybit forward candles")
    parser.add_argument("--log", default=LOG_PATH, help="Path to signal log CSV")
    parser.add_argument("--strategy", default="pump_short_profile", help="Filter by strategy name")
    parser.add_argument("--interval", default=TIMEFRAME, help="Bybit kline interval for forward simulation")
    parser.add_argument("--limit", type=int, default=200, help="Use only last N signals, 0 for all")
    parser.add_argument("--forward-candles", type=int, default=120, help="How many candles to scan forward per signal")
    parser.add_argument("--resolve-ambiguous-interval", default="1", help="Lower timeframe for ambiguous bar resolution")

    parser.add_argument("--fee-bps", type=float, default=5.0, help="Round-trip exchange fee side (per side, bps)")
    parser.add_argument("--slippage-bps", type=float, default=2.0, help="Expected slippage per side (bps)")
    parser.add_argument("--spread-bps", type=float, default=1.0, help="Bid/ask spread cost (bps, round-trip)")
    parser.add_argument("--funding-bps-per-8h", type=float, default=1.0, help="Funding impact in bps per 8h")

    parser.add_argument("--output", default="logs/mini_backtest_results.csv", help="Path to output CSV")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cost_cfg = CostConfig(
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slippage_bps),
        spread_bps=float(args.spread_bps),
        funding_bps_per_8h=float(args.funding_bps_per_8h),
    )
    run_backtest(
        log_path=Path(args.log),
        strategy=args.strategy,
        interval=str(args.interval),
        limit=int(args.limit),
        forward_candles=int(args.forward_candles),
        resolve_ambiguous_interval=str(args.resolve_ambiguous_interval),
        cost=cost_cfg,
        output=Path(args.output),
    )
