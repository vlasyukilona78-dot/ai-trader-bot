from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtesting.metrics import summarize_trades
from core.indicators import compute_indicators
from core.market_regime import detect_market_regime
from core.signal_generator import SignalConfig, SignalContext, SignalGenerator
from core.volume_profile import compute_volume_profile


@dataclass
class BacktestConfig:
    initial_equity: float = 1000.0
    risk_per_trade: float = 0.01
    fee_bps_per_side: float = 5.0
    slippage_bps_per_side: float = 2.0
    max_hold_bars: int = 120


def _discover_csv_candidates(limit: int = 8) -> list[str]:
    candidates: list[Path] = []
    for pattern in ("data/raw/*.csv", "data/processed/*.csv", "*.csv"):
        candidates.extend(sorted(Path.cwd().glob(pattern)))

    seen: set[str] = set()
    out: list[str] = []
    for item in candidates:
        norm = item.as_posix()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
        if len(out) >= limit:
            break
    return out


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        hints = _discover_csv_candidates()
        hints_text = f" Existing CSVs: {', '.join(hints)}" if hints else ""
        raise FileNotFoundError(
            f"OHLCV file not found: {csv_path}. Put file in data/raw or use fetch_ohlcv.py.{hints_text}"
        )

    df = pd.read_csv(csv_path)

    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        dt = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    elif "time" in df.columns:
        # support both seconds and ms
        raw = pd.to_numeric(df["time"], errors="coerce")
        unit = "ms" if raw.dropna().median() > 10_000_000_000 else "s"
        dt = pd.to_datetime(raw, unit=unit, utc=True, errors="coerce")
    else:
        raise ValueError("CSV must contain datetime/timestamp/time column")

    df["datetime"] = dt
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()

    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"Missing OHLCV column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    return df


def _simulate_trade_exit(df: pd.DataFrame, start_idx: int, side: str, tp: float, sl: float, max_hold: int):
    end = min(len(df) - 1, start_idx + max_hold)
    for j in range(start_idx + 1, end + 1):
        row = df.iloc[j]
        high = float(row["high"])
        low = float(row["low"])

        if side == "SHORT":
            hit_tp = low <= tp
            hit_sl = high >= sl
        else:
            hit_tp = high >= tp
            hit_sl = low <= sl

        if hit_tp and hit_sl:
            # conservative fill: assume SL first
            return j, sl, "ambiguous_sl"
        if hit_tp:
            return j, tp, "tp"
        if hit_sl:
            return j, sl, "sl"

    return end, float(df.iloc[end]["close"]), "timeout"


def run_backtest(df: pd.DataFrame, cfg: BacktestConfig, signal_cfg: SignalConfig | None = None) -> tuple[pd.DataFrame, dict]:
    signal_gen = SignalGenerator(signal_cfg or SignalConfig())

    enriched = compute_indicators(df)
    trades: list[dict] = []

    for i in range(80, len(enriched) - 2):
        hist = enriched.iloc[: i + 1]
        vp = compute_volume_profile(hist)
        regime = detect_market_regime(hist)

        sentiment = float(hist.iloc[-1].get("sentiment_index", 50.0)) if "sentiment_index" in hist.columns else 50.0
        funding = float(hist.iloc[-1].get("funding_rate", 0.0)) if "funding_rate" in hist.columns else 0.0
        ratio = float(hist.iloc[-1].get("long_short_ratio", 1.0)) if "long_short_ratio" in hist.columns else 1.0

        context = SignalContext(
            symbol="BACKTEST/USDT",
            df=hist,
            volume_profile=vp,
            regime=regime,
            sentiment_index=sentiment,
            funding_rate=funding,
            long_short_ratio=ratio,
        )
        signal = signal_gen.generate(context)
        if signal is None:
            continue

        entry = signal.entry
        exit_idx, exit_price, exit_reason = _simulate_trade_exit(
            enriched,
            start_idx=i,
            side=signal.side,
            tp=signal.tp,
            sl=signal.sl,
            max_hold=cfg.max_hold_bars,
        )

        # risk-based sizing
        risk_per_unit = abs(entry - signal.sl)
        if risk_per_unit <= 0:
            continue
        risk_usdt = cfg.initial_equity * cfg.risk_per_trade
        qty = risk_usdt / risk_per_unit

        # execution costs
        total_cost_pct = ((cfg.fee_bps_per_side + cfg.slippage_bps_per_side) * 2) / 10000.0

        if signal.side == "SHORT":
            gross = (entry - exit_price) * qty
        else:
            gross = (exit_price - entry) * qty

        cost = abs(entry * qty) * total_cost_pct
        pnl = gross - cost
        ret = pnl / max(cfg.initial_equity, 1e-9)

        trades.append(
            {
                "entry_time": hist.index[-1],
                "exit_time": enriched.index[exit_idx],
                "side": signal.side,
                "entry": entry,
                "exit": exit_price,
                "tp": signal.tp,
                "sl": signal.sl,
                "qty": qty,
                "pnl": pnl,
                "ret": ret,
                "confidence": signal.confidence,
                "reason": exit_reason,
            }
        )

    trades_df = pd.DataFrame(trades)
    stats = summarize_trades(trades_df, initial_equity=cfg.initial_equity)
    return trades_df, stats


class PaperTrader:
    """Forward-testing helper. Feed one bar at a time."""

    def __init__(self, signal_cfg: SignalConfig | None = None):
        self.signal_gen = SignalGenerator(signal_cfg or SignalConfig())
        self.history = pd.DataFrame()
        self.open_positions: list[dict] = []

    def on_new_bar(self, bar: dict, sentiment_index: float | None = 50.0, funding_rate: float | None = 0.0):
        row = pd.DataFrame([bar])
        if "datetime" in row.columns:
            row["datetime"] = pd.to_datetime(row["datetime"], utc=True)
            row = row.set_index("datetime")
        self.history = pd.concat([self.history, row], axis=0)
        if len(self.history) < 80:
            return None

        enriched = compute_indicators(self.history)
        vp = compute_volume_profile(enriched)
        regime = detect_market_regime(enriched)

        ctx = SignalContext(
            symbol=str(bar.get("symbol", "PAPER/USDT")),
            df=enriched,
            volume_profile=vp,
            regime=regime,
            sentiment_index=sentiment_index,
            funding_rate=funding_rate,
            long_short_ratio=1.0,
        )
        return self.signal_gen.generate(ctx)


def parse_args():
    parser = argparse.ArgumentParser(description="Run historical backtest for layered signal strategy")
    parser.add_argument("--data", required=True, help="Path to OHLCV CSV")
    parser.add_argument("--out", default="logs/backtest_trades.csv", help="Where to save trades")
    parser.add_argument("--equity", type=float, default=1000.0)
    parser.add_argument("--risk", type=float, default=0.01)
    parser.add_argument("--max-hold", type=int, default=120)
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = load_ohlcv_csv(args.data)
    trades_df, metrics = run_backtest(
        data,
        cfg=BacktestConfig(
            initial_equity=float(args.equity),
            risk_per_trade=float(args.risk),
            max_hold_bars=int(args.max_hold),
            fee_bps_per_side=float(args.fee_bps),
            slippage_bps_per_side=float(args.slippage_bps),
        ),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(out_path, index=False)

    print("Backtest complete")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"trades_csv: {out_path}")


