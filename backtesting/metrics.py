from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60


def max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max.replace(0, np.nan)
    return float(drawdown.min()) if drawdown.notna().any() else 0.0


def infer_trade_periods_per_year(timestamps: Iterable | pd.Series | None) -> float:
    if timestamps is None:
        return 0.0
    series = pd.Series(list(timestamps) if not isinstance(timestamps, pd.Series) else timestamps)
    if series.empty:
        return 0.0
    ts = pd.to_datetime(series, utc=True, errors="coerce").dropna().sort_values().drop_duplicates()
    if len(ts) < 2:
        return 0.0
    diffs = ts.diff().dt.total_seconds().dropna()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return 0.0
    median_gap_sec = float(diffs.median())
    if median_gap_sec <= 0.0:
        return 0.0
    return float(SECONDS_PER_YEAR / median_gap_sec)


def build_equity_curve(trades: pd.DataFrame, initial_equity: float = 1000.0) -> pd.Series:
    if trades.empty:
        return pd.Series([float(initial_equity)], name="equity", dtype="float64")

    ordered = trades.copy()
    if "exit_time" in ordered.columns:
        ordered["exit_time"] = pd.to_datetime(ordered["exit_time"], utc=True, errors="coerce")
        ordered = ordered.sort_values("exit_time", kind="stable")
    ordered["pnl"] = pd.to_numeric(ordered.get("pnl", 0.0), errors="coerce").fillna(0.0)

    if "equity_after" in ordered.columns:
        equity = pd.to_numeric(ordered["equity_after"], errors="coerce").ffill().fillna(float(initial_equity))
    else:
        equity = float(initial_equity) + ordered["pnl"].cumsum()
    equity.name = "equity"
    return equity.astype("float64")


def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: float | None = None,
    *,
    timestamps: Iterable | pd.Series | None = None,
) -> float:
    if returns.empty:
        return 0.0
    annualization = float(periods_per_year or 0.0)
    if annualization <= 0.0:
        annualization = infer_trade_periods_per_year(timestamps)
    if annualization <= 0.0:
        annualization = float(max(len(returns), 1))
    std = float(returns.std(ddof=0))
    if std <= 0.0:
        return 0.0
    mean = float(returns.mean())
    return float((mean / std) * math.sqrt(annualization))


def profit_factor(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    if gross_loss <= 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def summarize_trades(trades: pd.DataFrame, initial_equity: float = 1000.0) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "winrate": 0.0,
            "sharpe": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "net_pnl": 0.0,
            "final_equity": initial_equity,
        }

    ordered = trades.copy()
    ordered["pnl"] = pd.to_numeric(ordered["pnl"], errors="coerce").fillna(0.0)
    ordered["ret"] = pd.to_numeric(ordered.get("ret", 0.0), errors="coerce").fillna(0.0)
    exit_times = None
    if "exit_time" in ordered.columns:
        ordered["exit_time"] = pd.to_datetime(ordered["exit_time"], utc=True, errors="coerce")
        ordered = ordered.sort_values("exit_time", kind="stable")
        exit_times = ordered["exit_time"]

    winrate = float((ordered["pnl"] > 0).mean())
    pf = profit_factor(ordered["pnl"])
    sharpe = sharpe_ratio(ordered["ret"], timestamps=exit_times)

    equity = build_equity_curve(ordered, initial_equity=initial_equity)
    mdd = max_drawdown(equity)

    return {
        "trades": int(len(ordered)),
        "winrate": winrate,
        "sharpe": sharpe,
        "profit_factor": pf,
        "max_drawdown": mdd,
        "net_pnl": float(ordered["pnl"].sum()),
        "final_equity": float(equity.iloc[-1]) if not equity.empty else initial_equity,
    }
