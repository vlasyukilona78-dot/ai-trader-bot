from __future__ import annotations

import math

import numpy as np
import pandas as pd


def max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max.replace(0, np.nan)
    return float(drawdown.min()) if drawdown.notna().any() else 0.0


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 365 * 24) -> float:
    if returns.empty:
        return 0.0
    std = float(returns.std(ddof=0))
    if std <= 0:
        return 0.0
    mean = float(returns.mean())
    return float((mean / std) * math.sqrt(periods_per_year))


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

    trades = trades.copy()
    trades["pnl"] = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0)
    trades["ret"] = pd.to_numeric(trades.get("ret", 0.0), errors="coerce").fillna(0.0)

    winrate = float((trades["pnl"] > 0).mean())
    pf = profit_factor(trades["pnl"])
    sharpe = sharpe_ratio(trades["ret"])

    equity = initial_equity + trades["pnl"].cumsum()
    mdd = max_drawdown(equity)

    return {
        "trades": int(len(trades)),
        "winrate": winrate,
        "sharpe": sharpe,
        "profit_factor": pf,
        "max_drawdown": mdd,
        "net_pnl": float(trades["pnl"].sum()),
        "final_equity": float(equity.iloc[-1]) if not equity.empty else initial_equity,
    }
