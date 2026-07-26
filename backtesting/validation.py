"""Validation that survives contact with a fat-tailed, high-win-rate strategy.

A single train/test split is not enough here. The averaging scheme resolves
98-99% of trades by construction, so a favourable sample shows zero losses
purely by chance - at a 1.3% loss rate, 90 trades come out clean about a third
of the time. Concluding "edge" from that is the mistake this module exists to
prevent.

Three things it adds:

- walk-forward folds, so a result has to repeat across periods rather than in
  one lucky window;
- bootstrap intervals clustered by symbol, because forty events on one coin are
  not forty independent observations;
- a portfolio pass that respects capital and concurrency, since signals overlap
  in time and a backtest that takes every one of them is describing an account
  nobody has.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def walk_forward_folds(
    n_rows: int,
    n_folds: int = 4,
    min_train_frac: float = 0.4,
    *,
    decision_ts: np.ndarray | None = None,
    label_horizon_sec: int = 48 * 3600,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds over chronologically ordered rows.

    Each fold trains on everything before its test block, which is how the
    strategy would actually have been calibrated in real time.

    Splitting on row index alone is not enough when a label looks forward. An
    event decided shortly before the boundary is scored on 48 hours of price
    action that lie inside the test block, so it already knows part of what the
    test is supposed to reveal. Supplying `decision_ts` purges those events from
    training; without it the split is positional and that leakage stays.
    """
    if n_rows < 10 or n_folds < 1:
        return []

    start = int(n_rows * min_train_frac)
    if start >= n_rows - 1:
        return []

    ts = np.asarray(decision_ts, dtype=float) if decision_ts is not None else None
    edges = np.linspace(start, n_rows, n_folds + 1).astype(int)
    folds = []
    for i in range(n_folds):
        train_end, test_end = edges[i], edges[i + 1]
        if test_end <= train_end:
            continue

        train = np.arange(0, train_end)
        test = np.arange(train_end, test_end)
        if ts is not None and len(test):
            # drop training events whose label window reaches into the test block
            boundary = ts[test].min()
            train = train[ts[train] + label_horizon_sec <= boundary]
        if len(train) == 0:
            continue
        folds.append((train, test))
    return folds


def clustered_bootstrap_ci(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile interval for the mean, resampling whole symbols.

    Resampling individual events would treat one coin's forty pumps as forty
    independent draws and report an interval several times too narrow.
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    if len(values) == 0:
        return (float("nan"), float("nan"))

    unique = np.unique(groups)
    if len(unique) < 2:
        return (float("nan"), float("nan"))

    by_group = [values[groups == g] for g in unique]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.integers(0, len(by_group), size=len(by_group))
        sample = np.concatenate([by_group[i] for i in pick])
        means[b] = sample.mean()

    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


@dataclass
class PortfolioConfig:
    capital: float = 1000.0
    leg_notional: float = 20.0
    max_concurrent: int = 5
    max_total_notional: float | None = None


@dataclass
class PortfolioResult:
    taken: int
    skipped_capacity: int
    final_equity: float
    total_return: float
    max_drawdown: float
    peak_concurrent: int
    peak_notional: float


def simulate_portfolio(
    trades: pd.DataFrame,
    cfg: PortfolioConfig | None = None,
) -> PortfolioResult:
    """Sequential portfolio pass over dated trades.

    `trades` needs `entry_ts`, `exit_ts`, `pnl_on_initial` and `max_deployed`
    (in units of one leg). Signals that arrive while the book is full are
    skipped rather than silently taken, which is what a real account does.
    """
    cfg = cfg or PortfolioConfig()
    if trades.empty:
        return PortfolioResult(0, 0, cfg.capital, 0.0, 0.0, 0, 0.0)

    ordered = trades.sort_values("entry_ts").reset_index(drop=True)
    equity = cfg.capital
    peak_equity = cfg.capital
    max_dd = 0.0
    open_positions: list[tuple[float, float, float]] = []  # (exit_ts, pnl_quote, notional)
    taken = skipped = 0
    peak_concurrent = 0
    peak_notional = 0.0

    for _, row in ordered.iterrows():
        entry_ts = float(row["entry_ts"])

        # settle anything that closed before this signal
        still_open = []
        for exit_ts, pnl_quote, notional in open_positions:
            if exit_ts <= entry_ts:
                equity += pnl_quote
                peak_equity = max(peak_equity, equity)
                max_dd = max(max_dd, (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0)
            else:
                still_open.append((exit_ts, pnl_quote, notional))
        open_positions = still_open

        committed = sum(n for _, _, n in open_positions)
        needed = float(row["max_deployed"]) * cfg.leg_notional
        cap = cfg.max_total_notional if cfg.max_total_notional is not None else cfg.capital

        if len(open_positions) >= cfg.max_concurrent or committed + needed > cap:
            skipped += 1
            continue

        taken += 1
        pnl_quote = float(row["pnl_on_initial"]) * cfg.leg_notional
        open_positions.append((float(row["exit_ts"]), pnl_quote, needed))
        peak_concurrent = max(peak_concurrent, len(open_positions))
        peak_notional = max(peak_notional, committed + needed)

    for _, pnl_quote, _ in open_positions:
        equity += pnl_quote
        peak_equity = max(peak_equity, equity)
        max_dd = max(max_dd, (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0)

    return PortfolioResult(
        taken=taken,
        skipped_capacity=skipped,
        final_equity=equity,
        total_return=(equity - cfg.capital) / cfg.capital,
        max_drawdown=max_dd,
        peak_concurrent=peak_concurrent,
        peak_notional=peak_notional,
    )
