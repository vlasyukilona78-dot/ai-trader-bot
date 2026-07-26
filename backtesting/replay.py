"""Pathwise P&L replay for the short-the-fade strategy.

This replaces the ad-hoc score used earlier in this branch, which was not P&L and
could not establish profitability. The problems it had, and how they are handled
here:

- It credited a winner with `target * legs` regardless of the drawdown survived,
  and charged a loser the worst floating loss, which is not a price anyone can
  exit at. Here every event is replayed bar by bar and exits at a price that was
  actually available.
- It scored a reward at one target while the labels had been built at another.
  Here the target is applied during the replay, so the two cannot disagree.
- It blended legs arithmetically. That is only correct for equal-quantity legs;
  with equal-notional legs the blend is notional-weighted, which is what
  `deployed / quantity` gives.
- It normalised drawdown by the moving average entry, understating it. Here
  drawdown is expressed against the initial leg, so a number is comparable
  across events.
- It ignored fees, spread and how much capital a trade actually ties up.

Conventions, chosen to be pessimistic where the bar data is ambiguous: within a
bar the adverse extreme is assumed to happen first, so averaging legs fill before
any exit, and a stop wins a same-bar tie with the target.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# Relative tolerance for price-level comparisons, so a bar that touches a level
# exactly is not missed to floating-point representation.
_LEVEL_EPS = 1e-9


@dataclass
class ExecutionCosts:
    """Defaults are the values measured live on MEXC contracts, not guesses."""

    taker_fee: float = 0.0002
    maker_fee: float = 0.0
    half_spread: float = 0.000145
    slippage: float = 0.00014


@dataclass
class ReplayConfig:
    target_pct: float = 0.03
    dca_step_pct: float = 0.08
    max_adds: int = 6
    equal_notional_legs: bool = True
    # Risk stop expressed against deployed capital: the trade is abandoned once
    # the floating loss reaches this fraction of what has been committed.
    max_loss_on_deployed: float | None = 1.0
    # Optional price stop above the blended entry.
    stop_pct_from_blended: float | None = None
    entry_is_taker: bool = True
    exit_is_taker: bool = True
    costs: ExecutionCosts = field(default_factory=ExecutionCosts)


@dataclass
class ReplayResult:
    exit_reason: str
    pnl_on_initial: float
    pnl_on_deployed: float
    legs: int
    max_deployed: float
    worst_drawdown_on_initial: float
    bars_held: int
    fees_paid_on_initial: float
    exit_price: float
    blended_entry: float

    @property
    def resolved(self) -> bool:
        return self.exit_reason == "target"


def _entry_fill(price: float, costs: ExecutionCosts) -> float:
    """A short sells into the bid and gives up spread and slippage."""
    return price * (1.0 - costs.half_spread - costs.slippage)


def _exit_fill(price: float, costs: ExecutionCosts) -> float:
    """Closing a short lifts the ask, so the buy-back is worse than mid."""
    return price * (1.0 + costs.half_spread + costs.slippage)


def replay_short(
    bars: pd.DataFrame,
    entry_price: float,
    cfg: ReplayConfig | None = None,
) -> ReplayResult | None:
    """Replay one short from `entry_price` over `bars` (needs high/low/close).

    Returns P&L per unit of the initial leg's notional, so results are comparable
    across events regardless of absolute position size.
    """
    cfg = cfg or ReplayConfig()
    if bars is None or bars.empty or entry_price <= 0:
        return None

    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(high).any():
        return None

    unit_notional = 1.0  # everything is expressed per initial leg
    entry_fee_rate = cfg.costs.taker_fee if cfg.entry_is_taker else cfg.costs.maker_fee
    exit_fee_rate = cfg.costs.taker_fee if cfg.exit_is_taker else cfg.costs.maker_fee

    fill = _entry_fill(entry_price, cfg.costs)
    qty = unit_notional / fill
    deployed = unit_notional
    fees = unit_notional * entry_fee_rate
    adds = 0
    worst_dd = 0.0

    def blended() -> float:
        return deployed / qty

    for i in range(len(high)):
        h, l, c = high[i], low[i], close[i]
        if not (np.isfinite(h) and np.isfinite(l)):
            continue

        # 1) adverse extreme first: averaging legs fill on the way up.
        # The tolerance matters: entry * (1 + 0.10) lands a hair above a bar that
        # touched the level exactly, and without it a real fill is silently missed.
        while adds < cfg.max_adds and h >= entry_price * (1.0 + cfg.dca_step_pct * (adds + 1)) * (1.0 - _LEVEL_EPS):
            adds += 1
            add_price = entry_price * (1.0 + cfg.dca_step_pct * adds)
            add_fill = _entry_fill(add_price, cfg.costs)
            add_notional = unit_notional if cfg.equal_notional_legs else add_fill * (unit_notional / fill)
            qty += add_notional / add_fill
            deployed += add_notional
            fees += add_notional * entry_fee_rate

        avg = blended()
        # floating loss at the bar's worst point, measured against the first leg
        floating = (h - avg) * qty / unit_notional
        worst_dd = max(worst_dd, floating)

        # 2) risk stop, before any favourable exit on the same bar
        stop_price = None
        if cfg.stop_pct_from_blended is not None:
            stop_price = avg * (1.0 + cfg.stop_pct_from_blended)
        if cfg.max_loss_on_deployed is not None:
            loss_stop = avg + cfg.max_loss_on_deployed * deployed / qty
            stop_price = loss_stop if stop_price is None else min(stop_price, loss_stop)

        if stop_price is not None and h >= stop_price:
            px = _exit_fill(stop_price, cfg.costs)
            fees += px * qty * exit_fee_rate
            pnl = (avg - px) * qty - fees
            return ReplayResult("stop", pnl / unit_notional, pnl / deployed, adds + 1,
                                deployed, worst_dd, i + 1, fees / unit_notional, px, avg)

        # 3) target
        target_price = avg * (1.0 - cfg.target_pct)
        if l <= target_price:
            px = _exit_fill(target_price, cfg.costs)
            fees += px * qty * exit_fee_rate
            pnl = (avg - px) * qty - fees
            return ReplayResult("target", pnl / unit_notional, pnl / deployed, adds + 1,
                                deployed, worst_dd, i + 1, fees / unit_notional, px, avg)

    # horizon reached: mark out at the last close
    avg = blended()
    px = _exit_fill(float(close[-1]), cfg.costs)
    fees += px * qty * exit_fee_rate
    pnl = (avg - px) * qty - fees
    return ReplayResult("horizon", pnl / unit_notional, pnl / deployed, adds + 1,
                        deployed, worst_dd, len(high), fees / unit_notional, px, avg)


def summarise(results: list[ReplayResult]) -> dict[str, float]:
    """Aggregate replay results into figures that mean what they say."""
    if not results:
        return {}
    pnl_initial = np.array([r.pnl_on_initial for r in results])
    pnl_deployed = np.array([r.pnl_on_deployed for r in results])
    deployed = np.array([r.max_deployed for r in results])
    wins = pnl_initial > 0

    gains = pnl_initial[pnl_initial > 0].sum()
    losses = -pnl_initial[pnl_initial < 0].sum()

    return {
        "trades": float(len(results)),
        "win_rate": float(wins.mean()),
        "target_rate": float(np.mean([r.exit_reason == "target" for r in results])),
        "stop_rate": float(np.mean([r.exit_reason == "stop" for r in results])),
        "horizon_rate": float(np.mean([r.exit_reason == "horizon" for r in results])),
        "mean_pnl_on_initial": float(pnl_initial.mean()),
        "median_pnl_on_initial": float(np.median(pnl_initial)),
        "mean_pnl_on_deployed": float(pnl_deployed.mean()),
        "total_pnl_on_initial": float(pnl_initial.sum()),
        "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
        "mean_deployed": float(deployed.mean()),
        "max_deployed": float(deployed.max()),
        "mean_legs": float(np.mean([r.legs for r in results])),
        "worst_drawdown_on_initial": float(max(r.worst_drawdown_on_initial for r in results)),
        "mean_fees_on_initial": float(np.mean([r.fees_paid_on_initial for r in results])),
        "worst_trade": float(pnl_initial.min()),
        "best_trade": float(pnl_initial.max()),
    }
