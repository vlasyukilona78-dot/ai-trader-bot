from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from trading.market_data.reconciliation import ExchangeSnapshot
from trading.signals.signal_types import StrategyIntent
from trading.state.models import TradeState


@dataclass
class StrategyContext:
    symbol: str
    market_ohlcv: pd.DataFrame
    mark_price: float
    exchange: ExchangeSnapshot
    synced_state: TradeState
    timeframe: str | None = None
    synced_state_updated_at: float | None = None
    sentiment_index: float | None = None
    sentiment_value: float | None = None
    sentiment_source: str | None = None
    sentiment_degraded: bool | None = None
    funding_rate: float | None = None
    funding_source: str | None = None
    funding_degraded: bool | None = None
    long_short_ratio: float | None = None
    long_short_ratio_source: str | None = None
    long_short_ratio_degraded: bool | None = None
    open_interest: float | None = None
    open_interest_ratio: float | None = None
    oi_signal: float | None = None
    oi_source: str | None = None
    oi_degraded: bool | None = None
    open_interest_source: str | None = None
    news_veto: bool | None = None
    news_source: str | None = None
    news_degraded: bool | None = None


class StrategyInterface(Protocol):
    def generate(self, context: StrategyContext) -> StrategyIntent: ...
