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
    sentiment_index: float | None = None
    sentiment_source: str | None = None
    funding_rate: float | None = None
    long_short_ratio: float | None = None


class StrategyInterface(Protocol):
    def generate(self, context: StrategyContext) -> StrategyIntent: ...
