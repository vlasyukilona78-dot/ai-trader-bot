from __future__ import annotations

from trading.signals.signal_types import IntentAction, StrategyIntent
from trading.signals.strategy_interface import StrategyContext, StrategyInterface


class HoldStrategy(StrategyInterface):
    def generate(self, context: StrategyContext) -> StrategyIntent:
        return StrategyIntent(symbol=context.symbol, action=IntentAction.HOLD, reason="default_hold")
