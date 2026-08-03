from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from tests.v2.test_layered_strategy_v2 import _context
from trading.signals.layered_strategy import LayeredPumpStrategy
from trading.state.models import TradeState


class _InterleavingGenerator:
    def __init__(self) -> None:
        self.last_diagnostics: dict[str, object] = {}
        self._counter_lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def generate(self, context):
        with self._counter_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.025)
            self.last_diagnostics = {
                "failed_layer": "layer1_pump_detection",
                "symbol_marker": context.symbol,
            }
            return None
        finally:
            with self._counter_lock:
                self.active -= 1


class _ParallelCache:
    def __init__(self) -> None:
        self._counter_lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def get(self, symbol: str, *, as_of: float | None = None):
        del symbol, as_of
        with self._counter_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.025)
            return None
        finally:
            with self._counter_lock:
                self.active -= 1


def _symbol_context(symbol: str):
    context = _context(TradeState.FLAT)
    context.symbol = symbol
    context.candle_cutoff_ts = 1_700_000_000.0
    return context


def test_mutable_generator_state_is_serialized_and_trace_stays_with_symbol() -> None:
    strategy = LayeredPumpStrategy()
    generator = _InterleavingGenerator()
    strategy._generator = generator

    symbols = ["AAA_USDT", "BBB_USDT", "CCC_USDT", "DDD_USDT"]
    with ThreadPoolExecutor(max_workers=len(symbols)) as pool:
        intents = list(pool.map(lambda symbol: strategy.generate(_symbol_context(symbol)), symbols))

    assert generator.max_active == 1
    assert [intent.symbol for intent in intents] == symbols
    assert [intent.metadata["layer_trace"]["symbol_marker"] for intent in intents] == symbols


def test_higher_timeframe_cache_access_remains_parallel() -> None:
    strategy = LayeredPumpStrategy()
    strategy._generator = _InterleavingGenerator()
    cache = _ParallelCache()
    strategy.set_htf_cache(cache)

    symbols = ["AAA_USDT", "BBB_USDT", "CCC_USDT", "DDD_USDT"]
    with ThreadPoolExecutor(max_workers=len(symbols)) as pool:
        list(pool.map(lambda symbol: strategy.generate(_symbol_context(symbol)), symbols))

    assert cache.max_active > 1
