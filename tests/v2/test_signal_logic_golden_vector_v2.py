from __future__ import annotations

from hashlib import sha256
import json

import pandas as pd
import pytest

from core.mexc_strategy_spec import load_mexc_strategy_spec
from trading.exchange.schemas import AccountSnapshot
from trading.market_data.reconciliation import ExchangeSnapshot
from trading.signals.layered_strategy import LayeredPumpStrategy
from trading.signals.signal_types import IntentAction, StrategyIntent
from trading.signals.strategy_interface import StrategyContext
from trading.state.models import TradeState


PINNED_LAYERED_PUMP_SIGNAL_V1_DIGEST = (
    "d5736beda70ca2826dc4868c2d4d95cb17b1289ac2ba03a2a052d9db69587459"
)


def _pump_frame() -> pd.DataFrame:
    count = 80
    index = pd.date_range("2026-08-01T00:00:00Z", periods=count, freq="h")
    close = [100.0] * count
    for offset in range(25):
        close[count - 25 + offset] = 100.0 + 8.8 * offset / 24.0
    close[-3:] = [108.4, 108.9, 108.8]

    frame = pd.DataFrame(
        {
            "open": [value - 0.1 for value in close],
            "high": [value + 0.2 for value in close],
            "low": [value - 0.2 for value in close],
            "close": close,
            "volume": [100.0] * count,
        },
        index=index,
    )
    frame.iloc[-2, frame.columns.get_loc("high")] = 109.0
    frame["turnover"] = 10_000.0
    frame["rsi"] = 60.0
    frame.iloc[-2, frame.columns.get_loc("rsi")] = 80.0
    frame["volume_spike"] = 1.0
    frame.iloc[-2, frame.columns.get_loc("volume_spike")] = 5.0
    frame["bb_upper"] = 200.0
    frame.iloc[-2, frame.columns.get_loc("bb_upper")] = 108.0
    frame["bb_lower"] = 0.0
    frame["kc_upper"] = 200.0
    frame.iloc[-2, frame.columns.get_loc("kc_upper")] = 108.0
    frame["kc_lower"] = 0.0
    frame["atr"] = 5.01
    frame["vwap"] = 100.0
    frame["obv"] = range(count)
    frame["cvd"] = range(count)
    frame["ema20"] = 108.85
    frame["ema50"] = 105.0
    frame["adx"] = 30.0
    return frame


def _benchmark(frame: pd.DataFrame) -> pd.DataFrame:
    benchmark = frame[["open", "high", "low", "close", "volume"]].copy()
    benchmark[["open", "high", "low", "close"]] = 100.0
    return benchmark


def _higher_timeframe_frame() -> pd.DataFrame:
    index = pd.date_range("2026-07-20T00:00:00Z", periods=30, freq="4h")
    close = [100.0 + offset * 0.3 for offset in range(30)]
    return pd.DataFrame(
        {
            "open": close,
            "high": [min(109.0, value + 0.2) for value in close],
            "low": [value - 0.2 for value in close],
            "close": close,
            "volume": [100.0] * len(close),
        },
        index=index,
    )


def _confirmed_frame(frame: pd.DataFrame) -> pd.DataFrame:
    row = frame.iloc[-1].to_dict()
    row.update(open=108.8, high=108.9, low=108.2, close=108.5, atr=5.01)
    return pd.concat(
        [
            frame,
            pd.DataFrame(
                [row],
                index=[frame.index[-1] + pd.Timedelta(hours=1)],
            ),
        ]
    )


def _context(frame: pd.DataFrame) -> StrategyContext:
    exchange = ExchangeSnapshot(
        symbol="ALTUSDT",
        account=AccountSnapshot(
            equity_usdt=1_000.0,
            available_balance_usdt=1_000.0,
        ),
        positions=[],
        open_orders=[],
    )
    return StrategyContext(
        symbol="ALTUSDT",
        market_ohlcv=frame,
        mark_price=float(frame.iloc[-1]["close"]),
        exchange=exchange,
        synced_state=TradeState.FLAT,
        sentiment_index=78.0,
        sentiment_source="provided",
        funding_rate=0.001,
        long_short_ratio=1.2,
    )


def _stable_intent_payload(intent: StrategyIntent) -> dict[str, object]:
    # Wall-clock identity is deliberately excluded; every causal decision,
    # proposal, and diagnostic field remains covered by the pinned digest.
    metadata = {
        key: value
        for key, value in intent.metadata.items()
        if key != "legacy_signal_id"
    }
    return {
        "symbol": intent.symbol,
        "action": intent.action.value,
        "reason": intent.reason,
        "stop_loss": intent.stop_loss,
        "take_profit": intent.take_profit,
        "confidence": intent.confidence,
        "metadata": metadata,
    }


def _rounded(value):
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    return value


def _behavior_digest(*intents: StrategyIntent) -> str:
    payload = [_rounded(_stable_intent_payload(intent)) for intent in intents]
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def test_layered_pump_signal_v1_matches_the_frozen_decision_trace_and_proposal() -> None:
    spec = load_mexc_strategy_spec()
    assert spec.runtime_semantics.logic_revision == "layered_pump_signal_v1"

    frame = _pump_frame()
    higher_timeframe = _higher_timeframe_frame()
    strategy = LayeredPumpStrategy(strategy_spec=spec)
    strategy.set_benchmark(_benchmark(frame))

    class FixedHigherTimeframeCache:
        config = spec.to_timeframe_cache_config()

        @staticmethod
        def get(symbol: str, as_of: float | None = None) -> pd.DataFrame:
            assert symbol == "ALTUSDT"
            return higher_timeframe.copy()

    strategy.set_htf_cache(FixedHigherTimeframeCache())
    strategy.begin_sweep()
    armed = strategy.generate(_context(frame))
    strategy.begin_sweep()
    confirmed = strategy.generate(_context(_confirmed_frame(frame)))

    assert armed.action is IntentAction.HOLD
    assert armed.reason == "no_signal_layer_confirmation_pending"
    assert armed.metadata["layer_failed"] == "layer_confirmation_pending"
    assert armed.metadata["layer_trace"]["layers"]["layer_confirmation"] == {
        "passed": False,
        "details": {
            "status": "armed",
            "armed_close": 108.8,
            "invalidate_level": 111.405,
            "bars_waited": 0.0,
        },
    }
    assert confirmed.action is IntentAction.SHORT_ENTRY
    assert confirmed.reason == "layered_short_entry"
    assert confirmed.stop_loss == pytest.approx(111.405)
    assert confirmed.take_profit == pytest.approx(100.0923611111111)
    assert confirmed.confidence == pytest.approx(0.886)
    assert confirmed.metadata["layer_failed"] == ""
    assert confirmed.metadata["layer_trace"]["layers"]["layer_confirmation"][
        "details"
    ]["status"] == "confirmed"
    assert confirmed.metadata["layer_trace"]["layers"]["layer5_tp_sl"][
        "passed"
    ] is True
    assert _behavior_digest(armed, confirmed) == PINNED_LAYERED_PUMP_SIGNAL_V1_DIGEST
