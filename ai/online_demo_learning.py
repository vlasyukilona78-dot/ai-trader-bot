from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from ai.utils import DEFAULT_FEATURE_NAMES
except Exception:
    DEFAULT_FEATURE_NAMES = [
        "rsi",
        "ema20",
        "ema50",
        "atr",
        "vwap_dist",
        "bb_position",
        "volume_spike",
        "obv_div_short",
        "obv_div_long",
        "cvd_div_short",
        "cvd_div_long",
        "poc_dist",
        "vah_dist",
        "val_dist",
        "regime_num",
        "funding_rate",
        "open_interest",
        "long_short_ratio",
        "sentiment_index",
        "liq_high_dist",
        "liq_low_dist",
        "close_ret_5",
        "close_ret_20",
        "mtf_rsi_5m",
        "mtf_rsi_15m",
        "mtf_atr_norm_5m",
        "mtf_atr_norm_15m",
        "mtf_trend_5m",
        "mtf_trend_15m",
    ]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _norm_symbol(symbol: str) -> str:
    return str(symbol or "").replace("/", "").upper().strip()


@dataclass
class PendingTradeSample:
    symbol: str
    side: str
    market_regime: str
    entry_price: float
    qty: float
    entry_ts: float
    timeframe_minutes: int
    features: dict[str, float]


class TradeExecutionLearner:
    def __init__(
        self,
        *,
        dataset_path: str,
        pending_path: str = "data/runtime/trade_online_pending.json",
        timeframe_minutes: int = 1,
    ):
        self.dataset_path = Path(dataset_path)
        self.pending_path = Path(pending_path)
        self.timeframe_minutes = max(1, int(timeframe_minutes))
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self.pending_path.parent.mkdir(parents=True, exist_ok=True)
        self._pending = self._load_pending()

    def _load_pending(self) -> dict[str, PendingTradeSample]:
        if not self.pending_path.exists():
            return {}
        try:
            payload = json.loads(self.pending_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}

        out: dict[str, PendingTradeSample] = {}
        for key, row in payload.items():
            if not isinstance(row, dict):
                continue
            out[str(key)] = PendingTradeSample(
                symbol=_norm_symbol(row.get("symbol", key)),
                side=str(row.get("side", "")).upper(),
                market_regime=str(row.get("market_regime", "RANGE")).upper(),
                entry_price=_safe_float(row.get("entry_price"), 0.0),
                qty=max(_safe_float(row.get("qty"), 0.0), 0.0),
                entry_ts=_safe_float(row.get("entry_ts"), 0.0),
                timeframe_minutes=max(1, int(_safe_float(row.get("timeframe_minutes"), self.timeframe_minutes))),
                features={
                    str(name): _safe_float(value, 0.0)
                    for name, value in dict(row.get("features", {})).items()
                },
            )
        return out

    def _save_pending(self) -> None:
        payload = {
            symbol: {
                "symbol": trade.symbol,
                "side": trade.side,
                "market_regime": trade.market_regime,
                "entry_price": float(trade.entry_price),
                "qty": float(trade.qty),
                "entry_ts": float(trade.entry_ts),
                "timeframe_minutes": int(trade.timeframe_minutes),
                "features": {str(k): float(v) for k, v in trade.features.items()},
            }
            for symbol, trade in self._pending.items()
        }
        self.pending_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def record_entry(
        self,
        *,
        symbol: str,
        side: str,
        market_regime: str,
        entry_price: float,
        qty: float,
        entry_ts: float,
        features: dict[str, float],
    ) -> None:
        clean = _norm_symbol(symbol)
        feature_row = {name: _safe_float(features.get(name), 0.0) for name in DEFAULT_FEATURE_NAMES}
        self._pending[clean] = PendingTradeSample(
            symbol=clean,
            side=str(side or "").upper(),
            market_regime=str(market_regime or "RANGE").upper(),
            entry_price=max(_safe_float(entry_price), 1e-8),
            qty=max(_safe_float(qty), 0.0),
            entry_ts=max(_safe_float(entry_ts), 0.0),
            timeframe_minutes=self.timeframe_minutes,
            features=feature_row,
        )
        self._save_pending()

    def record_exit(
        self,
        *,
        symbol: str,
        exit_ts: float,
        realized_pnl: float,
        qty: float,
    ) -> dict[str, Any] | None:
        clean = _norm_symbol(symbol)
        trade = self._pending.pop(clean, None)
        if trade is None:
            return None

        self._save_pending()

        filled_qty = max(_safe_float(qty), trade.qty, 1e-8)
        entry_notional = max(trade.entry_price * filled_qty, 1e-8)
        future_return = float(_safe_float(realized_pnl) / entry_notional)
        target_win = 1 if future_return > 0 else 0
        elapsed_sec = max(_safe_float(exit_ts) - trade.entry_ts, 60.0)
        target_horizon = elapsed_sec / max(60.0 * trade.timeframe_minutes, 60.0)

        row = {
            "timestamp": pd.Timestamp.fromtimestamp(
                max(_safe_float(exit_ts), trade.entry_ts),
                tz="UTC",
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "market_regime": trade.market_regime,
            "target_win": int(target_win),
            "target_horizon": float(target_horizon),
            "future_return": float(future_return),
        }
        row.update({name: float(trade.features.get(name, 0.0)) for name in DEFAULT_FEATURE_NAMES})

        frame = pd.DataFrame([row])
        header = not self.dataset_path.exists() or self.dataset_path.stat().st_size <= 0
        frame.to_csv(self.dataset_path, mode="a", header=header, index=False)
        return row


DemoTradeLearner = TradeExecutionLearner
