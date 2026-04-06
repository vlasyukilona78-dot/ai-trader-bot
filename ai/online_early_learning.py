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


def _safe_timestamp(value: Any) -> str:
    if value is None:
        return ""
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.isoformat()
    except Exception:
        return ""


@dataclass
class PendingEarlySignalSample:
    symbol: str
    phase: str
    market_regime: str
    signal_price: float
    signal_ts: float
    signal_bar_ts: str
    timeframe_minutes: int
    horizon_bars: int
    success_move_pct: float
    failure_move_pct: float
    features: dict[str, float]


class EarlySignalOutcomeLearner:
    def __init__(
        self,
        *,
        dataset_path: str,
        pending_path: str = "data/runtime/early_online_pending.json",
        timeframe_minutes: int = 1,
    ):
        self.dataset_path = Path(dataset_path)
        self.pending_path = Path(pending_path)
        self.timeframe_minutes = max(1, int(timeframe_minutes))
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self.pending_path.parent.mkdir(parents=True, exist_ok=True)
        self._pending = self._load_pending()

    def _load_pending(self) -> dict[str, PendingEarlySignalSample]:
        if not self.pending_path.exists():
            return {}
        try:
            payload = json.loads(self.pending_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}

        out: dict[str, PendingEarlySignalSample] = {}
        for key, row in payload.items():
            if not isinstance(row, dict):
                continue
            out[str(key)] = PendingEarlySignalSample(
                symbol=_norm_symbol(row.get("symbol", key)),
                phase=str(row.get("phase", "WATCH")).upper(),
                market_regime=str(row.get("market_regime", "RANGE")).upper(),
                signal_price=max(_safe_float(row.get("signal_price"), 0.0), 1e-8),
                signal_ts=max(_safe_float(row.get("signal_ts"), 0.0), 0.0),
                signal_bar_ts=str(row.get("signal_bar_ts", "")),
                timeframe_minutes=max(1, int(_safe_float(row.get("timeframe_minutes"), self.timeframe_minutes))),
                horizon_bars=max(1, int(_safe_float(row.get("horizon_bars"), 24))),
                success_move_pct=max(_safe_float(row.get("success_move_pct"), 0.008), 0.001),
                failure_move_pct=max(_safe_float(row.get("failure_move_pct"), 0.006), 0.001),
                features={
                    str(name): _safe_float(value, 0.0)
                    for name, value in dict(row.get("features", {})).items()
                },
            )
        return out

    def _save_pending(self) -> None:
        payload = {
            symbol: {
                "symbol": sample.symbol,
                "phase": sample.phase,
                "market_regime": sample.market_regime,
                "signal_price": float(sample.signal_price),
                "signal_ts": float(sample.signal_ts),
                "signal_bar_ts": sample.signal_bar_ts,
                "timeframe_minutes": int(sample.timeframe_minutes),
                "horizon_bars": int(sample.horizon_bars),
                "success_move_pct": float(sample.success_move_pct),
                "failure_move_pct": float(sample.failure_move_pct),
                "features": {str(k): float(v) for k, v in sample.features.items()},
            }
            for symbol, sample in self._pending.items()
        }
        self.pending_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def record_signal(
        self,
        *,
        symbol: str,
        phase: str,
        market_regime: str,
        signal_price: float,
        signal_ts: float,
        signal_bar_ts: Any,
        features: dict[str, float],
        horizon_bars: int = 30,
        success_move_pct: float = 0.01,
        failure_move_pct: float = 0.008,
    ) -> None:
        clean = _norm_symbol(symbol)
        feature_row = {name: _safe_float(features.get(name), 0.0) for name in DEFAULT_FEATURE_NAMES}
        self._pending[clean] = PendingEarlySignalSample(
            symbol=clean,
            phase=str(phase or "WATCH").upper(),
            market_regime=str(market_regime or "RANGE").upper(),
            signal_price=max(_safe_float(signal_price), 1e-8),
            signal_ts=max(_safe_float(signal_ts), 0.0),
            signal_bar_ts=_safe_timestamp(signal_bar_ts),
            timeframe_minutes=self.timeframe_minutes,
            horizon_bars=max(1, int(horizon_bars)),
            success_move_pct=max(_safe_float(success_move_pct, 0.01), 0.001),
            failure_move_pct=max(_safe_float(failure_move_pct, 0.008), 0.001),
            features=feature_row,
        )
        self._save_pending()

    def resolve_with_frame(self, *, symbol: str, enriched: pd.DataFrame) -> dict[str, Any] | None:
        clean = _norm_symbol(symbol)
        sample = self._pending.get(clean)
        if sample is None or enriched is None or enriched.empty:
            return None

        try:
            index_utc = pd.to_datetime(enriched.index, utc=True, errors="coerce")
        except Exception:
            return None
        if len(index_utc) < sample.horizon_bars + 2:
            return None

        signal_bar_ts = pd.Timestamp(sample.signal_bar_ts) if sample.signal_bar_ts else pd.NaT
        if pd.isna(signal_bar_ts):
            return None
        if signal_bar_ts.tzinfo is None:
            signal_bar_ts = signal_bar_ts.tz_localize("UTC")
        else:
            signal_bar_ts = signal_bar_ts.tz_convert("UTC")

        future_mask = index_utc > signal_bar_ts
        if int(future_mask.sum()) < sample.horizon_bars:
            return None

        future = enriched.loc[future_mask].copy()
        future_index = index_utc[future_mask]
        future = future.iloc[: sample.horizon_bars]
        future_index = future_index[: sample.horizon_bars]
        if future.empty:
            return None

        highs = pd.to_numeric(future.get("high"), errors="coerce").fillna(sample.signal_price)
        lows = pd.to_numeric(future.get("low"), errors="coerce").fillna(sample.signal_price)
        signal_price = max(sample.signal_price, 1e-8)
        favorable = max((signal_price - float(lows.min())) / signal_price, 0.0)
        adverse = max((float(highs.max()) - signal_price) / signal_price, 0.0)

        success_threshold = max(sample.success_move_pct, 0.001)
        failure_threshold = max(sample.failure_move_pct, 0.001)

        success_hit_idx = next(
            (
                idx
                for idx, value in enumerate(lows.tolist(), start=1)
                if max((signal_price - float(value)) / signal_price, 0.0) >= success_threshold
            ),
            None,
        )
        failure_hit_idx = next(
            (
                idx
                for idx, value in enumerate(highs.tolist(), start=1)
                if max((float(value) - signal_price) / signal_price, 0.0) >= failure_threshold
            ),
            None,
        )

        if success_hit_idx is not None and (failure_hit_idx is None or success_hit_idx <= failure_hit_idx):
            target_win = 1
            target_horizon = float(success_hit_idx)
        elif failure_hit_idx is not None and (success_hit_idx is None or failure_hit_idx < success_hit_idx):
            target_win = 0
            target_horizon = float(failure_hit_idx)
        else:
            target_win = 1 if favorable >= success_threshold * 0.8 and favorable > adverse else 0
            target_horizon = float(len(future))

        row = {
            "timestamp": pd.Timestamp(future_index[-1]).strftime("%Y-%m-%d %H:%M:%S"),
            "market_regime": sample.market_regime,
            "target_win": int(target_win),
            "target_horizon": float(target_horizon),
            "future_return": float(favorable - adverse),
            "signal_phase": sample.phase,
            "signal_family": "early",
        }
        row.update({name: float(sample.features.get(name, 0.0)) for name in DEFAULT_FEATURE_NAMES})

        frame = pd.DataFrame([row])
        header = not self.dataset_path.exists() or self.dataset_path.stat().st_size <= 0
        frame.to_csv(self.dataset_path, mode="a", header=header, index=False)

        self._pending.pop(clean, None)
        self._save_pending()
        return row
