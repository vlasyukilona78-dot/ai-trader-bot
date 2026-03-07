from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DegradationConfig:
    min_closed_trades: int = 20
    min_win_rate: float = 0.40
    max_api_errors_per_min: int = 30
    alert_cooldown_sec: int = 900


class MetricsTracker:
    def __init__(self, metrics_path: str = "logs/metrics.json", config: DegradationConfig | None = None):
        self.metrics_path = Path(metrics_path)
        self.config = config or DegradationConfig()
        self.signal_detected = 0
        self.signal_ordered = 0
        self.trade_closed = 0
        self.trade_pnl_history = deque(maxlen=500)
        self.api_errors = deque(maxlen=2000)
        self.last_alert_ts = 0.0

    def record_signal_detected(self):
        self.signal_detected += 1

    def record_signal_ordered(self):
        self.signal_ordered += 1

    def record_trade_closed(self, pnl: float):
        self.trade_closed += 1
        self.trade_pnl_history.append(float(pnl))

    def record_api_error(self):
        self.api_errors.append(time.time())

    def _win_rate(self) -> float:
        if not self.trade_pnl_history:
            return 0.0
        wins = sum(1 for p in self.trade_pnl_history if p > 0)
        losses = sum(1 for p in self.trade_pnl_history if p <= 0)
        if wins + losses == 0:
            return 0.0
        return wins / (wins + losses)

    def _api_errors_last_min(self) -> int:
        now = time.time()
        return sum(1 for ts in self.api_errors if now - ts <= 60)

    def should_alert_degradation(self) -> str | None:
        now = time.time()
        if now - self.last_alert_ts < self.config.alert_cooldown_sec:
            return None

        message = None
        if len(self.trade_pnl_history) >= self.config.min_closed_trades:
            win_rate = self._win_rate()
            if win_rate < self.config.min_win_rate:
                message = f"Degradation: low win rate {win_rate:.2%}"

        errors_last_min = self._api_errors_last_min()
        if errors_last_min >= self.config.max_api_errors_per_min:
            message = f"Degradation: API errors last minute = {errors_last_min}"

        if message:
            self.last_alert_ts = now
        return message

    def snapshot(self) -> dict:
        return {
            "signal_detected": self.signal_detected,
            "signal_ordered": self.signal_ordered,
            "trade_closed": self.trade_closed,
            "recent_trades": len(self.trade_pnl_history),
            "win_rate_recent": self._win_rate(),
            "api_errors_last_min": self._api_errors_last_min(),
            "updated_at": time.time(),
        }

    def flush(self):
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("w", encoding="utf-8") as f:
            json.dump(self.snapshot(), f, ensure_ascii=False, indent=2)


def build_metrics_tracker_from_env() -> MetricsTracker:
    config = DegradationConfig(
        min_closed_trades=int(os.getenv("DEGRADATION_MIN_TRADES", "20")),
        min_win_rate=float(os.getenv("DEGRADATION_MIN_WIN_RATE", "0.40")),
        max_api_errors_per_min=int(os.getenv("DEGRADATION_MAX_API_ERRORS_PER_MIN", "30")),
        alert_cooldown_sec=int(os.getenv("DEGRADATION_ALERT_COOLDOWN_SEC", "900")),
    )
    return MetricsTracker(metrics_path=os.getenv("METRICS_PATH", "logs/metrics.json"), config=config)
