from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    import yaml
except Exception:
    yaml = None


@dataclass
class BotSettings:
    scan_interval_sec: int = 30
    timeframe: str = "1"
    candles_limit: int = 320
    symbols_limit: int = 0
    progress_log_every: int = 10
    fast_scan_in_dry_run: bool = True
    dry_run: bool = True


@dataclass
class StrategySettings:
    rsi_high: float = 68.0
    rsi_low: float = 32.0
    volume_spike_threshold: float = 1.6
    weakness_lookback: int = 4
    sentiment_bullish_threshold: float = 68.0
    sentiment_bearish_threshold: float = 32.0
    risk_reward: float = 1.6
    atr_sl_mult: float = 1.0
    entry_tolerance_pct: float = 0.004
    vwap_tolerance_pct: float = 0.0025
    funding_tolerance: float = 0.0003
    long_short_ratio_tolerance: float = 0.10
    volume_profile_window: int = 120
    volume_profile_bins: int = 48


@dataclass
class RiskSettings:
    account_equity_usdt: float = 1000.0
    max_risk_per_trade: float = 0.01
    max_open_positions: int = 3
    max_total_exposure_pct: float = 0.5
    daily_loss_limit_pct: float = 0.05
    max_consecutive_losses: int = 4
    cooldown_minutes: int = 30
    min_qty: float = 0.001
    max_qty: float = 100.0
    slippage_bps: float = 2.0


@dataclass
class MLSettings:
    min_probability: float = 0.25
    model_dir: str = "ai/models"
    use_regime_models: bool = True
    online_retrain_enabled: bool = True
    online_retrain_interval_sec: int = 6 * 3600
    online_retrain_min_rows: int = 100
    online_dataset_path: str = "data/processed/online_training_dataset.csv"


@dataclass
class AlertSettings:
    send_chart: bool = True


@dataclass
class MarketDataSettings:
    bybit_base_url: str = "https://api.bybit.com"
    sentiment_api_url: str = "https://api.alternative.me/fng/"
    request_timeout_sec: int = 8
    max_retries: int = 2


@dataclass
class AppSettings:
    bot: BotSettings
    strategy: StrategySettings
    risk: RiskSettings
    ml: MLSettings
    alerts: AlertSettings
    market_data: MarketDataSettings



def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return default


def _merge_dataclass(dc_obj, payload: dict[str, Any] | None):
    if not payload:
        return dc_obj
    for key, value in payload.items():
        if hasattr(dc_obj, key):
            setattr(dc_obj, key, value)
    return dc_obj


def load_settings(config_path: str = "config/config.yaml", env_path: str = "config/secrets.env") -> AppSettings:
    if Path(env_path).exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    raw: dict[str, Any] = {}
    if yaml is not None and Path(config_path).exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                parsed = yaml.safe_load(f) or {}
            if isinstance(parsed, dict):
                raw = parsed
        except Exception:
            raw = {}

    bot = _merge_dataclass(BotSettings(), raw.get("bot"))
    strategy = _merge_dataclass(StrategySettings(), raw.get("strategy"))
    risk = _merge_dataclass(RiskSettings(), raw.get("risk"))
    ml = _merge_dataclass(MLSettings(), raw.get("ml"))
    alerts = _merge_dataclass(AlertSettings(), raw.get("alerts"))
    market_data = _merge_dataclass(MarketDataSettings(), raw.get("market_data"))

    # env overrides
    bot.scan_interval_sec = int(os.getenv("BOT_SCAN_INTERVAL_SEC", bot.scan_interval_sec))
    bot.timeframe = str(os.getenv("BOT_TIMEFRAME", bot.timeframe))
    bot.candles_limit = int(os.getenv("BOT_CANDLES_LIMIT", bot.candles_limit))
    bot.symbols_limit = int(os.getenv("BOT_SYMBOLS_LIMIT", bot.symbols_limit))
    bot.progress_log_every = int(os.getenv("BOT_PROGRESS_LOG_EVERY", bot.progress_log_every))
    bot.fast_scan_in_dry_run = _as_bool(os.getenv("BOT_FAST_SCAN_IN_DRY_RUN"), bot.fast_scan_in_dry_run)
    bot.dry_run = _as_bool(os.getenv("BOT_DRY_RUN"), bot.dry_run)

    market_data.sentiment_api_url = os.getenv("SENTIMENT_API_URL", market_data.sentiment_api_url)
    market_data.bybit_base_url = os.getenv("BYBIT_BASE_URL", market_data.bybit_base_url)
    market_data.request_timeout_sec = int(os.getenv("MD_TIMEOUT_SEC", market_data.request_timeout_sec))
    market_data.max_retries = int(os.getenv("MD_MAX_RETRIES", market_data.max_retries))

    ml.online_retrain_enabled = _as_bool(os.getenv("ML_ONLINE_RETRAIN_ENABLED"), ml.online_retrain_enabled)
    ml.online_retrain_interval_sec = int(os.getenv("ML_ONLINE_RETRAIN_INTERVAL_SEC", ml.online_retrain_interval_sec))
    ml.online_retrain_min_rows = int(os.getenv("ML_ONLINE_RETRAIN_MIN_ROWS", ml.online_retrain_min_rows))
    ml.online_dataset_path = str(os.getenv("ML_ONLINE_DATASET_PATH", ml.online_dataset_path))
    ml.min_probability = float(os.getenv("ML_MIN_PROBABILITY", ml.min_probability))

    return AppSettings(
        bot=bot,
        strategy=strategy,
        risk=risk,
        ml=ml,
        alerts=alerts,
        market_data=market_data,
    )





