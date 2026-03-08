from __future__ import annotations

from dataclasses import dataclass
import importlib
import os

from trading.exchange.bybit_adapter import BybitAdapterConfig
from trading.risk.limits import RiskLimits, load_risk_limits_from_env


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeFlags:
    live_trading_enabled: bool = False
    online_retraining_enabled: bool = False
    model_auto_promotion_enabled: bool = False
    reconciliation_required: bool = True
    feature_runtime_enabled: bool = True
    ml_inference_enabled: bool = False


@dataclass(frozen=True)
class AlertConfig:
    telegram_token: str = ""
    telegram_chat_id: str = ""
    discord_webhook_url: str = ""


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str
    symbols: list[str]
    timeframe: str
    candles_limit: int
    scan_interval_sec: int
    adapter: BybitAdapterConfig
    risk_limits: RiskLimits
    alerts: AlertConfig
    flags: RuntimeFlags
    runtime_db_path: str
    ws_poll_interval_sec: int
    ws_event_staleness_sec: int
    stop_attach_grace_sec: int
    stale_open_order_sec: int
    max_exchange_retries: int
    maintenance_interval_sec: int
    live_startup_max_notional_usdt: float



def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in ("1", "true", "yes", "on")



def _parse_symbols(raw: str | None) -> list[str]:
    if not raw:
        return ["BTCUSDT"]
    out = []
    for token in str(raw).split(","):
        value = token.strip().replace("/", "").upper()
        if value:
            out.append(value)
    return sorted(set(out)) or ["BTCUSDT"]



def _resolve_runtime_mode() -> str:
    mode_v2 = os.getenv("BOT_RUNTIME_MODE")
    mode_legacy = os.getenv("BOT_MODE")

    if mode_v2 and mode_legacy and mode_v2.strip().lower() != mode_legacy.strip().lower():
        raise ConfigError("BOT_RUNTIME_MODE conflicts with BOT_MODE")

    mode = str(mode_v2 or mode_legacy or "paper").strip().lower()
    if mode not in ("dry_run", "paper", "testnet", "live"):
        raise ConfigError("BOT_RUNTIME_MODE must be one of: dry_run,paper,testnet,live")
    return mode



def _require_modules(modules: list[str], *, context: str):
    missing: list[str] = []
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)
    if missing:
        raise ConfigError(f"{context}_deps_missing:{','.join(sorted(set(missing)))}")



def load_runtime_config() -> RuntimeConfig:
    mode = _resolve_runtime_mode()

    flags = RuntimeFlags(
        live_trading_enabled=_env_bool("LIVE_TRADING_ENABLED", False),
        online_retraining_enabled=_env_bool("ML_ONLINE_RETRAIN_ENABLED", False),
        model_auto_promotion_enabled=_env_bool("ML_AUTO_PROMOTION_ENABLED", False),
        reconciliation_required=_env_bool("RECONCILIATION_REQUIRED", True),
        feature_runtime_enabled=_env_bool("FEATURE_RUNTIME_ENABLED", True),
        ml_inference_enabled=_env_bool("ML_INFERENCE_ENABLED", False),
    )

    alerts = AlertConfig(
        telegram_token=os.getenv("TELEGRAM_TOKEN", ""),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", os.getenv("CHAT_ID", "")),
        discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL", ""),
    )

    symbols = _parse_symbols(os.getenv("BOT_SYMBOLS", "BTCUSDT"))
    env_testnet_raw = os.getenv("BYBIT_TESTNET")
    env_testnet = _env_bool("BYBIT_TESTNET", True)

    if mode == "live":
        dry_run = False
        bybit_testnet = False
    elif mode == "testnet":
        dry_run = False
        bybit_testnet = True
    elif mode in ("paper", "dry_run"):
        dry_run = True
        bybit_testnet = True if mode == "dry_run" else env_testnet
    else:
        raise ConfigError("unsupported_mode")

    if mode == "testnet" and env_testnet_raw is not None and not env_testnet:
        raise ConfigError("BYBIT_TESTNET must be true for BOT_RUNTIME_MODE=testnet")
    if mode == "live" and env_testnet_raw is not None and env_testnet:
        raise ConfigError("BYBIT_TESTNET must be false for BOT_RUNTIME_MODE=live")

    bybit_api_key = os.getenv("BYBIT_API_KEY", "")
    bybit_api_secret = os.getenv("BYBIT_API_SECRET", "")

    adapter_cfg = BybitAdapterConfig(
        api_key=bybit_api_key,
        api_secret=bybit_api_secret,
        testnet=bybit_testnet,
        dry_run=dry_run,
        recv_window=int(os.getenv("BYBIT_RECV_WINDOW", "5000")),
        hedge_mode=_env_bool("BYBIT_HEDGE_MODE", False),
        instrument_rules_ttl_sec=int(os.getenv("INSTRUMENT_RULES_TTL_SEC", "900")),
        instrument_rules_max_age_sec=int(os.getenv("INSTRUMENT_RULES_MAX_AGE_SEC", "3600")),
        ws_enabled=_env_bool("WS_ENABLED", True),
        ws_private_enabled=_env_bool("WS_PRIVATE_ENABLED", True),
        ws_stale_after_sec=int(os.getenv("WS_STALE_AFTER_SEC", "25")),
        ws_reconnect_delay_sec=float(os.getenv("WS_RECONNECT_DELAY_SEC", "1.0")),
        ws_symbols=symbols,
    )

    cfg = RuntimeConfig(
        mode=mode,
        symbols=symbols,
        timeframe=str(os.getenv("BOT_TIMEFRAME", "1")),
        candles_limit=int(os.getenv("BOT_CANDLES_LIMIT", "320")),
        scan_interval_sec=int(os.getenv("BOT_SCAN_INTERVAL_SEC", "30")),
        adapter=adapter_cfg,
        risk_limits=load_risk_limits_from_env(),
        alerts=alerts,
        flags=flags,
        runtime_db_path=str(os.getenv("RUNTIME_DB_PATH", "data/runtime/v2_runtime.db")),
        ws_poll_interval_sec=int(os.getenv("SYNC_POLL_INTERVAL_SEC", "10")),
        ws_event_staleness_sec=int(os.getenv("SYNC_WS_STALENESS_SEC", "20")),
        stop_attach_grace_sec=int(os.getenv("EXEC_STOP_ATTACH_GRACE_SEC", "8")),
        stale_open_order_sec=int(os.getenv("EXEC_STALE_OPEN_ORDER_SEC", "120")),
        max_exchange_retries=int(os.getenv("EXEC_MAX_EXCHANGE_RETRIES", "2")),
        maintenance_interval_sec=int(os.getenv("RUNTIME_MAINTENANCE_INTERVAL_SEC", "300")),
        live_startup_max_notional_usdt=float(os.getenv("LIVE_STARTUP_MAX_NOTIONAL_USDT", "0")),
    )
    validate_runtime_config(cfg)
    return cfg



def validate_runtime_config(cfg: RuntimeConfig):
    if cfg.mode == "live":
        if not cfg.flags.live_trading_enabled:
            raise ConfigError("LIVE trading is disabled. Set LIVE_TRADING_ENABLED=true explicitly.")
        if cfg.adapter.testnet:
            raise ConfigError("BYBIT_TESTNET must be false for BOT_RUNTIME_MODE=live")
        if cfg.adapter.dry_run:
            raise ConfigError("dry_run must be false for BOT_RUNTIME_MODE=live")
        if cfg.live_startup_max_notional_usdt <= 0:
            raise ConfigError("LIVE_STARTUP_MAX_NOTIONAL_USDT must be > 0 for live mode")

    if cfg.mode != "live" and cfg.flags.live_trading_enabled:
        raise ConfigError("LIVE_TRADING_ENABLED=true is only valid for BOT_RUNTIME_MODE=live")

    if cfg.mode in ("testnet", "live"):
        if not cfg.adapter.api_key or not cfg.adapter.api_secret:
            raise ConfigError("BYBIT_API_KEY and BYBIT_API_SECRET are required for testnet/live modes")

    if cfg.mode == "testnet":
        if cfg.adapter.testnet is not True:
            raise ConfigError("BYBIT_TESTNET must be true for BOT_RUNTIME_MODE=testnet")
        if cfg.adapter.dry_run:
            raise ConfigError("dry_run must be false for BOT_RUNTIME_MODE=testnet")

    if cfg.mode in ("dry_run", "paper") and not cfg.adapter.dry_run:
        raise ConfigError("dry_run must be true for BOT_RUNTIME_MODE=dry_run/paper")

    if cfg.risk_limits.max_risk_per_trade_pct <= 0 or cfg.risk_limits.max_risk_per_trade_pct > 0.05:
        raise ConfigError("RISK_MAX_RISK_PER_TRADE_PCT must be in (0, 0.05]")

    if cfg.risk_limits.max_leverage <= 0:
        raise ConfigError("RISK_MAX_LEVERAGE must be > 0")

    if cfg.mode == "live" and cfg.risk_limits.max_total_notional_pct > 0.25:
        raise ConfigError("RISK_MAX_TOTAL_NOTIONAL_PCT must be <= 0.25 for first live rollout")

    if cfg.scan_interval_sec < 1:
        raise ConfigError("BOT_SCAN_INTERVAL_SEC must be >= 1")

    if cfg.flags.online_retraining_enabled and cfg.mode == "live":
        raise ConfigError("ML_ONLINE_RETRAIN_ENABLED is not allowed in live mode by default")

    if cfg.ws_poll_interval_sec < 1:
        raise ConfigError("SYNC_POLL_INTERVAL_SEC must be >= 1")

    if cfg.ws_event_staleness_sec < 1:
        raise ConfigError("SYNC_WS_STALENESS_SEC must be >= 1")

    if cfg.stop_attach_grace_sec < 1:
        raise ConfigError("EXEC_STOP_ATTACH_GRACE_SEC must be >= 1")

    if cfg.stale_open_order_sec < 10:
        raise ConfigError("EXEC_STALE_OPEN_ORDER_SEC must be >= 10")

    if cfg.max_exchange_retries < 1 or cfg.max_exchange_retries > 5:
        raise ConfigError("EXEC_MAX_EXCHANGE_RETRIES must be in [1,5]")

    if cfg.maintenance_interval_sec < 30:
        raise ConfigError("RUNTIME_MAINTENANCE_INTERVAL_SEC must be >= 30")

    if cfg.flags.feature_runtime_enabled:
        _require_modules(["numpy", "pandas"], context="feature_runtime")

    if cfg.flags.ml_inference_enabled:
        _require_modules(["numpy", "pandas", "joblib", "sklearn"], context="ml_inference")

    if cfg.adapter.ws_enabled:
        _require_modules(["websockets"], context="websocket_runtime")
