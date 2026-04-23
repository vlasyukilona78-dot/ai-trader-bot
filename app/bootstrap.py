from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import logging
import os
import re
import time

from trading.exchange.bybit_adapter import BybitAdapterConfig
from trading.risk.limits import RiskLimits, load_risk_limits_from_env

logger = logging.getLogger(__name__)

_DISCOVERY_EXCLUDED_BASES = {
    "XAU",
    "XAUT",
    "XAG",
    "XAGT",
    "PAXG",
    "USD1",
    "US",
    "USDC",
    "FDUSD",
    "USDE",
}
_DISCOVERY_EXCLUDED_SYMBOLS = {
    "ETHBTCUSDT",
    "PUMPBTCUSDT",
}
_DISCOVERY_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "runtime"))


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



def _load_dotenv_file(path: str = ".env") -> None:
    candidates = [
        os.path.abspath(path),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", path)),
    ]
    seen: set[str] = set()
    for env_path in candidates:
        normalized = os.path.normcase(env_path)
        if normalized in seen or not os.path.exists(env_path):
            continue
        seen.add(normalized)
        try:
            with open(env_path, "r", encoding="utf-8-sig") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    if line.lower().startswith("export "):
                        line = line[7:].strip()
                    key, value = line.split("=", 1)
                    key = key.strip()
                    if not key:
                        continue
                    value = value.strip()
                    if value and value[0] not in ("'", '"') and " #" in value:
                        value = value.split(" #", 1)[0].strip()
                    parsed = value.strip().strip('"').strip("'")
                    current = os.getenv(key)
                    if current is None or not str(current).strip():
                        os.environ[key] = parsed
        except Exception:
            continue


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _env_first_nonempty(*names: str) -> str:
    for name in names:
        raw = os.getenv(name)
        if raw is None:
            continue
        value = str(raw).strip()
        if value:
            return value
    return ""


def _resolve_signal_profile_env() -> str:
    raw = str(os.getenv("BOT_SIGNAL_PROFILE", "both")).strip().lower()
    if raw in {"main", "early"}:
        return raw

    generic_demo = _env_first_nonempty(
        "BYBIT_DEMO_API_KEY",
        "BYBIT_DEMO_KEY",
        "DEMO_BYBIT_API_KEY",
        "DEMO_API_KEY",
        "BYBIT_API_KEY",
        "API_KEY",
    )
    if generic_demo:
        return "both"

    main_demo = _env_first_nonempty(
        "BYBIT_DEMO_API_KEY_MAIN",
        "MAIN_BYBIT_DEMO_API_KEY",
        "BYBIT_API_KEY_MAIN",
        "API_KEY_MAIN",
    )
    if main_demo:
        return "main"

    early_demo = _env_first_nonempty(
        "BYBIT_DEMO_API_KEY_EARLY",
        "EARLY_BYBIT_DEMO_API_KEY",
        "BYBIT_API_KEY_EARLY",
        "API_KEY_EARLY",
    )
    if early_demo:
        return "early"

    return "both"


def _env_first_nonempty_profiled(profile: str, *names: str) -> str:
    profiled = str(profile or "").strip().lower()
    if profiled not in {"main", "early"}:
        expanded: list[str] = []
        for name in names:
            expanded.extend(
                [
                    name,
                    f"{name}_MAIN",
                    f"MAIN_{name}",
                    f"{name}_EARLY",
                    f"EARLY_{name}",
                ]
            )
        return _env_first_nonempty(*expanded)

    suffix = profiled.upper()
    expanded: list[str] = []
    for name in names:
        expanded.extend(
            [
                f"{name}_{suffix}",
                f"{suffix}_{name}",
                name,
            ]
        )
    return _env_first_nonempty(*expanded)


def _env_first_nonempty_profiled_info(profile: str, *names: str) -> tuple[str, bool]:
    profiled = str(profile or "").strip().lower()
    if profiled not in {"main", "early"}:
        for name in names:
            for candidate, is_profiled in (
                (name, False),
                (f"{name}_MAIN", True),
                (f"MAIN_{name}", True),
                (f"{name}_EARLY", True),
                (f"EARLY_{name}", True),
            ):
                raw = os.getenv(candidate)
                if raw is None:
                    continue
                value = str(raw).strip()
                if value:
                    return value, is_profiled
        return "", False

    suffix = profiled.upper()
    for name in names:
        for candidate, is_profiled in (
            (f"{name}_{suffix}", True),
            (f"{suffix}_{name}", True),
            (name, False),
        ):
            raw = os.getenv(candidate)
            if raw is None:
                continue
            value = str(raw).strip()
            if value:
                return value, is_profiled
    return "", False


def _profiled_path_env(base_path: str, profile: str, *override_names: str) -> str:
    profiled = str(profile or "").strip().lower()
    explicit, explicit_is_profiled = _env_first_nonempty_profiled_info(profiled, *override_names)
    chosen_path = explicit or base_path
    if profiled not in {"main", "early"}:
        return chosen_path
    if explicit and explicit_is_profiled:
        return explicit
    path = os.path.abspath(chosen_path) if os.path.isabs(chosen_path) else chosen_path
    root, ext = os.path.splitext(path)
    ext = ext or ".db"
    return f"{root}_{profiled}{ext}"



def _parse_symbols(raw: str | None) -> list[str]:
    if not raw:
        return ["BTCUSDT"]
    out = []
    for token in str(raw).split(","):
        value = token.strip().replace("/", "").upper()
        if value:
            out.append(value)
    return sorted(set(out)) or ["BTCUSDT"]


def _symbol_base(symbol: str) -> str:
    clean = str(symbol or "").replace("/", "").upper().strip()
    for suffix in ("USDT", "USDC", "USD"):
        if clean.endswith(suffix) and len(clean) > len(suffix):
            return clean[: -len(suffix)]
    return clean


def _quality_symbol_allowed(
    symbol: str,
    *,
    launch_time_ms: int | float | str | None = None,
    min_days_listed: int = 0,
    now_ts_ms: int | None = None,
) -> bool:
    clean = str(symbol or "").replace("/", "").upper().strip()
    if not clean:
        return False
    if clean in _DISCOVERY_EXCLUDED_SYMBOLS:
        return False

    base = _symbol_base(clean)
    if not base or base in _DISCOVERY_EXCLUDED_BASES:
        return False

    if re.match(r"^\d{3,}", base):
        return False
    if re.search(r"\d{3,}$", base):
        return False

    alpha_count = sum(ch.isalpha() for ch in base)
    digit_count = sum(ch.isdigit() for ch in base)
    if alpha_count == 0:
        return False
    if len(base) == 1:
        return False
    if len(base) <= 2 and digit_count > 0:
        return False
    if digit_count > alpha_count and len(base) <= 4:
        return False
    if min_days_listed > 0 and launch_time_ms not in (None, "", 0, "0"):
        try:
            listed_ms = int(float(launch_time_ms))
        except (TypeError, ValueError):
            listed_ms = 0
        if listed_ms > 0:
            current_ms = int(now_ts_ms or int(time.time() * 1000))
            age_days = max(0.0, (current_ms - listed_ms) / 86_400_000.0)
            if age_days < float(min_days_listed):
                return False

    return True


def _legacy_env_testnet_default(default: bool = True) -> bool:
    raw_testnet = os.getenv("BYBIT_TESTNET")
    if raw_testnet is not None:
        return _env_bool("BYBIT_TESTNET", default)

    legacy_env = str(os.getenv("BYBIT_ENV", "")).strip().lower()
    if legacy_env == "mainnet":
        return False
    if legacy_env:
        return True
    return bool(default)


def _discovery_cache_path(*, testnet: bool) -> str:
    filename = "bybit_symbol_cache_testnet.json" if testnet else "bybit_symbol_cache_mainnet.json"
    return os.path.join(_DISCOVERY_CACHE_DIR, filename)


def _save_discovery_cache(
    *,
    testnet: bool,
    symbols: list[str],
    min_turnover_usdt: float,
    max_turnover_usdt: float,
    perpetual_only: bool,
    quality_filter: bool,
) -> None:
    if not symbols:
        return
    try:
        os.makedirs(_DISCOVERY_CACHE_DIR, exist_ok=True)
        payload = {
            "saved_at_ts": time.time(),
            "testnet": bool(testnet),
            "min_turnover_usdt": float(min_turnover_usdt),
            "max_turnover_usdt": float(max_turnover_usdt),
            "perpetual_only": bool(perpetual_only),
            "quality_filter": bool(quality_filter),
            "symbols": list(symbols),
        }
        with open(_discovery_cache_path(testnet=testnet), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
    except Exception as exc:
        logger.debug("symbol discovery cache save skipped: %s", exc)


def _load_discovery_cache(*, testnet: bool, limit: int) -> list[str]:
    try:
        cache_path = _discovery_cache_path(testnet=testnet)
        if not os.path.exists(cache_path):
            return []
        with open(cache_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return []
        cached_testnet = bool(payload.get("testnet", testnet))
        if cached_testnet != bool(testnet):
            return []
        symbols = payload.get("symbols", [])
        if not isinstance(symbols, list):
            return []
        cleaned = [str(symbol).upper().strip() for symbol in symbols if str(symbol).strip()]
        cleaned = list(dict.fromkeys(cleaned))
        if limit > 0:
            cleaned = cleaned[:limit]
        return cleaned
    except Exception as exc:
        logger.debug("symbol discovery cache load skipped: %s", exc)
        return []


def _discover_linear_usdt_symbols(
    *,
    testnet: bool,
    min_turnover_usdt: float,
    max_turnover_usdt: float,
    perpetual_only: bool,
    quality_filter: bool,
    limit: int,
) -> list[str]:
    cache_fallback = _load_discovery_cache(testnet=testnet, limit=limit)
    try:
        from trading.exchange.bybit_client import BybitHttpClient
    except Exception as exc:
        logger.warning("symbol discovery disabled: bybit client unavailable: %s", exc)
        return cache_fallback or ["BTCUSDT"]

    client = BybitHttpClient(
        api_key="",
        api_secret="",
        testnet=bool(testnet),
        dry_run=True,
    )
    try:
        tradable: set[str] = set()
        launch_time_by_symbol: dict[str, int] = {}
        min_days_listed = max(0, int(os.getenv("BOT_SYMBOL_DISCOVERY_MIN_DAYS_LISTED", "21")))
        now_ts_ms = int(time.time() * 1000)
        cursor = ""
        while True:
            params: dict[str, object] = {"category": "linear", "settleCoin": "USDT", "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            payload = client.request_public("/v5/market/instruments-info", params=params)
            result = payload.get("result", {}) if isinstance(payload, dict) else {}
            items = result.get("list", []) if isinstance(result, dict) else []
            if not isinstance(items, list) or not items:
                break

            for item in items:
                if not isinstance(item, dict):
                    continue
                symbol = str(item.get("symbol", "")).upper().strip()
                status = str(item.get("status", "")).upper().strip()
                quote_coin = str(item.get("quoteCoin", "")).upper().strip()
                contract_type = str(item.get("contractType", "")).upper().strip()
                if not symbol or quote_coin != "USDT":
                    continue
                if status and status != "TRADING":
                    continue
                if perpetual_only and "PERPETUAL" not in contract_type:
                    continue
                tradable.add(symbol)
                try:
                    launch_time_by_symbol[symbol] = int(float(item.get("launchTime", 0) or 0))
                except (TypeError, ValueError):
                    launch_time_by_symbol[symbol] = 0

            cursor = str(result.get("nextPageCursor") or "").strip()
            if not cursor:
                break

        tickers_payload = client.request_public("/v5/market/tickers", params={"category": "linear"})
        tickers_result = tickers_payload.get("result", {}) if isinstance(tickers_payload, dict) else {}
        tickers = tickers_result.get("list", []) if isinstance(tickers_result, dict) else []

        ranked: list[tuple[float, str]] = []
        for item in tickers if isinstance(tickers, list) else []:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol", "")).upper().strip()
            if not symbol or (tradable and symbol not in tradable):
                continue
            if quality_filter and not _quality_symbol_allowed(
                symbol,
                launch_time_ms=launch_time_by_symbol.get(symbol, 0),
                min_days_listed=min_days_listed,
                now_ts_ms=now_ts_ms,
            ):
                continue
            try:
                turnover = float(item.get("turnover24h", 0.0) or 0.0)
            except (TypeError, ValueError):
                turnover = 0.0
            if turnover < float(min_turnover_usdt) or turnover > float(max_turnover_usdt):
                continue
            ranked.append((turnover, symbol))

        ranked.sort(key=lambda pair: (-pair[0], pair[1]))
        symbols = [symbol for _, symbol in ranked]
        if limit > 0:
            symbols = symbols[:limit]
        if symbols:
            _save_discovery_cache(
                testnet=testnet,
                symbols=symbols,
                min_turnover_usdt=min_turnover_usdt,
                max_turnover_usdt=max_turnover_usdt,
                perpetual_only=perpetual_only,
                quality_filter=quality_filter,
            )
            return symbols
    except Exception as exc:
        if cache_fallback:
            logger.warning("symbol discovery failed, using cached universe (%d symbols): %s", len(cache_fallback), exc)
            return cache_fallback
        logger.warning("symbol discovery failed, falling back to BTCUSDT: %s", exc)
    finally:
        try:
            client.close()
        except Exception:
            pass

    return cache_fallback or ["BTCUSDT"]


def _resolve_symbols(*, testnet: bool) -> list[str]:
    raw = os.getenv("BOT_SYMBOLS", "BTCUSDT")
    if str(raw or "").strip().upper() == "ALL_BYBIT_LINEAR_USDT":
        return _discover_linear_usdt_symbols(
            testnet=testnet,
            min_turnover_usdt=float(os.getenv("BOT_SYMBOL_24H_TURNOVER_MIN_USDT", "200000")),
            max_turnover_usdt=float(os.getenv("BOT_SYMBOL_24H_TURNOVER_MAX_USDT", "200000000")),
            perpetual_only=_env_bool("BOT_SYMBOL_DISCOVERY_PERPETUAL_ONLY", True),
            quality_filter=_env_bool("BOT_SYMBOL_DISCOVERY_QUALITY_FILTER", True),
            limit=int(os.getenv("BOT_SYMBOLS_LIMIT", "0")),
        )
    return _parse_symbols(raw)



def _resolve_runtime_mode() -> str:
    mode_v2 = os.getenv("BOT_RUNTIME_MODE")
    mode_legacy = os.getenv("BOT_MODE")

    if mode_v2 and mode_legacy and mode_v2.strip().lower() != mode_legacy.strip().lower():
        raise ConfigError("BOT_RUNTIME_MODE conflicts with BOT_MODE")

    if mode_v2 or mode_legacy:
        mode = str(mode_v2 or mode_legacy).strip().lower()
    else:
        legacy_env = str(os.getenv("BYBIT_ENV", "")).strip().lower()
        if legacy_env == "demo":
            mode = "demo"
        elif legacy_env == "mainnet":
            mode = "paper"
        else:
            mode = "paper"
    if mode not in ("dry_run", "paper", "testnet", "demo", "live"):
        raise ConfigError("BOT_RUNTIME_MODE must be one of: dry_run,paper,testnet,demo,live")
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
    _load_dotenv_file()
    mode = _resolve_runtime_mode()
    signal_profile = _resolve_signal_profile_env()

    runtime_db_default = "data/runtime/v2_runtime.db"
    if mode == "demo":
        runtime_db_default = "data/runtime/v2_demo_runtime.db"
    elif mode == "testnet":
        runtime_db_default = "data/runtime/v2_testnet_runtime.db"
    elif mode == "live":
        runtime_db_default = "data/runtime/v2_live_runtime.db"

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

    symbols: list[str]
    env_testnet_raw = os.getenv("BYBIT_TESTNET")
    env_testnet = _legacy_env_testnet_default(True)

    bybit_demo = False
    if mode == "live":
        dry_run = False
        bybit_testnet = False
    elif mode == "testnet":
        dry_run = False
        bybit_testnet = True
    elif mode == "demo":
        dry_run = False
        bybit_testnet = False
        bybit_demo = True
    elif mode in ("paper", "dry_run"):
        dry_run = True
        bybit_testnet = True if mode == "dry_run" else env_testnet
    else:
        raise ConfigError("unsupported_mode")

    symbols = _resolve_symbols(testnet=bybit_testnet)

    if mode == "testnet" and env_testnet_raw is not None and not env_testnet:
        raise ConfigError("BYBIT_TESTNET must be true for BOT_RUNTIME_MODE=testnet")
    if mode == "live" and env_testnet_raw is not None and env_testnet:
        raise ConfigError("BYBIT_TESTNET must be false for BOT_RUNTIME_MODE=live")
    if mode == "demo" and env_testnet_raw is not None and env_testnet:
        raise ConfigError("BYBIT_TESTNET must be false for BOT_RUNTIME_MODE=demo")

    if mode == "demo":
        bybit_api_key = _env_first_nonempty_profiled(
            signal_profile,
            "BYBIT_DEMO_API_KEY",
            "BYBIT_DEMO_APIKEY",
            "BYBIT_DEMO_KEY",
            "DEMO_BYBIT_KEY",
            "DEMO_BYBIT_API_KEY",
            "DEMO_API_KEY",
            "DEMO_KEY",
            "BYBIT_API_KEY_DEMO",
            "API_KEY_DEMO",
            "BYBIT_DEMO_PUBLIC_KEY",
            "BYBIT_KEY",
            "API_KEY",
            "BYBIT_API_KEY",
        )
        bybit_api_secret = _env_first_nonempty_profiled(
            signal_profile,
            "BYBIT_DEMO_API_SECRET",
            "BYBIT_DEMO_APISECRET",
            "BYBIT_DEMO_SECRET",
            "DEMO_BYBIT_SECRET",
            "DEMO_BYBIT_API_SECRET",
            "DEMO_API_SECRET",
            "DEMO_SECRET",
            "BYBIT_API_SECRET_DEMO",
            "API_SECRET_DEMO",
            "BYBIT_DEMO_PRIVATE_KEY",
            "BYBIT_SECRET",
            "API_SECRET",
            "BYBIT_API_SECRET",
        )
    elif mode == "testnet":
        bybit_api_key = _env_first_nonempty_profiled(
            signal_profile,
            "BYBIT_TESTNET_API_KEY",
            "BYBIT_TESTNET_APIKEY",
            "BYBIT_TESTNET_KEY",
            "TESTNET_BYBIT_KEY",
            "TESTNET_BYBIT_API_KEY",
            "BYBIT_KEY",
            "API_KEY",
            "BYBIT_API_KEY",
        )
        bybit_api_secret = _env_first_nonempty_profiled(
            signal_profile,
            "BYBIT_TESTNET_API_SECRET",
            "BYBIT_TESTNET_APISECRET",
            "BYBIT_TESTNET_SECRET",
            "TESTNET_BYBIT_SECRET",
            "TESTNET_BYBIT_API_SECRET",
            "BYBIT_SECRET",
            "API_SECRET",
            "BYBIT_API_SECRET",
        )
    else:
        bybit_api_key = _env_first_nonempty_profiled(
            signal_profile,
            "BYBIT_MAINNET_API_KEY",
            "BYBIT_MAINNET_APIKEY",
            "BYBIT_MAINNET_KEY",
            "MAINNET_BYBIT_KEY",
            "MAINNET_BYBIT_API_KEY",
            "BYBIT_KEY",
            "API_KEY",
            "BYBIT_API_KEY",
        )
        bybit_api_secret = _env_first_nonempty_profiled(
            signal_profile,
            "BYBIT_MAINNET_API_SECRET",
            "BYBIT_MAINNET_APISECRET",
            "BYBIT_MAINNET_SECRET",
            "MAINNET_BYBIT_SECRET",
            "MAINNET_BYBIT_API_SECRET",
            "BYBIT_SECRET",
            "API_SECRET",
            "BYBIT_API_SECRET",
        )

    adapter_cfg = BybitAdapterConfig(
        api_key=bybit_api_key,
        api_secret=bybit_api_secret,
        testnet=bybit_testnet,
        demo=bybit_demo,
        dry_run=dry_run,
        recv_window=int(os.getenv("BYBIT_RECV_WINDOW", "20000")),
        hedge_mode=_env_bool("BYBIT_HEDGE_MODE", False),
        instrument_rules_ttl_sec=int(os.getenv("INSTRUMENT_RULES_TTL_SEC", "900")),
        instrument_rules_max_age_sec=int(os.getenv("INSTRUMENT_RULES_MAX_AGE_SEC", "3600")),
        ws_enabled=_env_bool("WS_ENABLED", True),
        ws_private_enabled=_env_bool("WS_PRIVATE_ENABLED", True),
        ws_stale_after_sec=int(os.getenv("WS_STALE_AFTER_SEC", "45")),
        ws_reconnect_delay_sec=float(os.getenv("WS_RECONNECT_DELAY_SEC", "1.0")),
        ws_open_timeout_sec=float(os.getenv("WS_OPEN_TIMEOUT_SEC", "12.0")),
        ws_close_timeout_sec=float(os.getenv("WS_CLOSE_TIMEOUT_SEC", "6.0")),
        ws_ping_interval_sec=float(os.getenv("WS_PING_INTERVAL_SEC", "60.0")),
        ws_ping_timeout_sec=float(os.getenv("WS_PING_TIMEOUT_SEC", "30.0")),
        ws_symbols=symbols,
        target_entry_leverage=float(os.getenv("BYBIT_TARGET_ENTRY_LEVERAGE", os.getenv("RISK_MAX_LEVERAGE", "3.0"))),
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
        runtime_db_path=_profiled_path_env(
            str(os.getenv("RUNTIME_DB_PATH", runtime_db_default)),
            signal_profile,
            "RUNTIME_DB_PATH",
        ),
        ws_poll_interval_sec=int(os.getenv("SYNC_POLL_INTERVAL_SEC", "10")),
        ws_event_staleness_sec=int(os.getenv("SYNC_WS_STALENESS_SEC", "60")),
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

    if cfg.mode in ("testnet", "demo", "live"):
        if not cfg.adapter.api_key or not cfg.adapter.api_secret:
            if cfg.mode == "demo":
                raise ConfigError(
                    "Demo mode requires API keys. Supported env vars: BYBIT_DEMO_API_KEY/BYBIT_DEMO_API_SECRET, profile-specific BYBIT_DEMO_API_KEY_MAIN/BYBIT_DEMO_API_SECRET_MAIN and BYBIT_DEMO_API_KEY_EARLY/BYBIT_DEMO_API_SECRET_EARLY, or generic BYBIT_API_KEY/BYBIT_API_SECRET."
                )
            if cfg.mode == "testnet":
                raise ConfigError(
                    "Testnet mode requires API keys. Supported env vars: BYBIT_TESTNET_API_KEY/BYBIT_TESTNET_API_SECRET or generic BYBIT_API_KEY/BYBIT_API_SECRET."
                )
            raise ConfigError(
                "Live mode requires API keys. Supported env vars: BYBIT_MAINNET_API_KEY/BYBIT_MAINNET_API_SECRET or generic BYBIT_API_KEY/BYBIT_API_SECRET."
            )

    if cfg.mode == "testnet":
        if cfg.adapter.testnet is not True:
            raise ConfigError("BYBIT_TESTNET must be true for BOT_RUNTIME_MODE=testnet")
        if cfg.adapter.dry_run:
            raise ConfigError("dry_run must be false for BOT_RUNTIME_MODE=testnet")

    if cfg.mode == "demo":
        if cfg.adapter.testnet:
            raise ConfigError("BYBIT_TESTNET must be false for BOT_RUNTIME_MODE=demo")
        if cfg.adapter.dry_run:
            raise ConfigError("dry_run must be false for BOT_RUNTIME_MODE=demo")
        if not cfg.adapter.demo:
            raise ConfigError("demo flag must be true for BOT_RUNTIME_MODE=demo")

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
