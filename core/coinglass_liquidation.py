from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
import time
from typing import Any

import requests


logger = logging.getLogger("bot_v2")


_RANGE_VALUES = {"12h", "24h", "3d", "7d", "30d", "90d", "180d", "1y"}
_MODEL_VALUES = {"model1", "model2"}


@dataclass(frozen=True)
class CoinglassLiquidationConfig:
    enabled: bool = False
    api_key: str = ""
    base_url: str = "https://open-api-v4.coinglass.com"
    exchange: str = "Bybit"
    range: str = "3d"
    model: str = "model2"
    timeout_sec: float = 8.0
    ttl_sec: float = 600.0
    rate_limit_cooldown_sec: float = 900.0
    min_intensity: float = 0.12
    max_bands_per_side: int = 5
    proxy_url: str = ""


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    value = str(raw).strip().lower()
    if not value:
        return bool(default)
    return value in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return int(default)


def load_coinglass_liquidation_config() -> CoinglassLiquidationConfig:
    api_key = str(os.getenv("COINGLASS_API_KEY") or os.getenv("CG_API_KEY") or "").strip()
    enabled = _env_bool("COINGLASS_LIQUIDATION_ENABLED", bool(api_key))
    raw_range = str(os.getenv("COINGLASS_LIQUIDATION_RANGE") or "3d").strip()
    raw_model = str(os.getenv("COINGLASS_LIQUIDATION_MODEL") or "model2").strip().lower()
    return CoinglassLiquidationConfig(
        enabled=enabled and bool(api_key),
        api_key=api_key,
        base_url=str(os.getenv("COINGLASS_API_BASE_URL") or "https://open-api-v4.coinglass.com").strip().rstrip("/"),
        exchange=str(os.getenv("COINGLASS_LIQUIDATION_EXCHANGE") or "Bybit").strip() or "Bybit",
        range=raw_range if raw_range in _RANGE_VALUES else "3d",
        model=raw_model if raw_model in _MODEL_VALUES else "model2",
        timeout_sec=max(2.0, _env_float("COINGLASS_TIMEOUT_SEC", 8.0)),
        ttl_sec=max(30.0, _env_float("COINGLASS_LIQUIDATION_TTL_SEC", 600.0)),
        rate_limit_cooldown_sec=max(
            60.0,
            _env_float("COINGLASS_LIQUIDATION_RATE_LIMIT_COOLDOWN_SEC", 900.0),
        ),
        min_intensity=min(max(_env_float("COINGLASS_LIQUIDATION_MIN_INTENSITY", 0.12), 0.01), 0.90),
        max_bands_per_side=max(1, min(_env_int("COINGLASS_LIQUIDATION_MAX_BANDS_PER_SIDE", 5), 12)),
        proxy_url=str(os.getenv("COINGLASS_PROXY_URL") or "").strip(),
    )


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").replace("/", "").upper().strip()


def _coinglass_pair_symbol(symbol: str) -> str:
    return _normalize_symbol(symbol)


def _extract_data(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    data = payload.get("data")
    if isinstance(data, dict):
        return data
    result = payload.get("result")
    if isinstance(result, dict):
        return result
    return payload


def parse_coinglass_heatmap_bands(
    payload: Any,
    *,
    current_price: float,
    min_intensity: float = 0.12,
    max_bands_per_side: int = 5,
) -> list[dict[str, float | int | str]]:
    data = _extract_data(payload)
    y_values = data.get("y") or data.get("levels") or data.get("prices_y")
    liq_rows = data.get("liq") or data.get("liquidation") or data.get("heatmap")
    prices = data.get("prices") or data.get("price") or data.get("kline")
    if not isinstance(y_values, list) or not isinstance(liq_rows, list):
        return []

    levels = [_safe_float(value, 0.0) for value in y_values]
    if not levels:
        return []

    x_to_ts: dict[int, int] = {}
    if isinstance(prices, list):
        for idx, row in enumerate(prices):
            if not isinstance(row, (list, tuple)) or not row:
                continue
            ts = int(_safe_float(row[0], 0.0))
            if ts > 10_000_000_000:
                ts //= 1000
            if ts > 0:
                x_to_ts[idx] = ts

    by_level: dict[int, dict[str, float | int]] = {}
    max_cell = 0.0
    for row in liq_rows:
        if isinstance(row, dict):
            x_idx = int(_safe_float(row.get("x") or row.get("index_0"), -1.0))
            y_idx = int(_safe_float(row.get("y") or row.get("index_1"), -1.0))
            value = _safe_float(row.get("value") or row.get("liq") or row.get("index_2"), 0.0)
        elif isinstance(row, (list, tuple)) and len(row) >= 3:
            x_idx = int(_safe_float(row[0], -1.0))
            y_idx = int(_safe_float(row[1], -1.0))
            value = _safe_float(row[2], 0.0)
        else:
            continue
        if x_idx < 0 or y_idx < 0 or y_idx >= len(levels) or value <= 0:
            continue
        max_cell = max(max_cell, value)
        bucket = by_level.setdefault(
            y_idx,
            {
                "value_sum": 0.0,
                "value_max": 0.0,
                "x_min": x_idx,
                "x_max": x_idx,
            },
        )
        bucket["value_sum"] = float(bucket["value_sum"]) + value
        bucket["value_max"] = max(float(bucket["value_max"]), value)
        bucket["x_min"] = min(int(bucket["x_min"]), x_idx)
        bucket["x_max"] = max(int(bucket["x_max"]), x_idx)

    if not by_level or max_cell <= 0.0 or current_price <= 0.0:
        return []

    rows: list[dict[str, float | int | str]] = []
    for y_idx, item in by_level.items():
        level = levels[y_idx]
        if level <= 0:
            continue
        rel_intensity = float(item["value_max"]) / max_cell
        if rel_intensity < min_intensity:
            continue
        side = "above" if level >= current_price else "below"
        x_min = int(item["x_min"])
        x_max = int(item["x_max"])
        weight = 1.0 + 4.0 * min(max(rel_intensity, 0.0), 1.0)
        row: dict[str, float | int | str] = {
            "level": float(level),
            "weight": float(weight),
            "side": side,
            "source": "coinglass",
            "x_min": x_min,
            "x_max": x_max,
        }
        if x_min in x_to_ts:
            row["start_ts"] = int(x_to_ts[x_min])
        if x_max in x_to_ts:
            row["end_ts"] = int(x_to_ts[x_max])
        rows.append(row)

    above = sorted((row for row in rows if row["side"] == "above"), key=lambda row: float(row["weight"]), reverse=True)
    below = sorted((row for row in rows if row["side"] == "below"), key=lambda row: float(row["weight"]), reverse=True)
    kept = above[:max_bands_per_side] + below[:max_bands_per_side]
    return sorted(kept, key=lambda row: (str(row["side"]), float(row["level"])))


class CoinglassLiquidationClient:
    def __init__(self, cfg: CoinglassLiquidationConfig | None = None):
        self.cfg = cfg or load_coinglass_liquidation_config()
        self._session = requests.Session()
        self._session.trust_env = False
        self._session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "koteika-ultra/2.0",
            }
        )
        if self.cfg.api_key:
            self._session.headers.update({"CG-API-KEY": self.cfg.api_key})
        if self.cfg.proxy_url:
            self._session.proxies.update({"http": self.cfg.proxy_url, "https": self.cfg.proxy_url})
        self._cache: dict[tuple[str, str, str, str], tuple[float, list[dict[str, float | int | str]]]] = {}
        self._last_error_log_ts = 0.0
        self._global_cooldown_until_ts = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled and self.cfg.api_key)

    def close(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass

    def _endpoint(self) -> str:
        return f"{self.cfg.base_url}/api/futures/liquidation/heatmap/{self.cfg.model}"

    def fetch_heatmap_bands(self, symbol: str, *, current_price: float) -> list[dict[str, float | int | str]]:
        if not self.enabled or current_price <= 0.0:
            return []
        pair_symbol = _coinglass_pair_symbol(symbol)
        cache_key = (self.cfg.exchange, pair_symbol, self.cfg.range, self.cfg.model)
        now = time.time()
        cached = self._cache.get(cache_key)
        if cached is not None and now - cached[0] < self.cfg.ttl_sec:
            return list(cached[1])
        if now < self._global_cooldown_until_ts:
            remaining = max(0.0, self._global_cooldown_until_ts - now)
            self._log_fetch_issue(pair_symbol, f"rate_limit_cooldown active remaining_sec={remaining:.0f}")
            return []

        try:
            response = self._session.get(
                self._endpoint(),
                params={
                    "exchange": self.cfg.exchange,
                    "symbol": pair_symbol,
                    "range": self.cfg.range,
                },
                timeout=(3, max(4, int(self.cfg.timeout_sec))),
            )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "")
                delay_sec = self._rate_limit_cooldown_sec(retry_after)
                self._global_cooldown_until_ts = max(self._global_cooldown_until_ts, now + delay_sec)
                self._log_fetch_issue(
                    pair_symbol,
                    f"rate_limited status=429 retry_after={retry_after or 'n/a'} cooldown_sec={delay_sec:.0f}",
                )
                self._cache[cache_key] = (now, [])
                return []
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            self._log_fetch_issue(pair_symbol, str(exc))
            self._cache[cache_key] = (now, [])
            return []

        bands = parse_coinglass_heatmap_bands(
            payload,
            current_price=current_price,
            min_intensity=self.cfg.min_intensity,
            max_bands_per_side=self.cfg.max_bands_per_side,
        )
        self._cache[cache_key] = (now, bands)
        return list(bands)

    def _rate_limit_cooldown_sec(self, retry_after: object) -> float:
        delay = _safe_float(retry_after, self.cfg.rate_limit_cooldown_sec)
        if delay <= 0.0:
            delay = self.cfg.rate_limit_cooldown_sec
        return max(60.0, min(max(delay, self.cfg.rate_limit_cooldown_sec), 3600.0))

    def _log_fetch_issue(self, symbol: str, issue: str) -> None:
        now = time.time()
        if now - self._last_error_log_ts < 300:
            logger.debug("coinglass liquidation heatmap fetch skipped symbol=%s issue=%s", symbol, issue)
            return
        self._last_error_log_ts = now
        logger.warning("coinglass liquidation heatmap unavailable symbol=%s issue=%s fallback=internal_map", symbol, issue)
