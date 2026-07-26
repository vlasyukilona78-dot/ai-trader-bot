from __future__ import annotations

import os
import time
from dataclasses import dataclass

import pandas as pd

from trading.market_data.mexc_client import MexcContractClient

_INTERVAL_SECONDS = {
    "Min1": 60,
    "Min5": 300,
    "Min15": 900,
    "Min30": 1800,
    "Min60": 3600,
    "Hour4": 14400,
    "Hour8": 28800,
    "Day1": 86400,
}

_MAX_BARS_PER_REQUEST = 2000
_OHLCV_COLUMNS = ["time", "open", "high", "low", "close", "volume"]
# Exact quote turnover from the kline. `volume` is a contract count and is not
# comparable across symbols without the contract size, so liquidity work must
# read this instead.
_TURNOVER = "turnover"


@dataclass
class HistoryConfig:
    cache_dir: str = "data/history"
    request_pause_sec: float = 0.05
    max_pages: int = 60


class HistoryCollector:
    """Paginated historical OHLCV downloader with an on-disk cache.

    MEXC caps a kline response at 2000 bars, so longer ranges are walked in
    windows and stitched. Results are cached per (symbol, interval) so repeated
    dataset builds do not re-download, and only the missing tail is fetched.
    """

    def __init__(self, client: MexcContractClient | None = None, config: HistoryConfig | None = None):
        self.client = client or MexcContractClient()
        self.config = config or HistoryConfig()
        os.makedirs(self.config.cache_dir, exist_ok=True)

    def _cache_path(self, symbol: str, interval: str) -> str:
        safe = self.client.denormalize_symbol(symbol)
        return os.path.join(self.config.cache_dir, f"{safe}_{interval}.csv")

    def _read_cache(self, symbol: str, interval: str) -> pd.DataFrame:
        path = self._cache_path(symbol, interval)
        if not os.path.exists(path):
            return pd.DataFrame(columns=_OHLCV_COLUMNS)
        try:
            cached = pd.read_csv(path)
        except Exception:
            return pd.DataFrame(columns=_OHLCV_COLUMNS)

        # Caches written before turnover was captured cannot be repaired from
        # what they contain, and silently mixing them with new rows would leave
        # half the history without comparable liquidity. Refetch instead.
        if _TURNOVER not in cached.columns:
            return pd.DataFrame(columns=_OHLCV_COLUMNS)
        return cached

    def _write_cache(self, symbol: str, interval: str, df: pd.DataFrame):
        df.sort_values("time").drop_duplicates("time").to_csv(self._cache_path(symbol, interval), index=False)

    def _fetch_window(self, symbol: str, interval: str, start: int, end: int) -> pd.DataFrame:
        mexc_symbol = self.client.normalize_symbol(symbol)
        payload = self.client._request_public(
            f"/api/v1/contract/kline/{mexc_symbol}",
            params={"interval": interval, "start": int(start), "end": int(end)},
        )
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict) or not data.get("time"):
            return pd.DataFrame(columns=_OHLCV_COLUMNS)
        return pd.DataFrame(
            {
                "time": pd.to_numeric(pd.Series(data["time"]), errors="coerce"),
                "open": pd.to_numeric(pd.Series(data["open"]), errors="coerce"),
                "high": pd.to_numeric(pd.Series(data["high"]), errors="coerce"),
                "low": pd.to_numeric(pd.Series(data["low"]), errors="coerce"),
                "close": pd.to_numeric(pd.Series(data["close"]), errors="coerce"),
                "volume": pd.to_numeric(pd.Series(data["vol"]), errors="coerce"),
                _TURNOVER: pd.to_numeric(pd.Series(data.get("amount", [])), errors="coerce"),
            }
        ).dropna(subset=_OHLCV_COLUMNS)

    def fetch_range(
        self,
        symbol: str,
        interval: str,
        start_ts: int,
        end_ts: int | None = None,
        *,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """OHLCV between two epoch-second bounds, indexed by UTC datetime."""
        end_ts = int(end_ts if end_ts is not None else time.time())
        start_ts = int(start_ts)
        step = _INTERVAL_SECONDS.get(interval)
        if step is None:
            raise ValueError(f"unsupported interval: {interval}")

        cached = self._read_cache(symbol, interval) if use_cache else pd.DataFrame(columns=_OHLCV_COLUMNS)
        have = set(cached["time"].astype("int64")) if not cached.empty else set()

        span = _MAX_BARS_PER_REQUEST * step
        frames = [cached] if not cached.empty else []
        cursor = start_ts
        pages = 0

        while cursor < end_ts and pages < self.config.max_pages:
            window_end = min(cursor + span, end_ts)
            expected = range(cursor, window_end, step)
            # Skip windows already fully covered by cache.
            if have and all(t in have for t in list(expected)[::50] or [cursor]):
                cursor = window_end
                continue

            chunk = self._fetch_window(symbol, interval, cursor, window_end)
            pages += 1
            if chunk.empty:
                cursor = window_end
                continue
            frames.append(chunk)
            last = int(chunk["time"].max())
            cursor = max(last + step, cursor + step)
            if self.config.request_pause_sec:
                time.sleep(self.config.request_pause_sec)

        if not frames:
            return pd.DataFrame(columns=_OHLCV_COLUMNS)

        out = pd.concat(frames, ignore_index=True)
        out = out.dropna(subset=_OHLCV_COLUMNS).sort_values("time").drop_duplicates("time")
        if use_cache:
            self._write_cache(symbol, interval, out)

        out = out[(out["time"] >= start_ts) & (out["time"] <= end_ts)].reset_index(drop=True)
        out["datetime"] = pd.to_datetime(out["time"], unit="s", utc=True)
        return out.set_index("datetime")

    def fetch_funding_history(self, symbol: str, pages: int = 12) -> pd.DataFrame:
        """Historical funding settlements (8h cycle). Empty frame if unavailable."""
        mexc_symbol = self.client.normalize_symbol(symbol)
        rows: list[dict] = []
        for page in range(1, max(1, pages) + 1):
            payload = self.client._request_public(
                "/api/v1/contract/funding_rate/history",
                params={"symbol": mexc_symbol, "page_num": page, "page_size": 100},
            )
            data = (payload or {}).get("data") if isinstance(payload, dict) else None
            items = (data or {}).get("resultList") if isinstance(data, dict) else None
            if not items:
                break
            rows.extend(items)
            if len(items) < 100:
                break
            if self.config.request_pause_sec:
                time.sleep(self.config.request_pause_sec)

        if not rows:
            return pd.DataFrame(columns=["time", "funding_rate"])

        df = pd.DataFrame(
            {
                "time": pd.to_numeric(pd.Series([r.get("settleTime") for r in rows]), errors="coerce") // 1000,
                "funding_rate": pd.to_numeric(pd.Series([r.get("fundingRate") for r in rows]), errors="coerce"),
            }
        ).dropna()
        return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
