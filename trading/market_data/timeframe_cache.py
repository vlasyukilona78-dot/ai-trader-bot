from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass

import pandas as pd
import requests

from trading.market_data.bar_contract import (
    BarContractError,
    closed_boundary_ts,
    retain_closed_bars,
)
from trading.market_data.frame_provenance import (
    FrameRead,
    FrameQualityError,
    SourceReadEvidenceV1,
    canonical_frame_timeframe,
)
from trading.market_data.mexc_client import MexcOhlcvError


@dataclass
class TimeframeCacheConfig:
    interval: str = "Hour4"
    candles: int = 120
    ttl_sec: float = 1800.0
    max_symbols: int = 500


_SAFE_ERROR_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.]{0,127}$")


class HigherTimeframeCache:
    """Per-symbol/per-timeframe/window closed bars keyed by causal boundary.

    TTL controls request pacing inside one bar. Crossing a candle boundary always
    requires a new fetch. If that fetch fails, the previous boundary is never
    served as though it belonged to the current decision time.

    ``get_with_provenance`` is the authoritative API. ``get`` remains a rows-only
    compatibility wrapper for the existing strategy while the scanner is moved
    to explicit per-symbol evidence.
    """

    def __init__(self, feed, config: TimeframeCacheConfig | None = None):
        self.feed = feed
        self.config = config or TimeframeCacheConfig()
        self._frames: dict[tuple[str, str, int], pd.DataFrame] = {}
        self._evidence: dict[tuple[str, str, int], SourceReadEvidenceV1] = {}
        self._fetched_at: dict[tuple[str, str, int], float] = {}
        self._boundary_at: dict[tuple[str, str, int], float] = {}
        self._lock = threading.Lock()
        # A dedicated lock prevents duplicate same-key misses without serializing
        # independent symbols. Locks intentionally outlive eviction: removing a
        # lock while a waiter still owns its object could create two flights for
        # the same key.
        self._flight_locks: dict[tuple[str, str, int], threading.Lock] = {}
        # Legacy cycle aggregation can continue during the staged migration. The
        # exact evidence is returned directly and these spans now at least retain
        # symbol/timeframe/outcome instead of becoming anonymous.
        self._timings: list[dict[str, object]] = []

    def drain_timings(self) -> list[dict[str, object]]:
        """Take and clear the compatibility spans recorded since the last drain."""

        with self._lock:
            drained = list(self._timings)
            self._timings.clear()
        return drained

    @staticmethod
    def _safe_error_code(exc: BaseException) -> str:
        code = type(exc).__name__
        return code if _SAFE_ERROR_RE.fullmatch(code) else "UnknownError"

    def _source_identity(self, symbol: str) -> tuple[str, str]:
        client = getattr(self.feed, "_client", None)
        venue = str(
            getattr(self.feed, "venue", None)
            or getattr(client, "venue", None)
            or "public_market"
        )
        normalize = getattr(client, "normalize_symbol", None)
        venue_symbol = normalize(symbol) if callable(normalize) else str(symbol).upper()
        return venue, str(venue_symbol).upper()

    @staticmethod
    def _epoch(value: object) -> float:
        if isinstance(value, bool):
            raise ValueError("decision time must be numeric or timestamp-like")
        if isinstance(value, (int, float)):
            return float(value)
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            raise ValueError("decision time must be timezone-aware")
        return float(timestamp.tz_convert("UTC").timestamp())

    @staticmethod
    def _copy(frame: pd.DataFrame) -> pd.DataFrame:
        copied = frame.copy(deep=True)
        copied.attrs.update(frame.attrs)
        return copied

    def _record_timing(self, evidence: SourceReadEvidenceV1) -> None:
        with self._lock:
            if len(self._timings) < 5_000:
                self._timings.append(self._timing_dict(evidence))

    def _flight_lock(self, key: tuple[str, str, int]) -> threading.Lock:
        with self._lock:
            lock = self._flight_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._flight_locks[key] = lock
            return lock

    def _cache_hit(
        self,
        *,
        key: tuple[str, str, int],
        boundary: float,
        requested_as_of_ts: float,
        request_started_at: float,
        cache_now: float,
        ttl_sec: float,
    ) -> FrameRead | None:
        with self._lock:
            cached = self._frames.get(key)
            cached_evidence = self._evidence.get(key)
            cached_boundary = self._boundary_at.get(key)
            age = cache_now - self._fetched_at.get(key, 0.0)
            if (
                cached is None
                or cached_evidence is None
                or cached_boundary != boundary
                or not (0.0 <= age < ttl_sec)
            ):
                return None
            hit = self._copy(cached)
            received_at = time.time()
            source_ts = cached_evidence.source_ts or cached_evidence.received_at
            assert source_ts is not None
            hit_evidence = cached_evidence.with_cache_read(
                requested_as_of_ts=requested_as_of_ts,
                request_started_at=request_started_at,
                received_at=received_at,
                source_ts=source_ts,
                cache_age_sec=max(0.0, request_started_at - source_ts),
            )
            read = FrameRead(frame=hit, evidence=hit_evidence)
            if len(self._timings) < 5_000:
                self._timings.append(self._timing_dict(read.evidence))
            return read

    @staticmethod
    def _quality_rank(evidence: SourceReadEvidenceV1) -> tuple[float, int, int]:
        """Rank same-boundary coverage so a late read cannot degrade the cache."""

        data_through = (
            float(evidence.data_through_ts)
            if evidence.data_through_ts is not None
            else float("-inf")
        )
        return (
            data_through,
            int(evidence.bar_count),
            1 if evidence.outcome == "fresh" else 0,
        )

    def _fresh_read(
        self,
        *,
        symbol: str,
        timeframe: str,
        requested_as_of_ts: float,
        expected_boundary: float,
        started_at: float,
        candles: int,
    ) -> FrameRead:
        fetch_closed = getattr(self.feed, "fetch_closed_frame", None)
        if callable(fetch_closed):
            fetched = fetch_closed(
                symbol=symbol,
                timeframe=timeframe,
                candles=candles,
                as_of=expected_boundary,
            ).ohlcv
        else:
            fetched = self.feed.fetch_frame(
                symbol=symbol,
                timeframe=timeframe,
                candles=candles + 1,
            ).ohlcv
        frame = retain_closed_bars(
            fetched,
            interval=timeframe,
            as_of=expected_boundary,
        ).tail(candles)
        frame = self._copy(frame)
        received_at = time.time()
        venue, venue_symbol = self._source_identity(symbol)
        evidence = SourceReadEvidenceV1.from_frame(
            frame,
            source="higher_timeframe_ohlcv",
            venue=venue,
            symbol=str(symbol).upper(),
            venue_symbol=venue_symbol,
            timeframe=timeframe,
            requested_as_of_ts=requested_as_of_ts,
            request_started_at=started_at,
            received_at=received_at,
            source_ts=received_at,
            cache_hit=False,
            cache_age_sec=0.0,
        )
        return FrameRead(frame=frame, evidence=evidence)

    def get_with_provenance(
        self,
        symbol: str,
        *,
        as_of=None,
        now: float | None = None,
    ) -> FrameRead:
        """Return a deep-owned frame plus exact evidence for this one call."""

        observed_at = time.time()
        cache_now = float(now if now is not None else observed_at)
        decision_time = as_of if as_of is not None else observed_at
        requested_as_of_ts = self._epoch(decision_time)
        # Snapshot mutable config once. Content-bearing settings are part of the
        # key, so an in-flight config mutation cannot return the wrong window.
        timeframe = canonical_frame_timeframe(self.config.interval)
        candles = int(self.config.candles)
        ttl_sec = float(self.config.ttl_sec)
        max_symbols = int(self.config.max_symbols)
        if candles <= 0:
            raise ValueError("higher-timeframe candles must be positive")
        if ttl_sec < 0.0:
            raise ValueError("higher-timeframe ttl must not be negative")
        if max_symbols <= 0:
            raise ValueError("higher-timeframe max_symbols must be positive")
        boundary = float(closed_boundary_ts(decision_time, timeframe))
        key = (str(symbol).upper(), timeframe, candles)

        started_at = time.time()
        hit = self._cache_hit(
            key=key,
            boundary=boundary,
            requested_as_of_ts=requested_as_of_ts,
            request_started_at=started_at,
            cache_now=cache_now,
            ttl_sec=ttl_sec,
        )
        if hit is not None:
            return hit

        # Only one request per exact content key may proceed at a time. A waiter
        # rechecks after acquiring the lock and consumes the completed flight.
        with self._flight_lock(key):
            started_at = time.time()
            hit = self._cache_hit(
                key=key,
                boundary=boundary,
                requested_as_of_ts=requested_as_of_ts,
                request_started_at=started_at,
                cache_now=cache_now,
                ttl_sec=ttl_sec,
            )
            if hit is not None:
                return hit

            try:
                read = self._fresh_read(
                    symbol=str(symbol).upper(),
                    timeframe=timeframe,
                    requested_as_of_ts=requested_as_of_ts,
                    expected_boundary=boundary,
                    started_at=started_at,
                    candles=candles,
                )
            except (
                BarContractError,
                FrameQualityError,
                MexcOhlcvError,
                requests.RequestException,
            ) as exc:
                received_at = time.time()
                error_code = self._safe_error_code(exc)
                with self._lock:
                    cached = self._frames.get(key)
                    cached_evidence = self._evidence.get(key)
                    if (
                        cached is not None
                        and cached_evidence is not None
                        and cached_evidence.bar_count > 0
                        and self._boundary_at.get(key) == boundary
                    ):
                        fallback = self._copy(cached)
                    else:
                        cached_evidence = None
                        fallback = None
                if fallback is not None and cached_evidence is not None:
                    source_ts = cached_evidence.source_ts or cached_evidence.received_at
                    assert source_ts is not None
                    evidence = cached_evidence.with_cache_read(
                        requested_as_of_ts=requested_as_of_ts,
                        request_started_at=started_at,
                        received_at=received_at,
                        source_ts=source_ts,
                        cache_age_sec=max(0.0, started_at - source_ts),
                        refresh_error_code=error_code,
                    )
                    read = FrameRead(frame=fallback, evidence=evidence)
                else:
                    venue, venue_symbol = self._source_identity(str(symbol).upper())
                    evidence = SourceReadEvidenceV1.request_failed(
                        source="higher_timeframe_ohlcv",
                        venue=venue,
                        symbol=str(symbol).upper(),
                        venue_symbol=venue_symbol,
                        timeframe=timeframe,
                        requested_as_of_ts=requested_as_of_ts,
                        request_started_at=started_at,
                        received_at=received_at,
                        error_code=error_code,
                    )
                    read = FrameRead(frame=None, evidence=evidence)
                self._record_timing(read.evidence)
                return read

            # A same-boundary response with less coverage cannot replace a
            # stronger cached frame (for example, late stale after fresh). Use
            # the stronger frame but explicitly mark the degraded refresh.
            with self._lock:
                existing = self._frames.get(key)
                existing_evidence = self._evidence.get(key)
                existing_boundary = self._boundary_at.get(key)
                preserve_existing = bool(
                    existing is not None
                    and existing_evidence is not None
                    and existing_boundary == boundary
                    and self._quality_rank(existing_evidence)
                    > self._quality_rank(read.evidence)
                )
                if preserve_existing:
                    fallback = self._copy(existing)
                else:
                    fallback = None

                if not preserve_existing:
                    if existing_boundary is not None and existing_boundary > boundary:
                        returned = FrameRead(
                            frame=self._copy(read.frame) if read.frame is not None else None,
                            evidence=read.evidence,
                        )
                    else:
                        if len(self._frames) >= max_symbols and key not in self._frames:
                            oldest = min(
                                self._fetched_at,
                                key=self._fetched_at.get,
                                default=None,
                            )
                            if oldest is not None:
                                self._frames.pop(oldest, None)
                                self._evidence.pop(oldest, None)
                                self._fetched_at.pop(oldest, None)
                                self._boundary_at.pop(oldest, None)
                        assert read.frame is not None
                        self._frames[key] = self._copy(read.frame)
                        self._evidence[key] = read.evidence
                        self._fetched_at[key] = cache_now
                        self._boundary_at[key] = boundary
                        returned = FrameRead(
                            frame=self._copy(read.frame), evidence=read.evidence
                        )

            if preserve_existing:
                assert fallback is not None and existing_evidence is not None
                source_ts = existing_evidence.source_ts or existing_evidence.received_at
                assert source_ts is not None
                received_at = read.evidence.received_at
                assert received_at is not None
                refresh_code = (
                    "HigherTimeframeNoRows"
                    if read.evidence.outcome == "no_rows"
                    else "HigherTimeframeDataLag"
                )
                evidence = existing_evidence.with_cache_read(
                    requested_as_of_ts=requested_as_of_ts,
                    request_started_at=started_at,
                    received_at=received_at,
                    source_ts=source_ts,
                    cache_age_sec=max(0.0, started_at - source_ts),
                    refresh_error_code=refresh_code,
                )
                returned = FrameRead(frame=fallback, evidence=evidence)

            self._record_timing(returned.evidence)
            return returned

    @staticmethod
    def _timing_dict(evidence: SourceReadEvidenceV1) -> dict[str, object]:
        if evidence.outcome == "request_failed":
            status, error_code = "error", evidence.error_code
        elif evidence.outcome == "no_rows":
            status, error_code = "error", "HigherTimeframeNoRows"
        elif evidence.outcome == "stale" and evidence.error_code is not None:
            status, error_code = "stale_cache", evidence.error_code
        else:
            status, error_code = "ok", None
        return {
            "symbol": evidence.symbol,
            "timeframe": evidence.timeframe,
            "request_started_at": evidence.request_started_at,
            "received_at": evidence.received_at,
            "source_as_of": evidence.data_through_ts,
            "requested_as_of_ts": evidence.requested_as_of_ts,
            "expected_closed_boundary_ts": evidence.expected_closed_boundary_ts,
            "data_through_ts": evidence.data_through_ts,
            "status": status,
            "outcome": evidence.outcome,
            "cache_hit": evidence.cache_hit,
            "source_ts": evidence.source_ts,
            "cache_age_sec": evidence.cache_age_sec,
            "error_code": error_code,
            "frame_hash": evidence.frame_hash,
        }

    def get(
        self,
        symbol: str,
        *,
        as_of=None,
        now: float | None = None,
    ) -> pd.DataFrame | None:
        """Rows-only compatibility wrapper around ``get_with_provenance``.

        This deliberately preserves the legacy behavior of returning a lagging
        or refresh-fallback frame. Causal decision code must migrate to
        ``get_with_provenance`` and inspect ``evidence.outcome`` explicitly.
        """

        return self.get_with_provenance(symbol, as_of=as_of, now=now).frame

    @property
    def cached_symbols(self) -> int:
        return len(self._frames)
