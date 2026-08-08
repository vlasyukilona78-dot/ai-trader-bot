from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

# Non-crypto derivative products listed alongside perpetuals (equity/index proxies).
# They do not follow the alt/shitcoin pump-and-dump behavior this strategy targets.
DEFAULT_EXCLUDED_SYMBOLS: tuple[str, ...] = (
    "NAS100_USDT",
    "SP500_USDT",
    "DJ30_USDT",
    "MUSTOCK_USDT",
    "GOLD_USDT",
    "SILVER_USDT",
    "OIL_USDT",
)

# Tokenised equities and other TradFi proxies trade on a different clock and a
# different regime: in the labelled sample they resolved at 74% against 89% for
# the crypto universe, so leaving them in mixes two populations. Matched as
# substrings because the listings are named inconsistently.
DEFAULT_EXCLUDED_PATTERNS: tuple[str, ...] = (
    "STOCK", "EQUITY", "NASDAQ", "SP500", "DJ30", "NAS100",
    "GOLD", "SILVER", "OIL", "TSLA", "AAPL", "NVDA", "MSFT",
    "AMZN", "META", "GOOGL", "COIN", "MSTR", "SPX", "RDDT",
)


@dataclass
class UniverseConfig:
    """Selection rules for which contracts get scanned each cycle."""

    # Thin books carried a 4x higher share of runaway losers in the labelled
    # dataset, so the floor is set above the level where that risk showed up.
    min_turnover_24h_usdt: float = 400_000.0
    max_turnover_24h_usdt: float = 100_000_000.0
    quote: str = "USDT"
    refresh_sec: int = 300
    max_symbols: int = 0  # 0 = no cap
    min_change_24h: float | None = None  # e.g. 0.15 to only scan things already up 15%
    excluded_symbols: tuple[str, ...] = DEFAULT_EXCLUDED_SYMBOLS
    excluded_patterns: tuple[str, ...] = DEFAULT_EXCLUDED_PATTERNS
    # Roughly one in eight contracts has a minimum lot larger than a small
    # position, so signalling them would force oversizing. 0 disables the check.
    max_min_notional_usdt: float = 0.0


@dataclass
class UniverseEntry:
    symbol: str  # compact form used across the bot, e.g. CHILLGUYUSDT
    mexc_symbol: str  # exchange form, e.g. CHILLGUY_USDT
    turnover_24h_usdt: float
    change_24h: float
    funding_rate: float | None = None
    open_interest: float | None = None
    last_price: float | None = None
    min_notional_usdt: float | None = None
    max_leverage: float | None = None


@dataclass
class UniverseSnapshot:
    entries: list[UniverseEntry] = field(default_factory=list)
    total_contracts: int = 0
    # When the last successful snapshot became current. It anchors the refresh
    # TTL. Fresh snapshots use their response instant; stale fallbacks retain
    # the last successful anchor so the next cycle is still allowed to retry.
    # Source age remains a separate fact below.
    refreshed_at: float = 0.0
    request_started_at: float = 0.0
    received_at: float = 0.0
    # When the exchange actually produced these rows. On a cache hit this stays
    # at the original response instant, so cached data cannot be presented as a
    # fresh answer.
    source_ts: float | None = None
    cache_hit: bool = False
    cache_age_sec: float | None = None
    source_status: str = "ok"
    source_error_code: str | None = None
    # Contract details are a separate optional request; it has its own timing.
    details_request_started_at: float | None = None
    details_received_at: float | None = None
    details_status: str | None = None
    details_source_ts: float | None = None
    details_cache_hit: bool = False
    details_cache_age_sec: float | None = None
    details_error_code: str | None = None

    @property
    def symbols(self) -> list[str]:
        return [e.symbol for e in self.entries]


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None  # reject NaN


class SymbolUniverse:
    """Builds the per-cycle scan list from MEXC's full contract ticker feed.

    The strategy targets low/mid-cap alts inside a turnover band: below the band
    there is not enough liquidity to trade, above it the market-maker pump-and-dump
    pattern the strategy relies on is much rarer. Entries are ordered by 24h gain
    (most-pumped first), so capping via max_symbols keeps the best candidates.
    """

    def __init__(self, client, config: UniverseConfig | None = None):
        self.client = client
        self.config = config or UniverseConfig()
        self._snapshot = UniverseSnapshot()

    @property
    def snapshot(self) -> UniverseSnapshot:
        return self._snapshot

    @staticmethod
    def _safe_error_code(exc: BaseException, *, fallback: str) -> str:
        name = type(exc).__name__
        return name if name and name[0].isalpha() and name.replace("_", "").isalnum() else fallback

    @staticmethod
    def _safe_code_value(value: object, *, fallback: str) -> str:
        text = str(value or "")
        compact = text.replace("_", "").replace(".", "")
        if text and text[0].isalpha() and compact.isalnum() and len(text) <= 128:
            return text
        return fallback

    def _cached_snapshot(self, *, observed_at: float) -> UniverseSnapshot:
        """Expose a local snapshot reuse as a cache read, not a fresh response."""

        snapshot = self._snapshot
        source_ts = snapshot.source_ts
        cache_age = (
            max(0.0, observed_at - source_ts) if source_ts is not None else None
        )
        details_source_ts = snapshot.details_source_ts
        details_age = (
            max(0.0, observed_at - details_source_ts)
            if details_source_ts is not None
            else None
        )
        return replace(
            snapshot,
            request_started_at=observed_at,
            received_at=observed_at,
            cache_hit=source_ts is not None,
            cache_age_sec=cache_age,
            details_request_started_at=(
                observed_at if snapshot.details_request_started_at is not None else None
            ),
            details_received_at=(
                observed_at if snapshot.details_received_at is not None else None
            ),
            details_cache_hit=details_source_ts is not None,
            details_cache_age_sec=details_age,
        )

    def refresh(self, *, force: bool = False, now: float | None = None) -> UniverseSnapshot:
        now = now if now is not None else time.time()
        age = now - self._snapshot.refreshed_at
        if self._snapshot.entries and age < self.config.refresh_sec and not force:
            self._snapshot = self._cached_snapshot(observed_at=now)
            return self._snapshot

        # The response instant, not `now`. `now` was read before the request and
        # would date this data earlier than the process could possibly have held
        # it. Cache hits keep their own source instant so they are not laundered
        # into fresh responses.
        with_provenance = getattr(self.client, "fetch_all_tickers_with_provenance", None)
        provenance_supported = callable(with_provenance)
        if callable(with_provenance):
            tickers, provenance = with_provenance(force=force)
        else:
            started = time.time()
            try:
                tickers = self.client.fetch_all_tickers(force=force)
                legacy_error_code = None
            except Exception as exc:
                tickers = []
                legacy_error_code = self._safe_error_code(
                    exc, fallback="TickerRequestUnavailable"
                )
            answered = time.time()
            provenance = {
                "request_started_at": started,
                "received_at": answered,
                "source_ts": answered if legacy_error_code is None else None,
                "cache_hit": False,
                "cache_age_sec": 0.0 if legacy_error_code is None else None,
                "status": "ok" if legacy_error_code is None else "error",
                "error_code": legacy_error_code,
            }
        request_started_at = float(provenance["request_started_at"])
        received_at = float(provenance["received_at"])
        source_status = str(provenance.get("status") or "error")
        source_error_code = provenance.get("error_code")

        # The legacy rows-only surface cannot distinguish a failed empty response
        # from a real empty board. Treat it as unavailable; the real MEXC client
        # supplies provenance and can represent a successful empty response.
        if not tickers and not provenance_supported and source_status == "ok":
            source_status = "error"
            source_error_code = "LegacyTickerEmptyResponse"
            provenance["source_ts"] = None
            provenance["cache_age_sec"] = None

        if not tickers and source_status != "ok":
            if self._snapshot.entries and self._snapshot.source_ts is not None:
                # A failed attempt may use the prior derived universe, but its
                # source timestamp and age remain those of the prior response.
                # Contract specs reused with that derived universe are a cache
                # read too; do not leave their first-cycle provenance labelled
                # as fresh in this later fallback cycle.
                cached_snapshot = self._cached_snapshot(observed_at=received_at)
                source_ts = cached_snapshot.source_ts
                stale = replace(
                    cached_snapshot,
                    # Preserve the last successful TTL anchor so another cycle
                    # can retry instead of treating this failed attempt as a
                    # successful refresh for the next refresh interval.
                    refreshed_at=self._snapshot.refreshed_at,
                    request_started_at=request_started_at,
                    received_at=received_at,
                    source_ts=source_ts,
                    cache_hit=True,
                    cache_age_sec=max(0.0, request_started_at - source_ts),
                    source_status="stale_cache",
                    source_error_code=self._safe_code_value(
                        source_error_code, fallback="TickerRequestUnavailable"
                    ),
                )
                self._snapshot = stale
                return stale
            failed = UniverseSnapshot(
                entries=[],
                total_contracts=0,
                refreshed_at=now,
                request_started_at=request_started_at,
                received_at=received_at,
                source_ts=None,
                cache_hit=False,
                cache_age_sec=None,
                source_status="error",
                source_error_code=self._safe_code_value(
                    source_error_code, fallback="TickerRequestUnavailable"
                ),
            )
            self._snapshot = failed
            return failed

        cfg = self.config
        quote_suffix = f"_{cfg.quote.upper()}"
        excluded = {s.upper() for s in cfg.excluded_symbols}
        entries: list[UniverseEntry] = []

        details: dict[str, dict] = {}
        details_started: float | None = None
        details_received: float | None = None
        details_status: str | None = None
        details_source_ts: float | None = None
        details_cache_hit = False
        details_cache_age_sec: float | None = None
        details_error_code: str | None = None
        if cfg.max_min_notional_usdt > 0:
            # A second, independent request: it gets its own timing rather than
            # sharing the ticker's.
            details_with_provenance = getattr(
                self.client, "fetch_contract_details_with_provenance", None
            )
            if callable(details_with_provenance):
                try:
                    details, details_provenance = details_with_provenance(force=force)
                except Exception as exc:
                    details = {}
                    details_started = details_received = time.time()
                    details_status = "error"
                    details_error_code = self._safe_error_code(
                        exc, fallback="ContractDetailsUnavailable"
                    )
                else:
                    details_started = float(details_provenance["request_started_at"])
                    details_received = float(details_provenance["received_at"])
                    raw_details_source_ts = details_provenance.get("source_ts")
                    details_source_ts = (
                        float(raw_details_source_ts)
                        if raw_details_source_ts is not None
                        else None
                    )
                    details_cache_hit = bool(details_provenance.get("cache_hit"))
                    details_cache_age_sec = details_provenance.get("cache_age_sec")
                    details_status = str(details_provenance.get("status") or "error")
                    details_error_code = details_provenance.get("error_code")
            else:
                details_started = time.time()
                try:
                    details = self.client.fetch_contract_details(force=force)
                    details_status = "ok"
                except Exception as exc:
                    details = {}  # missing specs must not empty the scan list
                    details_status = "error"
                    details_error_code = self._safe_error_code(
                        exc, fallback="ContractDetailsUnavailable"
                    )
                details_received = time.time()
                if details_status == "ok":
                    details_source_ts = details_received
                    details_cache_age_sec = 0.0

        for item in tickers:
            if not isinstance(item, dict):
                continue
            mexc_symbol = str(item.get("symbol") or "").upper()
            if not mexc_symbol.endswith(quote_suffix) or mexc_symbol in excluded:
                continue
            base = mexc_symbol[: -len(quote_suffix)]
            if any(p in base for p in cfg.excluded_patterns):
                continue

            turnover = _as_float(item.get("amount24"))
            if turnover is None or not (cfg.min_turnover_24h_usdt <= turnover <= cfg.max_turnover_24h_usdt):
                continue

            change = _as_float(item.get("riseFallRate")) or 0.0
            if cfg.min_change_24h is not None and change < cfg.min_change_24h:
                continue

            last_price = _as_float(item.get("lastPrice"))
            min_notional = None
            max_lev = None
            spec = details.get(mexc_symbol)
            if spec and last_price:
                size = _as_float(spec.get("contractSize")) or 1.0
                min_vol = _as_float(spec.get("minVol")) or 1.0
                min_notional = size * min_vol * last_price
                max_lev = _as_float(spec.get("maxLeverage"))
                if cfg.max_min_notional_usdt > 0 and min_notional > cfg.max_min_notional_usdt:
                    continue

            entries.append(
                UniverseEntry(
                    symbol=mexc_symbol.replace("_", ""),
                    mexc_symbol=mexc_symbol,
                    turnover_24h_usdt=turnover,
                    change_24h=change,
                    funding_rate=_as_float(item.get("fundingRate")),
                    open_interest=_as_float(item.get("holdVol")),
                    last_price=last_price,
                    min_notional_usdt=min_notional,
                    max_leverage=max_lev,
                )
            )

        entries.sort(key=lambda e: e.change_24h, reverse=True)
        if cfg.max_symbols > 0:
            entries = entries[: cfg.max_symbols]

        self._snapshot = UniverseSnapshot(
            entries=entries,
            total_contracts=len(tickers),
            refreshed_at=(
                received_at
                if source_status == "ok"
                else float(provenance.get("source_ts") or 0.0)
            ),
            request_started_at=request_started_at,
            received_at=received_at,
            source_ts=(
                float(provenance["source_ts"])
                if provenance.get("source_ts") is not None
                else None
            ),
            cache_hit=bool(provenance.get("cache_hit")),
            cache_age_sec=provenance.get("cache_age_sec"),
            source_status=source_status,
            source_error_code=(
                self._safe_code_value(
                    source_error_code, fallback="TickerRequestUnavailable"
                )
                if source_error_code is not None
                else None
            ),
            details_request_started_at=details_started,
            details_received_at=details_received,
            details_status=details_status,
            details_source_ts=details_source_ts,
            details_cache_hit=details_cache_hit,
            details_cache_age_sec=details_cache_age_sec,
            details_error_code=(
                self._safe_code_value(
                    details_error_code, fallback="ContractDetailsUnavailable"
                )
                if details_error_code is not None
                else None
            ),
        )
        return self._snapshot

    def symbols(self, *, force: bool = False, now: float | None = None) -> list[str]:
        return self.refresh(force=force, now=now).symbols
