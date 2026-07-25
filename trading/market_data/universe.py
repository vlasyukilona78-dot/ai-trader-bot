from __future__ import annotations

import time
from dataclasses import dataclass, field
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


@dataclass
class UniverseConfig:
    """Selection rules for which contracts get scanned each cycle."""

    min_turnover_24h_usdt: float = 200_000.0
    max_turnover_24h_usdt: float = 100_000_000.0
    quote: str = "USDT"
    refresh_sec: int = 300
    max_symbols: int = 0  # 0 = no cap
    min_change_24h: float | None = None  # e.g. 0.15 to only scan things already up 15%
    excluded_symbols: tuple[str, ...] = DEFAULT_EXCLUDED_SYMBOLS


@dataclass
class UniverseEntry:
    symbol: str  # compact form used across the bot, e.g. CHILLGUYUSDT
    mexc_symbol: str  # exchange form, e.g. CHILLGUY_USDT
    turnover_24h_usdt: float
    change_24h: float
    funding_rate: float | None = None
    open_interest: float | None = None
    last_price: float | None = None


@dataclass
class UniverseSnapshot:
    entries: list[UniverseEntry] = field(default_factory=list)
    total_contracts: int = 0
    refreshed_at: float = 0.0

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

    def refresh(self, *, force: bool = False, now: float | None = None) -> UniverseSnapshot:
        now = now if now is not None else time.time()
        age = now - self._snapshot.refreshed_at
        if self._snapshot.entries and age < self.config.refresh_sec and not force:
            return self._snapshot

        tickers = self.client.fetch_all_tickers(force=force)
        if not tickers:
            # Keep serving the previous snapshot rather than emptying the scan list.
            return self._snapshot

        cfg = self.config
        quote_suffix = f"_{cfg.quote.upper()}"
        excluded = {s.upper() for s in cfg.excluded_symbols}
        entries: list[UniverseEntry] = []

        for item in tickers:
            if not isinstance(item, dict):
                continue
            mexc_symbol = str(item.get("symbol") or "").upper()
            if not mexc_symbol.endswith(quote_suffix) or mexc_symbol in excluded:
                continue

            turnover = _as_float(item.get("amount24"))
            if turnover is None or not (cfg.min_turnover_24h_usdt <= turnover <= cfg.max_turnover_24h_usdt):
                continue

            change = _as_float(item.get("riseFallRate")) or 0.0
            if cfg.min_change_24h is not None and change < cfg.min_change_24h:
                continue

            entries.append(
                UniverseEntry(
                    symbol=mexc_symbol.replace("_", ""),
                    mexc_symbol=mexc_symbol,
                    turnover_24h_usdt=turnover,
                    change_24h=change,
                    funding_rate=_as_float(item.get("fundingRate")),
                    open_interest=_as_float(item.get("holdVol")),
                    last_price=_as_float(item.get("lastPrice")),
                )
            )

        entries.sort(key=lambda e: e.change_24h, reverse=True)
        if cfg.max_symbols > 0:
            entries = entries[: cfg.max_symbols]

        self._snapshot = UniverseSnapshot(
            entries=entries,
            total_contracts=len(tickers),
            refreshed_at=now,
        )
        return self._snapshot

    def symbols(self, *, force: bool = False, now: float | None = None) -> list[str]:
        return self.refresh(force=force, now=now).symbols
