"""Versioned causal feature contract for MEXC pump-reversal research.

The registry is the source of truth shared by the runtime population journal,
future dataset builders and inference.  It deliberately distinguishes features
that are available in the current scanner from helpers that merely exist in an
offline module.  A helper being implemented is not the same thing as its value
being available point-in-time in the live decision path.

Only information known before the recorded decision is eligible. Bar-derived
values close no later than the scanner's candle cutoff; point-in-time universe
and context values use their own pre-decision observation semantics. Labels,
future returns, the strategy action and its legacy hand-written confidence are
excluded from this contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from functools import lru_cache
import hashlib
import json
import math
from typing import Any, Mapping


FEATURE_CONTRACT_VERSION = "mexc_reversal_features_v1"
_PINNED_CONTRACT_HASHES = {
    "mexc_reversal_features_v1": "bad45062961cac9638102a6e7a378fd64a93c4b6e025a135e3991698dab8b3d9",
}


class RuntimeStatus(str, Enum):
    """How (or whether) the current MEXC scanner produces a feature."""

    WIRED = "wired"
    CONDITIONAL = "conditional"
    OFFLINE_ONLY = "offline_only"
    SOURCE_MISSING = "source_missing"
    PLANNED = "planned"


class FeatureRole(str, Enum):
    """The boundary at which a value may be consumed."""

    MODEL = "model_candidate"
    PROPOSAL = "proposal_conditioning"
    POLICY = "deterministic_policy"
    CONTEXT = "context_candidate"
    DIAGNOSTIC = "diagnostic_only"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    layer: str
    role: FeatureRole
    source: str
    timing: str
    missing_policy: str
    runtime_status: RuntimeStatus
    path: tuple[str, ...] | None = None
    observed_path: tuple[str, ...] | None = None
    notes: str = ""

    @property
    def captured_now(self) -> bool:
        return self.path is not None and self.runtime_status in {
            RuntimeStatus.WIRED,
            RuntimeStatus.CONDITIONAL,
            RuntimeStatus.SOURCE_MISSING,
        }


def _spec(
    name: str,
    layer: str,
    role: FeatureRole,
    source: str,
    status: RuntimeStatus,
    path: str | None,
    *,
    timing: str = "closed_bar_cutoff",
    missing: str = "preserve_null_and_add_observed_flag",
    notes: str = "",
    observed_path: str | None = None,
) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        layer=layer,
        role=role,
        source=source,
        timing=timing,
        missing_policy=missing,
        runtime_status=status,
        path=tuple(path.split(".")) if path else None,
        observed_path=tuple(observed_path.split(".")) if observed_path else None,
        notes=notes,
    )


# Keep the order stable: it is part of the model schema.  Adding, removing or
# reordering captured features requires a new FEATURE_CONTRACT_VERSION.
FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    # Point-in-time population/universe context.
    _spec("turnover_24h_usdt", "universe", FeatureRole.MODEL, "MEXC ticker amount turnover", RuntimeStatus.WIRED, "universe.turnover_24h_usdt", timing="universe_snapshot_before_decision"),
    _spec("change_24h", "universe", FeatureRole.MODEL, "MEXC ticker", RuntimeStatus.WIRED, "universe.change_24h", timing="universe_snapshot_before_decision"),
    _spec("funding_rate", "universe", FeatureRole.MODEL, "MEXC contract ticker", RuntimeStatus.WIRED, "universe.funding_rate", timing="universe_snapshot_before_decision"),
    _spec("open_interest", "universe", FeatureRole.DIAGNOSTIC, "MEXC holdVol snapshot", RuntimeStatus.WIRED, "universe.open_interest", timing="universe_snapshot_before_decision", notes="Unknown cross-symbol units until contract-size normalization."),
    _spec("min_notional_usdt", "universe", FeatureRole.POLICY, "MEXC contract metadata", RuntimeStatus.WIRED, "universe.min_notional_usdt", timing="universe_snapshot_before_decision"),
    _spec("max_leverage", "universe", FeatureRole.POLICY, "MEXC contract metadata", RuntimeStatus.WIRED, "universe.max_leverage", timing="universe_snapshot_before_decision"),
    _spec("bar_count", "input", FeatureRole.DIAGNOSTIC, "closed OHLCV frame", RuntimeStatus.WIRED, "base.bar_count"),
    _spec("mark_price", "input", FeatureRole.DIAGNOSTIC, "last closed-bar close", RuntimeStatus.WIRED, "base.mark_price"),

    # Layer 1: recent pump event.
    _spec("pump_run_up_pct", "layer1_pump", FeatureRole.MODEL, "closed OHLCV window", RuntimeStatus.WIRED, "strategy.layer_trace.layers.layer1_pump_detection.details.run_up_pct"),
    _spec("pump_drop_pct", "layer1_pump", FeatureRole.MODEL, "closed OHLCV window", RuntimeStatus.WIRED, "strategy.layer_trace.layers.layer1_pump_detection.details.drop_pct"),
    _spec("pump_bars_since_peak", "layer1_pump", FeatureRole.MODEL, "closed OHLCV window", RuntimeStatus.WIRED, "strategy.layer_trace.layers.layer1_pump_detection.details.bars_since_peak"),
    _spec("pump_retrace_from_high", "layer1_pump", FeatureRole.MODEL, "closed OHLCV window", RuntimeStatus.WIRED, "strategy.layer_trace.layers.layer1_pump_detection.details.retrace_from_high"),
    _spec("pump_event_bars", "layer1_pump", FeatureRole.MODEL, "closed OHLCV indicators", RuntimeStatus.WIRED, "strategy.layer_trace.layers.layer1_pump_detection.details.pump_event_bars"),
    _spec("rsi_entry", "layer1_pump", FeatureRole.MODEL, "closed entry-timeframe OHLCV", RuntimeStatus.WIRED, "strategy.layer_trace.layers.layer1_pump_detection.details.rsi"),
    _spec("volume_spike", "layer1_pump", FeatureRole.MODEL, "closed entry-timeframe OHLCV", RuntimeStatus.WIRED, "strategy.layer_trace.layers.layer1_pump_detection.details.volume_spike"),

    # Layer 1b: volatility/liquidity quality.
    _spec("atr_pct", "layer1b_quality", FeatureRole.MODEL, "closed OHLCV indicators", RuntimeStatus.CONDITIONAL, "strategy.layer_trace.layers.layer1b_quality_gate.details.atr_pct"),
    _spec("turnover_recent_usdt", "layer1b_quality", FeatureRole.MODEL, "exact kline quote turnover", RuntimeStatus.CONDITIONAL, "strategy.layer_trace.layers.layer1b_quality_gate.details.usd_volume_recent", observed_path="strategy.layer_trace.layers.layer1b_quality_gate.details.exact_turnover_available"),
    _spec("atr_floor", "layer1b_quality", FeatureRole.DIAGNOSTIC, "frozen cross-sectional sweep", RuntimeStatus.CONDITIONAL, "strategy.layer_trace.layers.layer1b_quality_gate.details.min_atr_pct"),

    # Layer 1c: market and higher-timeframe context.
    _spec("relative_strength_btc", "layer1c_market", FeatureRole.MODEL, "closed BTC and symbol bars", RuntimeStatus.CONDITIONAL, "strategy.layer_trace.layers.layer1c_market_context.details.relative_strength"),
    _spec("rsi_4h", "layer1c_market", FeatureRole.MODEL, "point-in-time closed 4h cache", RuntimeStatus.CONDITIONAL, "strategy.layer_trace.layers.layer1c_market_context.details.rsi_htf"),
    _spec("chase_atr", "layer1c_market", FeatureRole.MODEL, "closed OHLCV indicators", RuntimeStatus.CONDITIONAL, "strategy.layer_trace.layers.layer1c_market_context.details.chase_atr"),
    _spec("overhead_level_distance", "layer1c_market", FeatureRole.MODEL, "causal horizontal levels", RuntimeStatus.CONDITIONAL, "strategy.layer_trace.layers.layer1c_market_context.details.level_dist", observed_path="strategy.layer_trace.layers.layer1c_market_context.details.level_available"),

    # Layer 2: weakness.  The implementation exists but is disabled by default;
    # absence must remain distinct from a measured zero.
    _spec("weak_price_up", "layer2_weakness", FeatureRole.MODEL, "closed OHLCV", RuntimeStatus.CONDITIONAL, "strategy.layer_trace.layers.layer2_weakness_confirmation.details.price_up"),
    _spec("weak_obv_down", "layer2_weakness", FeatureRole.MODEL, "closed OHLCV-derived OBV", RuntimeStatus.CONDITIONAL, "strategy.layer_trace.layers.layer2_weakness_confirmation.details.obv_down"),
    _spec("weak_cvd_down", "layer2_weakness", FeatureRole.MODEL, "closed OHLCV-derived CVD proxy", RuntimeStatus.CONDITIONAL, "strategy.layer_trace.layers.layer2_weakness_confirmation.details.cvd_down", notes="Proxy, not an aggressor-side trade feed."),

    # Layer 3: entry location and market-structure break.
    _spec("distance_from_extreme_pct", "layer3_location", FeatureRole.MODEL, "closed OHLCV plus closed 4h anchor", RuntimeStatus.CONDITIONAL, "strategy.layer_trace.layers.layer3_entry_location.details.dist_from_extreme_pct"),
    _spec("msb_down_recent", "layer3_location", FeatureRole.MODEL, "closed OHLCV", RuntimeStatus.CONDITIONAL, "strategy.layer_trace.layers.layer3_entry_location.details.msb_down_recent"),
    _spec("msb_struct_break_down", "layer3_location", FeatureRole.MODEL, "closed OHLCV", RuntimeStatus.CONDITIONAL, "strategy.layer_trace.layers.layer3_entry_location.details.msb_struct_break_down"),
    _spec("msb_ema_cross_down", "layer3_location", FeatureRole.MODEL, "closed OHLCV", RuntimeStatus.CONDITIONAL, "strategy.layer_trace.layers.layer3_entry_location.details.msb_ema_cross_down"),
    _spec("poc", "layer3_location", FeatureRole.DIAGNOSTIC, "causal volume profile", RuntimeStatus.CONDITIONAL, "strategy.layer_trace.layers.layer3_entry_location.details.poc", observed_path="strategy.layer_trace.layers.layer3_entry_location.details.vp_levels_available", notes="Raw cross-symbol price; use normalized distance for a model."),
    _spec("vah", "layer3_location", FeatureRole.DIAGNOSTIC, "causal volume profile", RuntimeStatus.CONDITIONAL, "strategy.layer_trace.layers.layer3_entry_location.details.vah", observed_path="strategy.layer_trace.layers.layer3_entry_location.details.vp_levels_available", notes="Raw cross-symbol price; use normalized distance for a model."),
    _spec("val", "layer3_location", FeatureRole.DIAGNOSTIC, "causal volume profile", RuntimeStatus.CONDITIONAL, "strategy.layer_trace.layers.layer3_entry_location.details.val", observed_path="strategy.layer_trace.layers.layer3_entry_location.details.vp_levels_available", notes="Raw cross-symbol price; use normalized distance for a model."),

    # Layer 4: crowd/context. Funding is now passed from the same frozen universe
    # snapshot; long/short ratio still has no MEXC runtime source.
    _spec("sentiment_index", "layer4_context", FeatureRole.CONTEXT, "configured point-in-time sentiment feed or explicit fallback", RuntimeStatus.CONDITIONAL, "strategy.layer_trace.layers.layer4_fake_filter.details.sentiment", timing="context_snapshot_before_decision", observed_path="strategy.layer_trace.layers.layer4_fake_filter.details.sentiment_available"),
    _spec("sentiment_degraded", "layer4_context", FeatureRole.DIAGNOSTIC, "sentiment provenance", RuntimeStatus.CONDITIONAL, "strategy.layer_trace.layers.layer4_fake_filter.details.degraded_mode", timing="context_snapshot_before_decision"),
    _spec("layer4_funding_rate", "layer4_context", FeatureRole.DIAGNOSTIC, "frozen MEXC universe snapshot", RuntimeStatus.CONDITIONAL, "strategy.layer_trace.layers.layer4_fake_filter.details.funding_rate", timing="universe_snapshot_before_decision", observed_path="strategy.layer_trace.layers.layer4_fake_filter.details.funding_available", notes="Duplicate diagnostic; the canonical model feature is universe funding_rate."),
    _spec("layer4_funding_observed", "layer4_context", FeatureRole.DIAGNOSTIC, "funding provenance", RuntimeStatus.CONDITIONAL, "strategy.layer_trace.layers.layer4_fake_filter.details.funding_available"),
    _spec("long_short_ratio", "layer4_context", FeatureRole.MODEL, "no MEXC runtime source wired", RuntimeStatus.SOURCE_MISSING, "strategy.layer_trace.layers.layer4_fake_filter.details.long_short_ratio", timing="context_snapshot_before_decision", observed_path="strategy.layer_trace.layers.layer4_fake_filter.details.long_short_ratio_available"),
    _spec("long_short_ratio_observed", "layer4_context", FeatureRole.DIAGNOSTIC, "ratio provenance", RuntimeStatus.SOURCE_MISSING, "strategy.layer_trace.layers.layer4_fake_filter.details.long_short_ratio_available", timing="context_snapshot_before_decision"),

    # Layer 5 values are causal, but belong to net-EV/policy evaluation rather
    # than an unconstrained direction predictor.
    _spec("stop_distance_pct", "layer5_contract", FeatureRole.PROPOSAL, "structural stop", RuntimeStatus.CONDITIONAL, "strategy.layer_trace.layers.layer5_tp_sl.details.stop_distance_pct"),
    _spec("realized_risk_reward", "layer5_contract", FeatureRole.PROPOSAL, "structural stop/target", RuntimeStatus.CONDITIONAL, "strategy.layer_trace.layers.layer5_tp_sl.details.realized_risk_reward"),
    _spec("max_safe_leverage", "layer5_contract", FeatureRole.POLICY, "structural stop distance", RuntimeStatus.CONDITIONAL, "strategy.layer_trace.layers.layer5_tp_sl.details.max_safe_leverage"),

    # Implemented elsewhere, but not yet consumed by the runtime decision path.
    _spec("rsi_1h", "intended_mtf", FeatureRole.MODEL, "closed 1h frame", RuntimeStatus.PLANNED, None),
    _spec("fib_500_distance", "intended_levels", FeatureRole.MODEL, "causal Fibonacci grid", RuntimeStatus.OFFLINE_ONLY, None),
    _spec("fib_618_distance", "intended_levels", FeatureRole.MODEL, "causal Fibonacci grid", RuntimeStatus.OFFLINE_ONLY, None),
    _spec("confluence_count", "intended_levels", FeatureRole.MODEL, "closed 15m/1h/4h levels", RuntimeStatus.OFFLINE_ONLY, None),
    _spec("confluence_strength", "intended_levels", FeatureRole.MODEL, "closed 15m/1h/4h levels", RuntimeStatus.OFFLINE_ONLY, None),
    _spec("estimated_liquidation_density_below", "intended_liquidation", FeatureRole.CONTEXT, "OHLCV leverage proxy", RuntimeStatus.OFFLINE_ONLY, None, notes="Must be tagged as an estimate, never represented as exchange liquidation data."),
    _spec("open_interest_change", "intended_derivatives", FeatureRole.MODEL, "point-in-time OI history", RuntimeStatus.PLANNED, None),
    _spec("orderbook_imbalance", "intended_microstructure", FeatureRole.MODEL, "timestamped depth snapshots", RuntimeStatus.SOURCE_MISSING, None),
    _spec("trade_aggressor_imbalance", "intended_microstructure", FeatureRole.MODEL, "timestamped public trades", RuntimeStatus.SOURCE_MISSING, None),
    _spec("news_context_score", "intended_context", FeatureRole.CONTEXT, "LLM-extracted public events with publication timestamps", RuntimeStatus.PLANNED, None),
    _spec("open_interest_notional_usdt", "intended_derivatives", FeatureRole.MODEL, "contract-size-normalized OI", RuntimeStatus.PLANNED, None),
    _spec("poc_distance_pct", "intended_levels", FeatureRole.MODEL, "price-normalized volume profile", RuntimeStatus.PLANNED, None),
    _spec("vah_distance_pct", "intended_levels", FeatureRole.MODEL, "price-normalized volume profile", RuntimeStatus.PLANNED, None),
    _spec("val_distance_pct", "intended_levels", FeatureRole.MODEL, "price-normalized volume profile", RuntimeStatus.PLANNED, None),
)


def captured_feature_specs() -> tuple[FeatureSpec, ...]:
    return tuple(spec for spec in FEATURE_SPECS if spec.captured_now)


def model_feature_specs() -> tuple[FeatureSpec, ...]:
    allowed = {FeatureRole.MODEL, FeatureRole.PROPOSAL, FeatureRole.CONTEXT}
    return tuple(spec for spec in captured_feature_specs() if spec.role in allowed)


def model_feature_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in model_feature_specs())


@lru_cache(maxsize=1)
def feature_contract_hash() -> str:
    """Hash only the executable captured schema, not roadmap prose."""

    features = []
    for spec in captured_feature_specs():
        features.append(
            {
                "name": spec.name,
                "layer": spec.layer,
                "role": spec.role.value,
                "timing": spec.timing,
                "missing_policy": spec.missing_policy,
                "runtime_status": spec.runtime_status.value,
                "path": list(spec.path) if spec.path is not None else None,
                "observed_path": list(spec.observed_path) if spec.observed_path is not None else None,
            }
        )
    payload = {
        "snapshot_schema": {
            "version": 1,
            "source_times": ("bar_cutoff_ts", "universe_refreshed_at"),
            "missing_representation": "nullable_value_plus_observed_bit_and_reason",
        },
        "features": features,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    expected = _PINNED_CONTRACT_HASHES.get(FEATURE_CONTRACT_VERSION)
    if expected != digest:
        raise RuntimeError("feature_contract_changed_without_version_bump")
    return digest


@lru_cache(maxsize=1)
def feature_registry_hash() -> str:
    """Hash the full implementation/roadmap registry for audit documentation."""

    payload = []
    for spec in FEATURE_SPECS:
        row = asdict(spec)
        row["role"] = spec.role.value
        row["runtime_status"] = spec.runtime_status.value
        row["path"] = list(spec.path) if spec.path is not None else None
        row["observed_path"] = list(spec.observed_path) if spec.observed_path is not None else None
        payload.append(row)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def configuration_hash(value: Any, *, component: str) -> str:
    """Return a stable hash for a dataclass/mapping runtime configuration."""

    payload = asdict(value) if is_dataclass(value) else value
    if payload is None:
        payload = {"unavailable_component": component}
    encoded = json.dumps(
        {"component": component, "config": payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _path_value(root: Mapping[str, Any], path: tuple[str, ...]) -> tuple[bool, Any]:
    current: Any = root
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return False, None
        current = current[key]
    return True, current


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def build_runtime_feature_snapshot(
    metadata: Mapping[str, Any],
    *,
    bar_cutoff_ts: float,
    universe_refreshed_at: float,
) -> dict[str, Any]:
    """Extract the current causal values without inventing missing observations."""

    values: dict[str, float | None] = {}
    observed: dict[str, int] = {}
    availability_reason: dict[str, str] = {}
    missing: list[str] = []
    for spec in captured_feature_specs():
        assert spec.path is not None
        present, raw = _path_value(metadata, spec.path)
        value = _numeric(raw) if present else None
        if spec.observed_path is not None:
            provenance_present, provenance_raw = _path_value(metadata, spec.observed_path)
            provenance = _numeric(provenance_raw) if provenance_present else None
            is_observed = value is not None and provenance is not None and provenance > 0.0
        else:
            is_observed = value is not None
        values[spec.name] = value
        observed[spec.name] = 1 if is_observed else 0
        if is_observed:
            availability_reason[spec.name] = "observed"
        elif not present:
            availability_reason[spec.name] = "not_computed"
        elif raw is None or value is None:
            availability_reason[spec.name] = "source_unavailable_or_invalid"
        elif spec.observed_path is not None:
            availability_reason[spec.name] = "provenance_unavailable"
        else:
            availability_reason[spec.name] = "unobserved"
        if not is_observed:
            missing.append(spec.name)

    expected = len(values)
    present_count = expected - len(missing)
    return {
        "contract_version": FEATURE_CONTRACT_VERSION,
        "contract_hash": feature_contract_hash(),
        "source_times": {
            "bar_cutoff_ts": _numeric(bar_cutoff_ts),
            "universe_refreshed_at": _numeric(universe_refreshed_at),
        },
        "values": values,
        "observed": observed,
        "availability_reason": availability_reason,
        "missing": missing,
        "coverage": {
            "present": present_count,
            "expected": expected,
            "fraction": (present_count / expected) if expected else 0.0,
        },
    }
