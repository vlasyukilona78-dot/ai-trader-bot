"""Strict bridge from the runtime population journal to model feature rows.

This module does not create labels and does not train a model.  It exists to
make the first half of the causal path executable without silently falling back
to the historical event-conditioned CSV builders.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterator, Mapping

from trading.market_data.bar_contract import interval_seconds
from trading.metrics.cycle_envelope import CycleEnvelope, CycleEnvelopeError
from trading.market_data.source_timing import SourceTiming, SourceTimingError
from trading.metrics.population_journal import (
    RECORD_TYPE_DECISION,
    RECORD_TYPE_FOOTER,
    RECORD_TYPE_HEADER,
    SCHEMA_VERSION,
    PopulationDecision,
    PopulationJournalError,
    make_cycle_id,
    rows_checksum,
)

from .feature_contract import (
    FEATURE_CONTRACT_VERSION,
    build_runtime_feature_snapshot,
    captured_feature_specs,
    feature_contract_hash,
    model_feature_names,
)


class PopulationDatasetError(ValueError):
    """Raised when a population journal cannot support reproducible research."""


_MAX_JOURNAL_LINE_CHARS = 1_000_000


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non_standard_json_constant:{value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate_json_key:{key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class PopulationFeatureRow:
    schema_version: int
    snapshot_id: str
    cycle_id: str
    cycle_ordinal: int
    cycle_size: int
    symbol: str
    timeframe: str
    status: str
    action: str
    bar_cutoff_ts: float
    decision_completed_ts: float
    universe_refreshed_at: float
    universe_received_at: float
    ranking_ready_ts: float
    cycle_completed_ts: float
    actionable_ts: float
    entry_eligible_ts: float
    entry_bar_open_ts: float
    strategy_config_hash: str
    universe_policy_hash: str
    feature_contract_version: str
    feature_contract_hash: str
    features: Mapping[str, float | None]
    observed: Mapping[str, int]
    availability_reason: Mapping[str, str]


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PopulationDatasetError(f"{field}_must_be_numeric")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise PopulationDatasetError(f"{field}_must_be_finite") from exc
    if not math.isfinite(number):
        raise PopulationDatasetError(f"{field}_must_be_finite")
    return number


def _integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PopulationDatasetError(f"{field}_must_be_integer")
    return int(value)


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PopulationDatasetError(f"{field}_must_be_mapping")
    return value


def _parse_feature_row(payload: Mapping[str, Any]) -> PopulationFeatureRow:
    metadata = _mapping(payload.get("metadata"), field="metadata")
    # Fail closed rather than reinterpret. A v1 row dated its universe data with a
    # timestamp taken before the request, and it carries no entry timing at all;
    # silently reading it as v2 would invent both.
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise PopulationDatasetError(
            f"unsupported_population_schema_version:expected_{SCHEMA_VERSION}"
        )
    try:
        record = PopulationDecision(
            schema_version=payload.get("schema_version"),
            cycle_id=payload.get("cycle_id"),
            snapshot_id=payload.get("snapshot_id"),
            input_hash=payload.get("input_hash"),
            universe_refreshed_at=payload.get("universe_refreshed_at"),
            universe_request_started_at=payload.get("universe_request_started_at"),
            universe_received_at=payload.get("universe_received_at"),
            scan_observed_at=payload.get("scan_observed_at"),
            candle_cutoff_ts=payload.get("candle_cutoff_ts"),
            decision_ts=payload.get("decision_ts"),
            ranking_ready_ts=payload.get("ranking_ready_ts"),
            cycle_completed_ts=payload.get("cycle_completed_ts"),
            actionable_ts=payload.get("actionable_ts"),
            entry_eligible_ts=payload.get("entry_eligible_ts"),
            entry_bar_open_ts=payload.get("entry_bar_open_ts"),
            symbol=payload.get("symbol"),
            timeframe=payload.get("timeframe"),
            status=payload.get("status"),
            base_bar_open_ts=payload.get("base_bar_open_ts"),
            base_bar_close_ts=payload.get("base_bar_close_ts"),
            action=payload.get("action"),
            reason=payload.get("reason"),
            confidence=payload.get("confidence"),
            metadata=metadata,
            cycle_ordinal=payload.get("cycle_ordinal"),
            cycle_size=payload.get("cycle_size"),
            error_code=payload.get("error_code"),
        )
    except (PopulationJournalError, TypeError, ValueError) as exc:
        raise PopulationDatasetError("invalid_population_record") from exc

    encoded_timeframe_seconds = _integer(
        payload.get("timeframe_seconds"), field="timeframe_seconds"
    )
    if encoded_timeframe_seconds != interval_seconds(record.timeframe):
        raise PopulationDatasetError("timeframe_seconds_mismatch")

    provenance = _mapping(metadata.get("provenance"), field="provenance")
    strategy_config_hash = str(provenance.get("strategy_config_hash") or "")
    universe_policy_hash = str(provenance.get("universe_policy_hash") or "")
    for name, value in (
        ("strategy_config_hash", strategy_config_hash),
        ("universe_policy_hash", universe_policy_hash),
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise PopulationDatasetError(f"{name}_must_be_sha256")

    snapshot = _mapping(metadata.get("feature_snapshot"), field="feature_snapshot")
    contract_version = str(snapshot.get("contract_version") or "")
    contract_hash = str(snapshot.get("contract_hash") or "")
    if contract_version != FEATURE_CONTRACT_VERSION:
        raise PopulationDatasetError("feature_contract_version_mismatch")
    if contract_hash != feature_contract_hash():
        raise PopulationDatasetError("feature_contract_hash_mismatch")
    rebuilt_snapshot = build_runtime_feature_snapshot(
        metadata,
        bar_cutoff_ts=record.candle_cutoff_ts,
        universe_refreshed_at=record.universe_refreshed_at,
    )
    if snapshot != rebuilt_snapshot:
        raise PopulationDatasetError("feature_snapshot_source_mismatch")

    raw_values = _mapping(snapshot.get("values"), field="feature_values")
    raw_observed = _mapping(snapshot.get("observed"), field="feature_observed")
    raw_reasons = _mapping(snapshot.get("availability_reason"), field="feature_availability_reason")
    expected_names = [spec.name for spec in captured_feature_specs()]
    if (
        set(raw_values) != set(expected_names)
        or set(raw_observed) != set(expected_names)
        or set(raw_reasons) != set(expected_names)
    ):
        raise PopulationDatasetError("feature_schema_mismatch")

    features: dict[str, float | None] = {}
    observed: dict[str, int] = {}
    availability_reason: dict[str, str] = {}
    for name in expected_names:
        flag = raw_observed[name]
        if isinstance(flag, bool) or not isinstance(flag, int) or flag not in (0, 1):
            raise PopulationDatasetError(f"invalid_observed_flag:{name}")
        reason = raw_reasons[name]
        if not isinstance(reason, str) or not reason:
            raise PopulationDatasetError(f"invalid_availability_reason:{name}")
        observed[name] = int(flag)
        availability_reason[name] = reason
        if flag == 0:
            features[name] = None
            continue
        features[name] = _finite_number(raw_values[name], field=f"feature:{name}")

    input_hash = record.input_hash
    causal_payload = {
        "schema_version": record.schema_version,
        "cycle_id": record.cycle_id,
        "symbol": record.symbol,
        "timeframe_seconds": encoded_timeframe_seconds,
        "status": record.status,
        "candle_cutoff_ts": record.candle_cutoff_ts,
        "base_bar_open_ts": record.base_bar_open_ts,
        "base_bar_close_ts": record.base_bar_close_ts,
        "action": record.action,
        "reason": record.reason,
        "confidence": record.confidence,
        "metadata": metadata,
        "error_code": record.error_code,
    }
    encoded = json.dumps(
        causal_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    expected_input_hash = hashlib.sha256(encoded).hexdigest()
    if input_hash != expected_input_hash:
        raise PopulationDatasetError("population_input_hash_mismatch")
    expected_snapshot_id = hashlib.sha256(
        json.dumps(
            {
                "schema_version": record.schema_version,
                "cycle_id": record.cycle_id,
                "symbol": record.symbol,
                "input_hash": input_hash,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if record.snapshot_id != expected_snapshot_id:
        raise PopulationDatasetError("population_snapshot_id_mismatch")

    return PopulationFeatureRow(
        schema_version=record.schema_version,
        snapshot_id=record.snapshot_id,
        cycle_id=record.cycle_id,
        cycle_ordinal=record.cycle_ordinal,
        cycle_size=record.cycle_size,
        symbol=record.symbol,
        timeframe=record.timeframe,
        status=record.status,
        action=record.action,
        bar_cutoff_ts=record.candle_cutoff_ts,
        decision_completed_ts=record.decision_ts,
        universe_refreshed_at=record.universe_refreshed_at,
        universe_received_at=record.universe_received_at,
        ranking_ready_ts=record.ranking_ready_ts,
        cycle_completed_ts=record.cycle_completed_ts,
        actionable_ts=record.actionable_ts,
        entry_eligible_ts=record.entry_eligible_ts,
        entry_bar_open_ts=record.entry_bar_open_ts,
        strategy_config_hash=strategy_config_hash,
        universe_policy_hash=universe_policy_hash,
        feature_contract_version=contract_version,
        feature_contract_hash=contract_hash,
        features=features,
        observed=observed,
        availability_reason=availability_reason,
    )


def _rebuild_envelope(payload: Mapping[str, Any]) -> CycleEnvelope:
    """Reconstruct the envelope from its own fields rather than trusting them.

    Rebuilding re-runs every temporal invariant and re-derives actionable and
    entry timing, so a hand-edited or drifted header cannot pass by simply
    carrying self-consistent numbers.
    """

    try:
        timings = tuple(
            SourceTiming(
                source=str(item.get("source")),
                request_started_at=item.get("request_started_at"),
                received_at=item.get("received_at"),
                status=str(item.get("status")),
                source_as_of=item.get("source_as_of"),
                error_code=item.get("error_code"),
            )
            for item in payload.get("source_timings", ())
        )
    except (AttributeError, SourceTimingError, TypeError, ValueError) as exc:
        raise PopulationDatasetError("invalid_cycle_source_timing") from exc
    if not timings:
        raise PopulationDatasetError("cycle_envelope_has_no_source_timings")

    universe_name = str(_mapping(payload.get("universe_timing"), field="universe_timing").get("source"))
    universe = next((timing for timing in timings if timing.source == universe_name), None)
    if universe is None:
        raise PopulationDatasetError("cycle_envelope_universe_timing_missing")

    try:
        rebuilt = CycleEnvelope.build(
            cycle_id=str(payload.get("cycle_id") or ""),
            timeframe=str(payload.get("timeframe") or ""),
            cycle_started_at=payload.get("cycle_started_at"),
            candle_cutoff_ts=payload.get("candle_cutoff_ts"),
            universe_symbols=tuple(payload.get("universe_symbols", ())),
            universe_timing=universe,
            source_timings=timings,
            strategy_config_hash=str(payload.get("strategy_config_hash") or ""),
            universe_policy_hash=str(payload.get("universe_policy_hash") or ""),
            ranking_ready_ts=payload.get("ranking_ready_ts"),
            cycle_completed_ts=payload.get("cycle_completed_ts"),
            status=str(payload.get("status") or ""),
            error_code=payload.get("error_code"),
        )
    except (CycleEnvelopeError, TypeError, ValueError) as exc:
        raise PopulationDatasetError("invalid_cycle_envelope") from exc

    if rebuilt.as_dict() != dict(payload):
        raise PopulationDatasetError("cycle_envelope_source_mismatch")
    return rebuilt


def _check_row_against_envelope(row: PopulationFeatureRow, envelope: CycleEnvelope) -> None:
    """Every cycle-level fact on a row must be the envelope's, not its own."""

    for row_field, envelope_field in (
        ("cycle_id", "cycle_id"),
        ("timeframe", "timeframe"),
        ("bar_cutoff_ts", "candle_cutoff_ts"),
        ("ranking_ready_ts", "ranking_ready_ts"),
        ("cycle_completed_ts", "cycle_completed_ts"),
        ("actionable_ts", "actionable_ts"),
        ("entry_eligible_ts", "entry_eligible_ts"),
        ("entry_bar_open_ts", "entry_bar_open_ts"),
        ("strategy_config_hash", "strategy_config_hash"),
        ("universe_policy_hash", "universe_policy_hash"),
    ):
        if getattr(row, row_field) != getattr(envelope, envelope_field):
            raise PopulationDatasetError(f"row_disagrees_with_cycle_envelope:{row_field}")
    if row.universe_received_at != envelope.universe_timing.received_at:
        raise PopulationDatasetError("row_disagrees_with_cycle_envelope:universe_received_at")


def iter_population_cycles(path: str | Path) -> Iterator[tuple[CycleEnvelope, list[PopulationFeatureRow]]]:
    """Yield each complete, verified cycle as its envelope plus ordered rows."""

    source = Path(path)
    if not source.exists():
        raise PopulationDatasetError("population_journal_not_found")
    try:
        handle = source.open("r", encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise PopulationDatasetError("population_journal_unreadable") from exc

    envelope: CycleEnvelope | None = None
    declared_rows = 0
    cycle_rows: list[PopulationFeatureRow] = []
    completed_cycles: set[str] = set()
    saw_any = False

    with handle:
        try:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                if len(line) > _MAX_JOURNAL_LINE_CHARS:
                    raise PopulationDatasetError(f"journal_line_too_large:{line_number}")
                try:
                    decoded = json.loads(
                        line,
                        parse_constant=_reject_json_constant,
                        object_pairs_hook=_unique_json_object,
                    )
                except (json.JSONDecodeError, ValueError) as exc:
                    raise PopulationDatasetError(f"invalid_json_line:{line_number}") from exc
                payload = _mapping(decoded, field="journal_row")
                if payload.get("schema_version") != SCHEMA_VERSION:
                    raise PopulationDatasetError(
                        f"unsupported_population_schema_version:expected_{SCHEMA_VERSION}"
                    )
                record_type = payload.get("record_type")

                if record_type == RECORD_TYPE_HEADER:
                    if envelope is not None:
                        raise PopulationDatasetError("new_cycle_before_previous_cycle_completed")
                    envelope = _rebuild_envelope(
                        _mapping(payload.get("envelope"), field="envelope")
                    )
                    if envelope.cycle_id != payload.get("cycle_id"):
                        raise PopulationDatasetError("cycle_header_id_mismatch")
                    if envelope.cycle_id in completed_cycles:
                        raise PopulationDatasetError("duplicate_cycle")
                    declared_rows = _integer(payload.get("row_count"), field="row_count")
                    cycle_rows = []
                    saw_any = True
                    continue

                if envelope is None:
                    raise PopulationDatasetError("journal_row_before_its_cycle_header")

                if record_type == RECORD_TYPE_DECISION:
                    row = _parse_feature_row(payload)
                    if row.cycle_ordinal != len(cycle_rows) or row.cycle_size != declared_rows:
                        raise PopulationDatasetError("incomplete_or_unordered_cycle")
                    if any(item.symbol == row.symbol for item in cycle_rows):
                        raise PopulationDatasetError("duplicate_symbol_or_snapshot_in_cycle")
                    if any(item.snapshot_id == row.snapshot_id for item in cycle_rows):
                        raise PopulationDatasetError("duplicate_symbol_or_snapshot_in_cycle")
                    _check_row_against_envelope(row, envelope)
                    cycle_rows.append(row)
                    continue

                if record_type == RECORD_TYPE_FOOTER:
                    if payload.get("cycle_id") != envelope.cycle_id:
                        raise PopulationDatasetError("cycle_footer_id_mismatch")
                    if _integer(payload.get("row_count"), field="row_count") != len(cycle_rows):
                        raise PopulationDatasetError("cycle_row_count_mismatch")
                    if payload.get("rows_checksum") != rows_checksum(
                        [item.snapshot_id for item in cycle_rows]
                    ):
                        raise PopulationDatasetError("cycle_rows_checksum_mismatch")
                    if cycle_rows:
                        expected_cycle_id = make_cycle_id(
                            timeframe=envelope.timeframe,
                            candle_cutoff_ts=envelope.candle_cutoff_ts,
                            universe_received_at=envelope.universe_timing.received_at,
                            universe_symbols=[item.symbol for item in cycle_rows],
                            schema_version=SCHEMA_VERSION,
                        )
                        if expected_cycle_id != envelope.cycle_id:
                            raise PopulationDatasetError("population_cycle_id_mismatch")
                    completed_cycles.add(envelope.cycle_id)
                    yield envelope, cycle_rows
                    envelope = None
                    declared_rows = 0
                    cycle_rows = []
                    continue

                raise PopulationDatasetError(f"unknown_journal_record_type:{record_type!r}")
        except UnicodeError as exc:
            raise PopulationDatasetError("population_journal_invalid_encoding") from exc

    if envelope is not None:
        raise PopulationDatasetError("journal_ends_with_incomplete_cycle")
    if not saw_any:
        raise PopulationDatasetError("population_journal_is_empty")


def iter_population_feature_rows(path: str | Path) -> Iterator[PopulationFeatureRow]:
    """Yield only rows belonging to complete, envelope-verified cycles."""

    for _envelope, rows in iter_population_cycles(path):
        yield from rows


def population_feature_records(path: str | Path) -> list[dict[str, Any]]:
    """Return flat records while keeping availability masks out of feature values."""

    records: list[dict[str, Any]] = []
    for row in iter_population_feature_rows(path):
        record: dict[str, Any] = {
            "snapshot_id": row.snapshot_id,
            "schema_version": row.schema_version,
            "cycle_id": row.cycle_id,
            "cycle_ordinal": row.cycle_ordinal,
            "cycle_size": row.cycle_size,
            "symbol": row.symbol,
            "timeframe": row.timeframe,
            "status": row.status,
            "action": row.action,
            "bar_cutoff_ts": row.bar_cutoff_ts,
            "decision_completed_ts": row.decision_completed_ts,
            "universe_refreshed_at": row.universe_refreshed_at,
            "universe_received_at": row.universe_received_at,
            "ranking_ready_ts": row.ranking_ready_ts,
            "cycle_completed_ts": row.cycle_completed_ts,
            "actionable_ts": row.actionable_ts,
            "entry_eligible_ts": row.entry_eligible_ts,
            "entry_bar_open_ts": row.entry_bar_open_ts,
            "strategy_config_hash": row.strategy_config_hash,
            "universe_policy_hash": row.universe_policy_hash,
            "feature_contract_version": row.feature_contract_version,
            "feature_contract_hash": row.feature_contract_hash,
        }
        record.update(row.features)
        record.update({f"{name}__observed": value for name, value in row.observed.items()})
        record.update(
            {f"{name}__availability_reason": value for name, value in row.availability_reason.items()}
        )
        records.append(record)
    return records


def model_input_records(path: str | Path) -> list[dict[str, Any]]:
    """Return nested, whitelist-only inputs with no action/status/rule columns."""

    names = model_feature_names()
    records: list[dict[str, Any]] = []
    for row in iter_population_feature_rows(path):
        records.append(
            {
                "snapshot_id": row.snapshot_id,
                "cycle_id": row.cycle_id,
                "symbol": row.symbol,
                "bar_cutoff_ts": row.bar_cutoff_ts,
                "feature_names": names,
                "features": {name: row.features[name] for name in names},
                "observed": {name: row.observed[name] for name in names},
            }
        )
    return records
