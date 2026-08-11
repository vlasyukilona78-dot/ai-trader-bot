"""Strict bridge from the runtime population journal to model feature rows.

This module does not create labels and does not train a model.  It exists to
make the first half of the causal path executable without silently falling back
to the historical event-conditioned CSV builders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterator, Mapping

from trading.market_data.bar_contract import interval_seconds
from trading.market_data.frame_provenance import SourceReadEvidenceV1
from trading.metrics.cycle_envelope import CycleEnvelope, CycleEnvelopeError
from trading.signals.lifecycle_contract import (
    CandidateLifecycleEventV1,
    LifecycleContractError,
)
from trading.metrics.population_journal import (
    CYCLE_IDENTITY_VERSION,
    CURRENT_WRITE_SCHEMA,
    EVIDENCE_CONTRACT_KEYS,
    FEATURE_PROVENANCE_KEYS,
    FOOTER_CORE_KEYS,
    FOOTER_KEYS,
    HEADER_KEYS,
    HEADER_KEYS_V5,
    HEADER_KEYS_V6,
    RECORD_TYPE_DECISION,
    RECORD_TYPE_FOOTER,
    RECORD_TYPE_HEADER,
    SCHEMA_VERSION,
    JournalCheckpointReceipt,
    PopulationDecision,
    PopulationDecisionV6,
    PopulationJournalError,
    _causal_metadata,
    _LifecycleChainState,
    _validate_benchmark_source_evidence,
    _validate_cycle_source_timings,
    _validate_record_against_envelope,
    _validate_v6_decision_evidence,
    _validated_decision_record,
    _validated_evidence_contracts,
    _validated_source_evidence,
    compute_cycle_commit,
    genesis_cycle_commit,
    header_keys_for_schema,
    make_cycle_id,
    update_rows_checksum,
)

from .feature_contract import (
    FEATURE_CONTRACT_VERSION,
    build_runtime_feature_snapshot,
    captured_feature_specs,
    feature_contract_hash,
    market_feature_hash,
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
    journal_schema_version: int
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
    universe_source_ts: float | None
    universe_cache_hit: bool
    ranking_ready_ts: float
    cycle_completed_ts: float
    actionable_ts: float
    entry_eligible_ts: float
    entry_bar_open_ts: float
    strategy_config_hash: str
    universe_policy_hash: str
    feature_contract_version: str
    feature_contract_hash: str
    envelope_hash: str
    market_feature_hash: str
    features: Mapping[str, float | None]
    observed: Mapping[str, int]
    availability_reason: Mapping[str, str]
    evidence_contracts: Mapping[str, str] | None = None
    benchmark_source_evidence: Mapping[str, object] | None = None
    base_source_evidence: Mapping[str, object] | None = None
    higher_timeframe_source_evidence: Mapping[str, object] | None = None
    raw_frame_bundle_hash: str | None = None
    lifecycle_event: Mapping[str, object] | None = None
    typed_evidence_status: str = "legacy_missing"


@dataclass(frozen=True)
class PopulationJournalTrustState:
    """Integrity boundary proven by one complete reader pass."""

    integrity: str
    journal_schema_version: int
    journal_id: str
    last_sequence_no: int
    last_cycle_id: str
    last_cycle_commit: str
    anchored_through_sequence_no: int | None


@dataclass
class _ReaderState:
    journal_schema_version: int | None = None
    journal_id: str | None = None
    strategy_spec_identity: tuple[str, str, str] | None = None
    evidence_contracts: Mapping[str, str] | None = None
    last_sequence_no: int = -1
    last_cycle_id: str | None = None
    last_cycle_commit: str | None = None
    checkpoint_seen: bool = False
    cycle_ids: list[str] = field(default_factory=list)
    cycle_commits: list[str] = field(default_factory=list)

    def bind_schema(self, schema_version: int) -> None:
        if schema_version not in {SCHEMA_VERSION, CURRENT_WRITE_SCHEMA}:
            raise PopulationDatasetError("unsupported_population_schema_version")
        if self.journal_schema_version is None:
            self.journal_schema_version = schema_version
        elif schema_version != self.journal_schema_version:
            raise PopulationDatasetError("mixed_population_schema_versions")

    def bind_evidence_contracts(self, payload: Mapping[str, str] | None) -> None:
        if self.journal_schema_version == SCHEMA_VERSION:
            if payload is not None:
                raise PopulationDatasetError("legacy_v5_must_not_claim_typed_evidence")
            return
        if payload is None:
            raise PopulationDatasetError("v6_evidence_contracts_missing")
        if self.evidence_contracts is None:
            self.evidence_contracts = dict(payload)
        elif dict(payload) != dict(self.evidence_contracts):
            raise PopulationDatasetError("mixed_evidence_contract_identities")

    def bind_strategy_spec(self, envelope: CycleEnvelope) -> None:
        identity = (
            envelope.strategy_spec_version,
            envelope.strategy_spec_contract_hash,
            envelope.strategy_spec_instance_hash,
        )
        if self.strategy_spec_identity is None:
            self.strategy_spec_identity = identity
        elif identity != self.strategy_spec_identity:
            raise PopulationDatasetError("mixed_strategy_spec_identities")

    def trust_state(
        self,
        trusted_checkpoint: JournalCheckpointReceipt | None,
    ) -> PopulationJournalTrustState:
        if (
            self.journal_id is None
            or self.journal_schema_version is None
            or self.last_cycle_id is None
            or self.last_cycle_commit is None
            or self.last_sequence_no < 0
        ):
            raise PopulationDatasetError("population_journal_has_no_complete_cycles")
        if trusted_checkpoint is not None and not self.checkpoint_seen:
            raise PopulationDatasetError("trusted_checkpoint_not_found_in_journal")
        anchored = trusted_checkpoint.sequence_no if trusted_checkpoint is not None else None
        return PopulationJournalTrustState(
            integrity=(
                f"anchored_through_sequence={anchored}"
                if anchored is not None
                else "internally_consistent_unanchored"
            ),
            journal_schema_version=self.journal_schema_version,
            journal_id=self.journal_id,
            last_sequence_no=self.last_sequence_no,
            last_cycle_id=self.last_cycle_id,
            last_cycle_commit=self.last_cycle_commit,
            anchored_through_sequence_no=anchored,
        )


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


def _parse_feature_row(
    payload: Mapping[str, Any],
    *,
    schema_version: int,
    envelope: CycleEnvelope,
    evidence_contracts: Mapping[str, str] | None,
    benchmark_source_evidence: SourceReadEvidenceV1 | None,
) -> PopulationFeatureRow:
    metadata = _mapping(payload.get("metadata"), field="metadata")
    # Fail closed rather than reinterpret. A v1 row dated its universe data with a
    # timestamp taken before the request, and it carries no entry timing at all;
    # silently reading it as v2 would invent both.
    if payload.get("schema_version") != schema_version:
        raise PopulationDatasetError("population_row_schema_version_mismatch")
    # Diagnose the semantic feature contract before the row-level digest.  A
    # stale input_hash still fails closed, but callers get the actual schema or
    # source-drift reason instead of an opaque reconstruction wrapper.
    pre_snapshot = _mapping(
        metadata.get("feature_snapshot"), field="feature_snapshot"
    )
    if pre_snapshot.get("contract_version") != FEATURE_CONTRACT_VERSION:
        raise PopulationDatasetError("feature_contract_version_mismatch")
    if pre_snapshot.get("contract_hash") != feature_contract_hash():
        raise PopulationDatasetError("feature_contract_hash_mismatch")
    try:
        pre_rebuilt_snapshot = build_runtime_feature_snapshot(
            metadata,
            bar_cutoff_ts=_finite_number(
                payload.get("candle_cutoff_ts"), field="candle_cutoff_ts"
            ),
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise PopulationDatasetError("feature_snapshot_cannot_be_rebuilt") from exc
    if pre_snapshot != pre_rebuilt_snapshot:
        raise PopulationDatasetError("feature_snapshot_source_mismatch")
    try:
        record = _validated_decision_record(
            payload,
            schema_version=schema_version,
        )
    except (PopulationJournalError, TypeError, ValueError) as exc:
        raise PopulationDatasetError("invalid_population_record") from exc
    if schema_version == CURRENT_WRITE_SCHEMA:
        if (
            not isinstance(record, PopulationDecisionV6)
            or evidence_contracts is None
            or benchmark_source_evidence is None
        ):
            raise PopulationDatasetError("v6_typed_evidence_context_missing")
        try:
            _validate_record_against_envelope(record, envelope=envelope)
            _validate_v6_decision_evidence(
                record,
                envelope=envelope,
                benchmark_source_evidence=benchmark_source_evidence,
            )
        except PopulationJournalError as exc:
            raise PopulationDatasetError("invalid_v6_decision_evidence") from exc

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
    )
    if snapshot != rebuilt_snapshot:
        raise PopulationDatasetError("feature_snapshot_source_mismatch")

    feature_provenance = _mapping(
        metadata.get("feature_provenance"), field="feature_provenance"
    )
    if set(feature_provenance) != FEATURE_PROVENANCE_KEYS:
        raise PopulationDatasetError("feature_provenance_schema_mismatch")
    recorded_envelope_hash = feature_provenance.get("envelope_hash")
    recorded_market_hash = feature_provenance.get("market_feature_hash")
    provenance_received_at = _finite_number(
        feature_provenance.get("universe_received_at"),
        field="feature_provenance_universe_received_at",
    )
    raw_source_ts = feature_provenance.get("universe_source_ts")
    provenance_source_ts = (
        _finite_number(raw_source_ts, field="feature_provenance_universe_source_ts")
        if raw_source_ts is not None
        else None
    )
    provenance_cache_hit = feature_provenance.get("universe_cache_hit")
    if type(provenance_cache_hit) is not bool:
        raise PopulationDatasetError("feature_provenance_universe_cache_hit_must_be_boolean")
    if provenance_received_at != record.universe_received_at:
        raise PopulationDatasetError("feature_provenance_universe_received_at_mismatch")
    if not isinstance(recorded_envelope_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", recorded_envelope_hash
    ):
        raise PopulationDatasetError("feature_provenance_envelope_hash_must_be_sha256")
    if not isinstance(recorded_market_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", recorded_market_hash
    ):
        raise PopulationDatasetError("feature_provenance_market_feature_hash_must_be_sha256")
    if recorded_market_hash != market_feature_hash(
        rebuilt_snapshot,
        symbol=record.symbol,
        timeframe_seconds=interval_seconds(record.timeframe),
    ):
        raise PopulationDatasetError("market_feature_hash_mismatch")

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

    return PopulationFeatureRow(
        journal_schema_version=schema_version,
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
        universe_source_ts=provenance_source_ts,
        universe_cache_hit=provenance_cache_hit,
        ranking_ready_ts=record.ranking_ready_ts,
        cycle_completed_ts=record.cycle_completed_ts,
        actionable_ts=record.actionable_ts,
        entry_eligible_ts=record.entry_eligible_ts,
        entry_bar_open_ts=record.entry_bar_open_ts,
        strategy_config_hash=strategy_config_hash,
        universe_policy_hash=universe_policy_hash,
        feature_contract_version=contract_version,
        feature_contract_hash=contract_hash,
        envelope_hash=recorded_envelope_hash,
        market_feature_hash=recorded_market_hash,
        features=features,
        observed=observed,
        availability_reason=availability_reason,
        evidence_contracts=(
            dict(evidence_contracts) if evidence_contracts is not None else None
        ),
        benchmark_source_evidence=(
            benchmark_source_evidence.as_dict()
            if benchmark_source_evidence is not None
            else None
        ),
        base_source_evidence=(
            record.base_source_evidence.as_dict()
            if isinstance(record, PopulationDecisionV6)
            and record.base_source_evidence is not None
            else None
        ),
        higher_timeframe_source_evidence=(
            record.higher_timeframe_source_evidence.as_dict()
            if isinstance(record, PopulationDecisionV6)
            and record.higher_timeframe_source_evidence is not None
            else None
        ),
        raw_frame_bundle_hash=(
            record.raw_frame_bundle_hash
            if isinstance(record, PopulationDecisionV6)
            else None
        ),
        lifecycle_event=(
            record.lifecycle_event.as_dict()
            if isinstance(record, PopulationDecisionV6)
            and record.lifecycle_event is not None
            else None
        ),
        typed_evidence_status=(
            "typed_v6" if schema_version == CURRENT_WRITE_SCHEMA else "legacy_missing"
        ),
    )


def _rebuild_envelope(payload: Mapping[str, Any]) -> CycleEnvelope:
    """Reconstruct the envelope from its own fields rather than trusting them.

    Rebuilding re-runs every temporal invariant and re-derives actionable and
    entry timing, so a hand-edited or drifted header cannot pass by simply
    carrying self-consistent numbers.
    """

    try:
        rebuilt = CycleEnvelope.from_dict(payload)
    except (CycleEnvelopeError, TypeError, ValueError) as exc:
        raise PopulationDatasetError("invalid_cycle_envelope") from exc
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
    if row.universe_source_ts != envelope.universe_timing.source_ts:
        raise PopulationDatasetError("row_disagrees_with_cycle_envelope:universe_source_ts")
    if row.universe_cache_hit is not envelope.universe_timing.cache_hit:
        raise PopulationDatasetError("row_disagrees_with_cycle_envelope:universe_cache_hit")
    if row.envelope_hash != envelope.envelope_hash():
        raise PopulationDatasetError("row_disagrees_with_cycle_envelope:envelope_hash")


def _journal_fingerprint(source: Path) -> tuple[int, int, int, int, int]:
    """Return a cheap stability token and reject a torn final JSONL record."""

    try:
        stat = source.stat()
        if stat.st_size:
            with source.open("rb") as handle:
                handle.seek(-1, 2)
                if handle.read(1) != b"\n":
                    raise PopulationDatasetError("journal_ends_without_newline")
    except PopulationDatasetError:
        raise
    except OSError as exc:
        raise PopulationDatasetError("population_journal_unreadable") from exc
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _coerce_trusted_checkpoint(
    value: JournalCheckpointReceipt | Mapping[str, object] | None,
) -> JournalCheckpointReceipt | None:
    if value is None:
        return None
    if isinstance(value, JournalCheckpointReceipt):
        return value
    try:
        return JournalCheckpointReceipt.from_dict(value)
    except (PopulationJournalError, TypeError, ValueError) as exc:
        raise PopulationDatasetError("invalid_trusted_checkpoint_receipt") from exc


def _parse_population_cycles(
    source: Path,
    *,
    trusted_checkpoint: JournalCheckpointReceipt | None,
    state: _ReaderState,
) -> Iterator[tuple[CycleEnvelope, list[PopulationFeatureRow]]]:
    """Parse one immutable view and rebuild its complete v5 commitment chain."""

    try:
        handle = source.open("rb")
    except OSError as exc:
        raise PopulationDatasetError("population_journal_unreadable") from exc

    envelope: CycleEnvelope | None = None
    current_header: Mapping[str, Any] | None = None
    current_journal_id: str | None = None
    current_sequence_no: int | None = None
    current_prev_commit: str | None = None
    current_schema_version: int | None = None
    current_evidence_contracts: Mapping[str, str] | None = None
    current_benchmark_source_evidence: SourceReadEvidenceV1 | None = None
    declared_rows = 0
    declared_envelope_hash: str | None = None
    cycle_rows: list[PopulationFeatureRow] = []
    row_payloads: list[Mapping[str, object]] = []
    seen_symbols: set[str] = set()
    seen_snapshots: set[str] = set()
    completed_cycles: set[str] = set()
    completed_commits: list[str] = []
    rows_digest = hashlib.sha256()
    file_digest = hashlib.sha256()
    prefix_length = 0
    saw_any = False
    lifecycle_chain = _LifecycleChainState()
    cycle_base_source_evidences: list[SourceReadEvidenceV1] = []
    cycle_higher_timeframe_source_evidences: list[SourceReadEvidenceV1] = []

    with handle:
        for line_number, raw in enumerate(handle, start=1):
            file_digest.update(raw)
            prefix_length += len(raw)
            if not raw.strip():
                raise PopulationDatasetError("population_journal_contains_blank_line")
            if len(raw) > _MAX_JOURNAL_LINE_CHARS:
                raise PopulationDatasetError(f"journal_line_too_large:{line_number}")
            try:
                line = raw.decode("utf-8")
                decoded = json.loads(
                    line,
                    parse_constant=_reject_json_constant,
                    object_pairs_hook=_unique_json_object,
                )
            except UnicodeError as exc:
                raise PopulationDatasetError("population_journal_invalid_encoding") from exc
            except (json.JSONDecodeError, ValueError) as exc:
                raise PopulationDatasetError(f"invalid_json_line:{line_number}") from exc
            payload = _mapping(decoded, field="journal_row")
            raw_schema_version = payload.get("schema_version")
            if isinstance(raw_schema_version, bool) or not isinstance(
                raw_schema_version, int
            ):
                raise PopulationDatasetError("invalid_population_schema_version")
            schema_version = int(raw_schema_version)
            state.bind_schema(schema_version)
            record_type = payload.get("record_type")

            if record_type == RECORD_TYPE_HEADER:
                if set(payload) != header_keys_for_schema(schema_version):
                    raise PopulationDatasetError("cycle_header_schema_mismatch")
                if envelope is not None:
                    raise PopulationDatasetError("new_cycle_before_previous_cycle_completed")
                envelope = _rebuild_envelope(
                    _mapping(payload.get("envelope"), field="envelope")
                )
                state.bind_strategy_spec(envelope)
                current_schema_version = schema_version
                if schema_version == CURRENT_WRITE_SCHEMA:
                    try:
                        current_evidence_contracts = _validated_evidence_contracts(
                            payload.get("evidence_contracts")
                        )
                        current_benchmark_source_evidence = (
                            _validated_source_evidence(
                                payload.get("benchmark_source_evidence"),
                                field="benchmark source evidence",
                            )
                        )
                        _validate_benchmark_source_evidence(
                            current_benchmark_source_evidence,
                            envelope=envelope,
                        )
                    except PopulationJournalError as exc:
                        raise PopulationDatasetError(
                            "invalid_v6_cycle_evidence_header"
                        ) from exc
                    state.bind_evidence_contracts(current_evidence_contracts)
                else:
                    current_evidence_contracts = None
                    current_benchmark_source_evidence = None
                    state.bind_evidence_contracts(None)
                current_header = payload
                if envelope.cycle_id != payload.get("cycle_id"):
                    raise PopulationDatasetError("cycle_header_id_mismatch")
                expected_cycle_id = make_cycle_id(
                    timeframe=envelope.timeframe,
                    candle_cutoff_ts=envelope.candle_cutoff_ts,
                    universe_received_at=envelope.universe_timing.received_at,
                    universe_symbols=envelope.universe_symbols,
                    schema_version=CYCLE_IDENTITY_VERSION,
                )
                if expected_cycle_id != envelope.cycle_id:
                    raise PopulationDatasetError("population_cycle_id_mismatch")
                if envelope.cycle_id in completed_cycles:
                    raise PopulationDatasetError("duplicate_cycle")

                journal_id = payload.get("journal_id")
                if not isinstance(journal_id, str) or not re.fullmatch(r"[0-9a-f]{64}", journal_id):
                    raise PopulationDatasetError("invalid_population_journal_id")
                if state.journal_id is None:
                    state.journal_id = journal_id
                    if trusted_checkpoint is not None and trusted_checkpoint.journal_id != journal_id:
                        raise PopulationDatasetError("trusted_checkpoint_journal_id_mismatch")
                elif state.journal_id != journal_id:
                    raise PopulationDatasetError("mixed_population_journal_ids")
                current_journal_id = journal_id
                current_sequence_no = _integer(
                    payload.get("sequence_no"), field="sequence_no"
                )
                if current_sequence_no != len(completed_commits):
                    raise PopulationDatasetError("non_contiguous_cycle_sequence")
                expected_prev = (
                    genesis_cycle_commit(
                        journal_id,
                        schema_version=schema_version,
                    )
                    if not completed_commits
                    else completed_commits[-1]
                )
                current_prev_commit = payload.get("prev_cycle_commit")  # type: ignore[assignment]
                if current_prev_commit != expected_prev:
                    raise PopulationDatasetError("cycle_predecessor_mismatch")

                declared_rows = _integer(payload.get("row_count"), field="row_count")
                if declared_rows < 0:
                    raise PopulationDatasetError("cycle_row_count_must_not_be_negative")
                declared_envelope_hash = payload.get("envelope_hash")
                if declared_envelope_hash != envelope.envelope_hash():
                    raise PopulationDatasetError("cycle_header_envelope_hash_mismatch")
                cycle_rows = []
                row_payloads = []
                rows_digest = hashlib.sha256()
                seen_symbols = set()
                seen_snapshots = set()
                cycle_base_source_evidences = []
                cycle_higher_timeframe_source_evidences = []
                saw_any = True
                continue

            if envelope is None:
                raise PopulationDatasetError("journal_row_before_its_cycle_header")

            if record_type == RECORD_TYPE_DECISION:
                if current_schema_version is None:
                    raise PopulationDatasetError("cycle_header_schema_state_missing")
                row = _parse_feature_row(
                    payload,
                    schema_version=current_schema_version,
                    envelope=envelope,
                    evidence_contracts=current_evidence_contracts,
                    benchmark_source_evidence=current_benchmark_source_evidence,
                )
                if row.cycle_ordinal != len(cycle_rows) or row.cycle_size != declared_rows:
                    raise PopulationDatasetError("incomplete_or_unordered_cycle")
                if row.symbol in seen_symbols or row.snapshot_id in seen_snapshots:
                    raise PopulationDatasetError("duplicate_symbol_or_snapshot_in_cycle")
                seen_symbols.add(row.symbol)
                seen_snapshots.add(row.snapshot_id)
                _check_row_against_envelope(row, envelope)
                if schema_version == CURRENT_WRITE_SCHEMA:
                    try:
                        if (
                            row.base_source_evidence is None
                            or row.higher_timeframe_source_evidence is None
                        ):
                            raise PopulationJournalError(
                                "v6 row lacks typed source evidence"
                            )
                        cycle_base_source_evidences.append(
                            _validated_source_evidence(
                                row.base_source_evidence,
                                field="base source evidence",
                            )
                        )
                        cycle_higher_timeframe_source_evidences.append(
                            _validated_source_evidence(
                                row.higher_timeframe_source_evidence,
                                field="higher-timeframe source evidence",
                            )
                        )
                        lifecycle_chain.observe(
                            symbol=row.symbol,
                            event=(
                                CandidateLifecycleEventV1.from_dict(
                                    row.lifecycle_event
                                )
                                if row.lifecycle_event is not None
                                else None
                            ),
                        )
                    except (PopulationJournalError, LifecycleContractError) as exc:
                        raise PopulationDatasetError(
                            "invalid_cross_cycle_lifecycle_chain"
                        ) from exc
                cycle_rows.append(row)
                row_payloads.append(payload)
                update_rows_checksum(rows_digest, payload)
                continue

            if record_type == RECORD_TYPE_FOOTER:
                if set(payload) != FOOTER_KEYS:
                    raise PopulationDatasetError("cycle_footer_schema_mismatch")
                if payload.get("cycle_id") != envelope.cycle_id:
                    raise PopulationDatasetError("cycle_footer_id_mismatch")
                footer_rows = _integer(payload.get("row_count"), field="row_count")
                if footer_rows != declared_rows or declared_rows != len(cycle_rows):
                    raise PopulationDatasetError("cycle_row_count_mismatch")
                if payload.get("envelope_hash") != declared_envelope_hash:
                    raise PopulationDatasetError("cycle_footer_envelope_hash_mismatch")
                if payload.get("rows_checksum") != rows_digest.hexdigest():
                    raise PopulationDatasetError("cycle_rows_checksum_mismatch")
                if payload.get("journal_id") != current_journal_id:
                    raise PopulationDatasetError("cycle_footer_journal_id_mismatch")
                if payload.get("sequence_no") != current_sequence_no:
                    raise PopulationDatasetError("cycle_footer_sequence_mismatch")
                if payload.get("prev_cycle_commit") != current_prev_commit:
                    raise PopulationDatasetError("cycle_footer_predecessor_mismatch")
                if current_header is None or current_sequence_no is None:
                    raise PopulationDatasetError("cycle_header_state_missing")
                footer_core = {
                    key: value for key, value in payload.items() if key != "cycle_commit"
                }
                if set(footer_core) != FOOTER_CORE_KEYS:
                    raise PopulationDatasetError("cycle_footer_core_schema_mismatch")
                try:
                    expected_commit = compute_cycle_commit(
                        current_header,
                        row_payloads,
                        footer_core,
                    )
                except PopulationJournalError as exc:
                    raise PopulationDatasetError("cycle_commitment_cannot_be_rebuilt") from exc
                if payload.get("cycle_commit") != expected_commit:
                    raise PopulationDatasetError("cycle_commitment_mismatch")

                row_symbols = tuple(item.symbol for item in cycle_rows)
                if envelope.status == "completed":
                    if not row_symbols:
                        raise PopulationDatasetError("completed_cycle_has_no_decision_rows")
                    if row_symbols != envelope.universe_symbols:
                        raise PopulationDatasetError("cycle_population_universe_mismatch")
                elif row_symbols:
                    raise PopulationDatasetError(
                        f"{envelope.status}_cycle_must_not_contain_decision_rows"
                    )
                if current_schema_version == CURRENT_WRITE_SCHEMA:
                    if current_benchmark_source_evidence is None:
                        raise PopulationDatasetError(
                            "v6_benchmark_source_evidence_missing"
                        )
                    try:
                        _validate_cycle_source_timings(
                            envelope=envelope,
                            benchmark_source_evidence=(
                                current_benchmark_source_evidence
                            ),
                            base_source_evidences=cycle_base_source_evidences,
                            higher_timeframe_source_evidences=(
                                cycle_higher_timeframe_source_evidences
                            ),
                        )
                    except PopulationJournalError as exc:
                        raise PopulationDatasetError(
                            "cycle_source_timing_evidence_mismatch"
                        ) from exc
                    lifecycle_chain.finish_cycle(envelope)

                if trusted_checkpoint is not None and current_sequence_no == trusted_checkpoint.sequence_no:
                    if trusted_checkpoint.journal_schema_version != current_schema_version:
                        raise PopulationDatasetError(
                            "trusted_checkpoint_journal_schema_mismatch"
                        )
                    if trusted_checkpoint.cycle_id != envelope.cycle_id:
                        raise PopulationDatasetError("trusted_checkpoint_cycle_id_mismatch")
                    if trusted_checkpoint.cycle_commit != expected_commit:
                        raise PopulationDatasetError("trusted_checkpoint_cycle_commit_mismatch")
                    if trusted_checkpoint.prefix_length_bytes != prefix_length:
                        raise PopulationDatasetError("trusted_checkpoint_prefix_length_mismatch")
                    if trusted_checkpoint.prefix_sha256 != file_digest.hexdigest():
                        raise PopulationDatasetError("trusted_checkpoint_prefix_sha256_mismatch")
                    state.checkpoint_seen = True

                completed_cycles.add(envelope.cycle_id)
                completed_commits.append(expected_commit)
                state.last_sequence_no = current_sequence_no
                state.last_cycle_id = envelope.cycle_id
                state.last_cycle_commit = expected_commit
                state.cycle_ids.append(envelope.cycle_id)
                state.cycle_commits.append(expected_commit)
                yield envelope, cycle_rows
                envelope = None
                current_header = None
                current_journal_id = None
                current_sequence_no = None
                current_prev_commit = None
                current_schema_version = None
                current_evidence_contracts = None
                current_benchmark_source_evidence = None
                declared_rows = 0
                declared_envelope_hash = None
                cycle_rows = []
                row_payloads = []
                rows_digest = hashlib.sha256()
                seen_symbols = set()
                seen_snapshots = set()
                cycle_base_source_evidences = []
                cycle_higher_timeframe_source_evidences = []
                continue

            raise PopulationDatasetError(f"unknown_journal_record_type:{record_type!r}")

    if envelope is not None:
        raise PopulationDatasetError("journal_ends_with_incomplete_cycle")
    if not saw_any:
        raise PopulationDatasetError("population_journal_is_empty")
    state.trust_state(trusted_checkpoint)


def verify_population_journal(
    path: str | Path,
    *,
    trusted_checkpoint: JournalCheckpointReceipt | Mapping[str, object] | None = None,
) -> PopulationJournalTrustState:
    """Validate the complete chain and return its explicit trust boundary.

    No sidecar is discovered automatically.  Supplying ``trusted_checkpoint``
    asserts that the caller obtained that exact receipt from a trust domain the
    journal writer could not rewrite.
    """

    source = Path(path)
    if not source.exists():
        raise PopulationDatasetError("population_journal_not_found")
    checkpoint = _coerce_trusted_checkpoint(trusted_checkpoint)
    before = _journal_fingerprint(source)
    state = _ReaderState()
    for _ in _parse_population_cycles(
        source,
        trusted_checkpoint=checkpoint,
        state=state,
    ):
        pass
    if _journal_fingerprint(source) != before:
        raise PopulationDatasetError("population_journal_changed_during_validation")
    return state.trust_state(checkpoint)


def iter_population_cycles(
    path: str | Path,
    *,
    trusted_checkpoint: JournalCheckpointReceipt | Mapping[str, object] | None = None,
    anchored_only: bool | None = None,
) -> Iterator[tuple[CycleEnvelope, list[PopulationFeatureRow]]]:
    """Yield cycles only after the entire journal has passed strict validation.

    The first pass deliberately yields nothing to the caller.  This prevents a
    streaming export from consuming valid early cycles before discovering a
    corrupt or mixed-schema record later in the same file.
    """

    source = Path(path)
    if not source.exists():
        raise PopulationDatasetError("population_journal_not_found")
    checkpoint = _coerce_trusted_checkpoint(trusted_checkpoint)
    if anchored_only is None:
        resolved_anchored_only = checkpoint is not None
    elif not isinstance(anchored_only, bool):
        raise PopulationDatasetError("anchored_only_must_be_boolean_or_none")
    elif anchored_only and checkpoint is None:
        raise PopulationDatasetError("trusted_checkpoint_required_for_anchored_read")
    else:
        resolved_anchored_only = anchored_only
    before = _journal_fingerprint(source)
    validation_state = _ReaderState()
    for _ in _parse_population_cycles(
        source,
        trusted_checkpoint=checkpoint,
        state=validation_state,
    ):
        pass
    validated = _journal_fingerprint(source)
    if validated != before:
        raise PopulationDatasetError("population_journal_changed_during_validation")
    yield_state = _ReaderState()
    for sequence_no, cycle in enumerate(_parse_population_cycles(
        source,
        trusted_checkpoint=checkpoint,
        state=yield_state,
    )):
        if sequence_no >= len(validation_state.cycle_commits) or (
            yield_state.cycle_ids[sequence_no] != validation_state.cycle_ids[sequence_no]
            or yield_state.cycle_commits[sequence_no]
            != validation_state.cycle_commits[sequence_no]
        ):
            raise PopulationDatasetError("population_journal_cycle_changed_after_validation")
        if _journal_fingerprint(source) != validated:
            raise PopulationDatasetError("population_journal_changed_after_validation")
        if (
            checkpoint is not None
            and resolved_anchored_only
            and sequence_no > checkpoint.sequence_no
        ):
            continue
        yield cycle
    if (
        yield_state.cycle_ids != validation_state.cycle_ids
        or yield_state.cycle_commits != validation_state.cycle_commits
        or _journal_fingerprint(source) != validated
    ):
        raise PopulationDatasetError("population_journal_changed_after_validation")


def iter_population_feature_rows(
    path: str | Path,
    *,
    trusted_checkpoint: JournalCheckpointReceipt | Mapping[str, object] | None = None,
    anchored_only: bool | None = None,
) -> Iterator[PopulationFeatureRow]:
    """Yield only rows belonging to complete, envelope-verified cycles."""

    for _envelope, rows in iter_population_cycles(
        path,
        trusted_checkpoint=trusted_checkpoint,
        anchored_only=anchored_only,
    ):
        yield from rows


def population_feature_records(
    path: str | Path,
    *,
    trusted_checkpoint: JournalCheckpointReceipt | Mapping[str, object] | None = None,
    anchored_only: bool | None = None,
) -> list[dict[str, Any]]:
    """Return flat records while keeping availability masks out of feature values."""

    records: list[dict[str, Any]] = []
    for row in iter_population_feature_rows(
        path,
        trusted_checkpoint=trusted_checkpoint,
        anchored_only=anchored_only,
    ):
        record: dict[str, Any] = {
            "snapshot_id": row.snapshot_id,
            "schema_version": row.schema_version,
            "journal_schema_version": row.journal_schema_version,
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
            "universe_source_ts": row.universe_source_ts,
            "universe_cache_hit": row.universe_cache_hit,
            "ranking_ready_ts": row.ranking_ready_ts,
            "cycle_completed_ts": row.cycle_completed_ts,
            "actionable_ts": row.actionable_ts,
            "entry_eligible_ts": row.entry_eligible_ts,
            "entry_bar_open_ts": row.entry_bar_open_ts,
            "strategy_config_hash": row.strategy_config_hash,
            "universe_policy_hash": row.universe_policy_hash,
            "feature_contract_version": row.feature_contract_version,
            "feature_contract_hash": row.feature_contract_hash,
            "envelope_hash": row.envelope_hash,
            "market_feature_hash": row.market_feature_hash,
            # Typed evidence remains nested, top-level metadata.  It is never
            # flattened into the numeric feature namespace below.
            "typed_evidence_status": row.typed_evidence_status,
            "evidence_contracts": row.evidence_contracts,
            "benchmark_source_evidence": row.benchmark_source_evidence,
            "base_source_evidence": row.base_source_evidence,
            "higher_timeframe_source_evidence": (
                row.higher_timeframe_source_evidence
            ),
            "raw_frame_bundle_hash": row.raw_frame_bundle_hash,
            "lifecycle_event": row.lifecycle_event,
        }
        record.update(row.features)
        record.update({f"{name}__observed": value for name, value in row.observed.items()})
        record.update(
            {f"{name}__availability_reason": value for name, value in row.availability_reason.items()}
        )
        records.append(record)
    return records


def model_input_records(
    path: str | Path,
    *,
    trusted_checkpoint: JournalCheckpointReceipt | Mapping[str, object] | None = None,
    allow_unanchored: bool = False,
    allow_legacy_v5: bool = False,
) -> list[dict[str, Any]]:
    """Return whitelist-only features plus non-predictive evidence identity."""

    if trusted_checkpoint is None and not allow_unanchored:
        raise PopulationDatasetError("trusted_checkpoint_required_for_model_inputs")

    trust = verify_population_journal(
        path,
        trusted_checkpoint=trusted_checkpoint,
    )
    if trust.journal_schema_version == SCHEMA_VERSION and not allow_legacy_v5:
        raise PopulationDatasetError(
            "legacy_v5_model_export_requires_explicit_opt_in"
        )

    names = model_feature_names()
    records: list[dict[str, Any]] = []
    for envelope, rows in iter_population_cycles(
        path,
        trusted_checkpoint=trusted_checkpoint,
        anchored_only=trusted_checkpoint is not None,
    ):
        for row in rows:
            records.append(
                {
                    "snapshot_id": row.snapshot_id,
                    "cycle_id": row.cycle_id,
                    "symbol": row.symbol,
                    "bar_cutoff_ts": row.bar_cutoff_ts,
                    "envelope_hash": row.envelope_hash,
                    "market_feature_hash": row.market_feature_hash,
                    "journal_schema_version": row.journal_schema_version,
                    "typed_evidence_status": row.typed_evidence_status,
                    "evidence_contracts": row.evidence_contracts,
                    "benchmark_source_evidence": row.benchmark_source_evidence,
                    "base_source_evidence": row.base_source_evidence,
                    "higher_timeframe_source_evidence": (
                        row.higher_timeframe_source_evidence
                    ),
                    "raw_frame_bundle_hash": row.raw_frame_bundle_hash,
                    "lifecycle_event": row.lifecycle_event,
                    # Evidence/partition metadata only. These fields deliberately
                    # remain outside ``features`` and must never become predictors.
                    "strategy_spec_version": envelope.strategy_spec_version,
                    "strategy_spec_contract_hash": (
                        envelope.strategy_spec_contract_hash
                    ),
                    "strategy_spec_instance_hash": (
                        envelope.strategy_spec_instance_hash
                    ),
                    "feature_names": names,
                    "features": {name: row.features[name] for name in names},
                    "observed": {name: row.observed[name] for name in names},
                }
            )
    return records
