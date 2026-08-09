from __future__ import annotations

import hashlib
import json
from pathlib import Path

from core.mexc_strategy_spec import MexcStrategySpec
from trading.metrics.cycle_envelope import CycleEnvelope


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "mexc_strategy_v2_cycle_envelope_v3.json"
)
V2_CONTRACT_HASH = "9c62b88b7804e9663bae6f0eb429c58c541680b61d307c4f16032cb0b62fe3dd"
V2_INSTANCE_HASH = "9f0b2d7035c2a82ab1b6d8595245b8c3a7a8b9faad17bea8c57f6fcacb189466"
FIXTURE_CANONICAL_SHA256 = "87e3f049ca356f9cd7654464a6fa0cbb12ee319979e145de8aa021c858ee0e5e"


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_frozen_v2_evidence_remains_readable_after_future_spec_bumps() -> None:
    """A new current spec must not silently orphan committed v2 evidence."""

    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert _canonical_sha256(payload) == FIXTURE_CANONICAL_SHA256
    assert payload["schema_version"] == 3
    assert payload["strategy_spec_version"] == "mexc_strategy_v2"
    assert payload["strategy_spec_contract_hash"] == V2_CONTRACT_HASH
    assert payload["strategy_spec_instance_hash"] == V2_INSTANCE_HASH

    spec_payload = payload["strategy_spec_payload"]
    spec = MexcStrategySpec.from_mapping(spec_payload)
    assert spec.spec_version == "mexc_strategy_v2"
    assert spec.to_mapping() == spec_payload
    assert spec.instance_hash == V2_INSTANCE_HASH

    envelope = CycleEnvelope.from_dict(payload)
    assert envelope.strategy_spec_contract_hash == V2_CONTRACT_HASH
    assert envelope.strategy_spec_instance_hash == V2_INSTANCE_HASH
    assert envelope.as_dict() == payload
