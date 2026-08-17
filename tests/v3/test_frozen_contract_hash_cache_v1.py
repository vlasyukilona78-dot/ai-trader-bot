"""The shared contract-hash cache must not change a single hash value.

Contract identity in this project is a SHA-256 over canonical JSON, and those
hashes are read far more often than the contracts are built. `_frozen_contract_hash`
memoizes them by value. That is only safe while it stays a pure accelerator, so
these tests pin the two properties that make it one: it agrees with direct
recomputation, and equal-but-rebuilt contracts share an entry.

The last test guards a trap that is easy to walk into. `mexc_endpoint_official_evidence`
deliberately canonicalizes differently from every other module — it appends a
trailing newline before hashing — so routing its properties through the shared
helper silently changes every hash it produces. Its own pinned fixtures catch
that, but only after a confusing failure; this states the reason up front.
"""

from __future__ import annotations

from dataclasses import replace

import trading.market_data.mexc_endpoint_official_evidence as official
import trading.market_data.strict_history as shared
from trading.market_data.mexc_futures_transport import (
    candidate_history_resource_limits_v1,
    candidate_history_retry_policy_v1,
)

_SAMPLE = {"b": 1, "a": ["x", 2], "c": {"nested": True}}


def test_cached_hash_equals_direct_recomputation() -> None:
    for build in (candidate_history_resource_limits_v1, candidate_history_retry_policy_v1):
        contract = build()
        assert contract.contract_hash == shared._sha256_payload(contract.as_dict())


def test_equal_contracts_rebuilt_separately_share_one_hash() -> None:
    """The cache is keyed by value, so a fresh equal contract is not a miss."""

    first = candidate_history_resource_limits_v1()
    second = candidate_history_resource_limits_v1()

    assert first is not second
    assert first == second
    assert first.contract_hash == second.contract_hash


def test_a_changed_field_cannot_inherit_the_original_hash() -> None:
    """Value keying is what keeps tamper detection working: change a field and
    the contract stops comparing equal, so it gets its own cache entry."""

    original = candidate_history_resource_limits_v1()
    changed = replace(original, max_pages=original.max_pages - 1)

    assert changed != original
    assert changed.contract_hash != original.contract_hash
    assert changed.contract_hash == shared._sha256_payload(changed.as_dict())


def test_official_evidence_canonicalizes_differently_and_must_stay_separate() -> None:
    """Do not route official-evidence hashes through the shared helper."""

    assert official._sha256_payload(_SAMPLE) != shared._sha256_payload(_SAMPLE)
    assert official._canonical_json_bytes(_SAMPLE).endswith(b"\n")
    assert not shared._canonical_bytes(_SAMPLE).endswith(b"\n")
