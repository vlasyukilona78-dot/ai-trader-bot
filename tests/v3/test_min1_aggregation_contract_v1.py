from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pandas as pd
import pytest

from trading.market_data.min1_aggregation import (
    AggregatedMin1FrameV1,
    DuplicateMin1BarError,
    IncompleteAggregationGroupError,
    InvalidMin1FrameError,
    Min1AggregationError,
    Min1BarReceiptV1,
    Min1GapError,
    Min1ReceiptError,
    UnalignedMin1BarError,
    UnsupportedAggregationTargetError,
    aggregate_canonical_min1,
    min1_aggregation_contract_hash,
    normalized_min1_source_row_hash,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _frame(
    periods: int = 5,
    *,
    start: str = "2026-01-01T00:00:00Z",
) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="min")
    opens = np.arange(10.0, 10.0 + periods)
    return pd.DataFrame(
        {
            "open": opens,
            "high": opens + 2.0,
            "low": opens - 1.0,
            "close": opens + 1.0,
            "volume": np.arange(1.0, periods + 1.0),
            "turnover": np.arange(1.0, periods + 1.0) * 10.0,
        },
        index=index,
    )


def _receipts(
    frame: pd.DataFrame,
    *,
    receipt_delay: float = 1.0,
    content_prefix: str = "raw-page",
) -> list[Min1BarReceiptV1]:
    receipts: list[Min1BarReceiptV1] = []
    for ordinal, bar_open in enumerate(frame.index.tz_convert("UTC")):
        close_ts = float(bar_open.timestamp()) + 60.0
        receipts.append(
            Min1BarReceiptV1(
                bar_open_ts=float(bar_open.timestamp()),
                request_started_at=close_ts + receipt_delay - 0.25,
                received_at=close_ts + receipt_delay + ordinal / 10.0,
                source_content_hash=_digest(f"{content_prefix}:{ordinal}"),
                source_lineage_hash=_digest(f"manifest:{ordinal // 2}"),
                normalized_row_hash=normalized_min1_source_row_hash(
                    venue="mexc_contract",
                    symbol="BTCUSDT",
                    venue_symbol="BTC_USDT",
                    bar_open_ts=float(bar_open.timestamp()),
                    values=frame.iloc[ordinal].to_dict(),
                ),
            )
        )
    return receipts


def _aggregate(
    frame: pd.DataFrame,
    *,
    target: str = "Min5",
    receipts: list[Min1BarReceiptV1] | None = None,
) -> AggregatedMin1FrameV1:
    return aggregate_canonical_min1(
        frame,
        target_timeframe=target,
        receipts=_receipts(frame) if receipts is None else receipts,
        venue="mexc_contract",
        symbol="BTCUSDT",
        venue_symbol="BTC_USDT",
    )


def test_contract_hash_is_pinned() -> None:
    assert min1_aggregation_contract_hash() == (
        "0d851b253cde913d95a693e0db7296b59ff78a6048bacb20838386f2e8e20a21"
    )


def test_min5_aggregation_uses_exact_ohlc_volume_and_turnover_rules() -> None:
    frame = _frame()
    receipts = _receipts(frame)

    result = _aggregate(frame, target="5", receipts=list(reversed(receipts)))

    assert result.target_timeframe == "Min5"
    assert list(result.frame.columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
    ]
    assert list(result.frame.index) == [pd.Timestamp("2026-01-01T00:00:00Z")]
    assert result.frame.iloc[0].to_dict() == {
        "open": 10.0,
        "high": 16.0,
        "low": 9.0,
        "close": 15.0,
        "volume": 15.0,
        "turnover": 150.0,
    }
    assert result.frame.attrs["source_timeframe"] == "Min1"
    assert result.frame.attrs["target_timeframe"] == "Min5"

    evidence = result.evidence[0]
    assert evidence.source_bar_count == 5
    assert evidence.target_bar_open_ts == pd.Timestamp(
        "2026-01-01T00:00:00Z"
    ).timestamp()
    assert evidence.target_bar_close_ts == pd.Timestamp(
        "2026-01-01T00:05:00Z"
    ).timestamp()
    assert evidence.available_at == max(receipt.received_at for receipt in receipts)
    assert result.available_at == evidence.available_at
    assert [receipt.bar_open_ts for receipt in evidence.source_receipts] == [
        timestamp.timestamp() for timestamp in frame.index
    ]
    assert len(evidence.input_row_hashes) == 5
    assert all(len(value) == 64 for value in evidence.input_row_hashes)
    assert len(evidence.source_bundle_hash) == 64
    assert len(evidence.derived_content_hash) == 64
    assert len(evidence.evidence_hash) == 64
    assert evidence.input_row_hashes[0] == (
        "ece882aa1b8e87881aaaa37d47498a3d450fd3456e36156ba27a938557ffa3ad"
    )
    assert evidence.source_bundle_hash == (
        "f4f904cd695fc45f427300dea9f59bf2f89584dc419621f70b95a20d3a8c96e7"
    )
    assert evidence.derived_content_hash == (
        "4db3d8825312c0b61fd81e84b326323713c7a98f80ed2382425b86f6ab1a44fd"
    )
    assert evidence.evidence_hash == (
        "92745bce93713484337c4f36c806de939c2c0a162b4c677358d761d201446fe7"
    )


def test_preregistered_five_row_golden_vector() -> None:
    frame = pd.DataFrame(
        [
            (100, 103, 99, 102, 10, 1000),
            (102, 104, 101, 103, 20, 2100),
            (103, 105, 100, 101, 30, 3100),
            (101, 102, 97, 98, 40, 3900),
            (98, 100, 96, 99, 50, 4900),
        ],
        columns=["open", "high", "low", "close", "volume", "turnover"],
        index=pd.date_range("2026-01-01T00:00:00Z", periods=5, freq="min"),
        dtype=float,
    )
    result = _aggregate(frame)
    assert result.bars[0].values_dict() == {
        "open": 100.0,
        "high": 105.0,
        "low": 96.0,
        "close": 99.0,
        "volume": 150.0,
        "turnover": 15000.0,
    }


def test_content_identity_excludes_operational_latency_but_evidence_does_not() -> None:
    frame = _frame()
    fast = _aggregate(frame, receipts=_receipts(frame, receipt_delay=1.0))
    slow = _aggregate(frame, receipts=_receipts(frame, receipt_delay=20.0))

    fast_evidence = fast.evidence[0]
    slow_evidence = slow.evidence[0]
    assert fast_evidence.input_row_hashes == slow_evidence.input_row_hashes
    assert fast_evidence.source_bundle_hash == slow_evidence.source_bundle_hash
    assert fast_evidence.derived_content_hash == slow_evidence.derived_content_hash
    assert fast_evidence.available_at < slow_evidence.available_at
    assert fast_evidence.evidence_hash != slow_evidence.evidence_hash


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_pandas_datetime_storage_unit_does_not_change_aggregation(unit: str) -> None:
    frame = _frame()
    frame.index = frame.index.as_unit(unit)

    result = _aggregate(frame)

    assert result.frame.index.dtype == "datetime64[ns, UTC]"
    assert result.frame.iloc[0]["close"] == 15.0
    assert result.evidence[0].input_row_hashes[0] == (
        "ece882aa1b8e87881aaaa37d47498a3d450fd3456e36156ba27a938557ffa3ad"
    )


def test_projected_frame_is_a_detached_copy_bound_to_immutable_rows() -> None:
    result = _aggregate(_frame())
    original_hash = result.evidence[0].derived_content_hash
    projected = result.frame
    projected.iloc[0, projected.columns.get_loc("close")] = 15.5

    assert result.frame.iloc[0]["close"] == 15.0
    assert result.evidence[0].derived_content_hash == original_hash


def test_strict_round_trip_recomputes_all_evidence_and_content_bindings() -> None:
    result = _aggregate(_frame(periods=10))

    restored = AggregatedMin1FrameV1.from_dict(result.as_dict())

    pd.testing.assert_frame_equal(restored.frame, result.frame)
    assert restored.evidence == result.evidence
    assert restored.as_dict() == result.as_dict()


def test_parser_rejects_extra_keys_and_mutated_market_values() -> None:
    result = _aggregate(_frame())
    extra = result.as_dict()
    extra["unexpected"] = True
    with pytest.raises(Min1AggregationError, match="keys_are_not_exact"):
        AggregatedMin1FrameV1.from_dict(extra)

    changed_bar = result.as_dict()
    changed_bar["bars"][0]["close"] = 15.5
    with pytest.raises(Min1AggregationError, match="content_hash_mismatch"):
        AggregatedMin1FrameV1.from_dict(changed_bar)


def test_parser_rejects_changed_receipt_or_available_at_with_stale_hash() -> None:
    result = _aggregate(_frame())
    changed_receipt = result.as_dict()
    changed_receipt["evidence"][0]["source_receipts"][0]["received_at"] += 1.0
    with pytest.raises(Min1AggregationError):
        AggregatedMin1FrameV1.from_dict(changed_receipt)

    changed_available = result.as_dict()
    changed_available["evidence"][0]["available_at"] += 1.0
    with pytest.raises(Min1AggregationError, match="last_source_receipt"):
        AggregatedMin1FrameV1.from_dict(changed_available)


def test_direct_evidence_replacement_revalidates_hashes() -> None:
    evidence = _aggregate(_frame()).evidence[0]
    with pytest.raises(Min1AggregationError, match="last_source_receipt"):
        replace(evidence, available_at=evidence.available_at + 1.0)


@pytest.mark.parametrize("changed_field", ["source_content_hash", "source_lineage_hash"])
def test_upstream_content_lineage_is_bound_without_changing_derived_market_hash(
    changed_field: str,
) -> None:
    frame = _frame()
    baseline = _aggregate(frame)
    changed_receipts = _receipts(frame)
    changed_receipts[2] = replace(
        changed_receipts[2], **{changed_field: _digest(f"different-{changed_field}")}
    )
    changed = _aggregate(frame, receipts=changed_receipts)

    assert (
        baseline.evidence[0].derived_content_hash
        == changed.evidence[0].derived_content_hash
    )
    assert (
        baseline.evidence[0].source_bundle_hash
        != changed.evidence[0].source_bundle_hash
    )
    assert baseline.evidence[0].evidence_hash != changed.evidence[0].evidence_hash


def test_each_group_has_its_own_last_input_receipt_availability() -> None:
    frame = _frame(periods=10)
    receipts = _receipts(frame)
    receipts[2] = replace(
        receipts[2],
        request_started_at=receipts[2].request_started_at + 1_000.0,
        received_at=receipts[2].received_at + 1_000.0,
    )

    result = _aggregate(frame, receipts=receipts)

    assert len(result.frame) == 2
    assert result.evidence[0].available_at == receipts[2].received_at
    assert result.evidence[1].available_at == max(
        receipt.received_at for receipt in receipts[5:]
    )
    assert result.available_at == receipts[2].received_at


@pytest.mark.parametrize(
    ("target", "periods", "canonical"),
    [
        ("Min5", 5, "Min5"),
        ("15", 15, "Min15"),
        ("Min60", 60, "Min60"),
        ("Hour4", 240, "Hour4"),
    ],
)
def test_all_contract_targets_require_and_accept_one_complete_group(
    target: str,
    periods: int,
    canonical: str,
) -> None:
    result = _aggregate(_frame(periods=periods), target=target)
    assert len(result.frame) == 1
    assert result.target_timeframe == canonical
    assert result.evidence[0].source_bar_count == periods
    expected_sum = periods * (periods + 1) / 2.0
    assert result.frame.iloc[0].to_dict() == {
        "open": 10.0,
        "high": float(periods + 11),
        "low": 9.0,
        "close": float(periods + 10),
        "volume": expected_sum,
        "turnover": expected_sum * 10.0,
    }
    assert result.evidence[0].target_bar_close_ts == (
        pd.Timestamp("2026-01-01T00:00:00Z").timestamp() + periods * 60.0
    )


def test_contiguous_partial_group_is_an_explicit_error() -> None:
    frame = _frame(periods=4)
    with pytest.raises(
        IncompleteAggregationGroupError,
        match="incomplete_target_group",
    ):
        _aggregate(frame)


def test_internal_missing_minute_is_a_gap_error_not_a_partial_success() -> None:
    frame = _frame().drop(_frame().index[2])
    with pytest.raises(Min1GapError, match="missing_or_irregular_minutes"):
        _aggregate(frame)


def test_duplicate_minute_is_an_explicit_error() -> None:
    frame = pd.concat([_frame(), _frame().iloc[[2]]]).sort_index()
    with pytest.raises(DuplicateMin1BarError, match="duplicate_min1_bar_open"):
        _aggregate(frame)


def test_non_minute_timestamp_is_an_unaligned_error() -> None:
    frame = _frame()
    receipts = _receipts(frame)
    frame.index = frame.index + pd.Timedelta(seconds=30)
    with pytest.raises(UnalignedMin1BarError, match="utc_minute_aligned"):
        _aggregate(frame, receipts=receipts)


def test_complete_count_starting_inside_target_bucket_is_unaligned() -> None:
    frame = _frame(start="2026-01-01T00:01:00Z")
    with pytest.raises(UnalignedMin1BarError, match="target_utc_aligned"):
        _aggregate(frame)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda frame: frame.tz_localize(None),
        lambda frame: frame.iloc[::-1],
        lambda frame: frame.drop(columns="turnover"),
    ],
)
def test_noncanonical_frame_shape_fails_closed(mutate) -> None:
    invalid = mutate(_frame())
    with pytest.raises(InvalidMin1FrameError):
        _aggregate(invalid, receipts=_receipts(_frame()))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("close", float("nan"), "finite_number"),
        ("volume", -1.0, "volume_must_not_be_negative"),
        ("turnover", -1.0, "turnover_must_not_be_negative"),
        ("high", 0.0, "prices_must_be_positive"),
    ],
)
def test_invalid_market_values_fail_closed(
    column: str,
    value: float,
    message: str,
) -> None:
    frame = _frame()
    receipts = _receipts(frame)
    frame.iloc[2, frame.columns.get_loc(column)] = value
    with pytest.raises(InvalidMin1FrameError, match=message):
        _aggregate(frame, receipts=receipts)


@pytest.mark.parametrize("price", [0.0, -1.0])
def test_nonpositive_prices_are_not_canonical_mexc_bars(price: float) -> None:
    frame = _frame()
    receipts = _receipts(frame)
    for column in ("open", "high", "low", "close"):
        frame.loc[frame.index[0], column] = price
    with pytest.raises(InvalidMin1FrameError, match="prices_must_be_positive"):
        _aggregate(frame, receipts=receipts)


def test_serialized_receipt_row_hash_cannot_drift_from_input_hash() -> None:
    result = _aggregate(_frame())
    payload = result.as_dict()
    payload["evidence"][0]["source_receipts"][0]["normalized_row_hash"] = "0" * 64
    with pytest.raises(Min1ReceiptError, match="normalized_row_hash"):
        AggregatedMin1FrameV1.from_dict(payload)


def test_missing_or_extra_receipt_is_explicit() -> None:
    frame = _frame()
    with pytest.raises(Min1ReceiptError, match="do_not_match_input_rows"):
        _aggregate(frame, receipts=_receipts(frame)[:-1])

    extra_frame = _frame(periods=6)
    with pytest.raises(Min1ReceiptError, match="do_not_match_input_rows"):
        _aggregate(frame, receipts=_receipts(extra_frame))


def test_duplicate_receipt_is_explicit() -> None:
    frame = _frame()
    receipts = _receipts(frame)
    receipts[-1] = receipts[0]
    with pytest.raises(DuplicateMin1BarError, match="duplicate_min1_receipt"):
        _aggregate(frame, receipts=receipts)


def test_receipt_cannot_precede_its_closed_minute() -> None:
    open_ts = pd.Timestamp("2026-01-01T00:00:00Z").timestamp()
    with pytest.raises(Min1ReceiptError, match="precedes_source_bar_close"):
        Min1BarReceiptV1(
            bar_open_ts=open_ts,
            request_started_at=open_ts + 1.0,
            received_at=open_ts + 59.0,
            source_content_hash=_digest("raw"),
            source_lineage_hash=_digest("manifest"),
            normalized_row_hash=_digest("row"),
        )


def test_response_after_close_does_not_rehabilitate_a_preclose_request() -> None:
    open_ts = pd.Timestamp("2026-01-01T00:00:00Z").timestamp()
    with pytest.raises(
        Min1ReceiptError,
        match="request_started_at_precedes_source_bar_close",
    ):
        Min1BarReceiptV1(
            bar_open_ts=open_ts,
            request_started_at=open_ts + 59.9,
            received_at=open_ts + 61.0,
            source_content_hash=_digest("raw"),
            source_lineage_hash=_digest("manifest"),
            normalized_row_hash=_digest("row"),
        )


@pytest.mark.parametrize(
    "values",
    [
        {"source_content_hash": "not-a-hash"},
        {"source_lineage_hash": "A" * 64},
        {"normalized_row_hash": "not-a-hash"},
        {"request_started_at": 100.0, "received_at": 99.0},
    ],
)
def test_invalid_receipt_metadata_fails_closed(values: dict[str, object]) -> None:
    open_ts = pd.Timestamp("1970-01-01T00:00:00Z").timestamp()
    defaults: dict[str, object] = {
        "bar_open_ts": open_ts,
        "request_started_at": 60.0,
        "received_at": 61.0,
        "source_content_hash": _digest("raw"),
        "source_lineage_hash": _digest("manifest"),
        "normalized_row_hash": _digest("row"),
    }
    defaults.update(values)
    with pytest.raises(Min1ReceiptError):
        Min1BarReceiptV1(**defaults)


@pytest.mark.parametrize("target", ["Min1", "Min30", "Day1", "Month1"])
def test_target_outside_explicit_v3_set_is_rejected(target: str) -> None:
    frame = _frame()
    with pytest.raises(UnsupportedAggregationTargetError):
        _aggregate(frame, target=target)
