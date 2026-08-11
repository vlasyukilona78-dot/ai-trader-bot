from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import requests

from trading.market_data.frame_provenance import (
    FrameProvenanceError,
    FrameRead,
    SourceReadEvidenceV1,
    canonical_closed_frame_hash,
    frame_provenance_contract_hash,
    parse_source_read_evidence,
    raw_frame_bundle_hash,
)
from trading.market_data.mexc_client import (
    MexcContractClient,
    MexcOhlcvApiError,
    MexcOhlcvJsonError,
    MexcOhlcvPayloadError,
    MexcOhlcvRequestError,
)


_CUTOFF = pd.Timestamp("2026-01-01T12:00:00Z").timestamp()


def _frame(*, close_shift: float = 0.0, turnover=None) -> pd.DataFrame:
    index = pd.date_range("2026-01-01T09:00:00Z", periods=3, freq="h")
    frame = pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0],
            "high": [10.5, 11.5, 12.5],
            "low": [9.5, 10.5, 11.5],
            "close": [10.1, 11.1, 12.1 + close_shift],
            "volume": [100.0, 110.0, 120.0],
        },
        index=index,
    )
    if turnover is not None:
        frame["turnover"] = turnover
    return frame


def _evidence(frame: pd.DataFrame, *, started: float, received: float):
    return SourceReadEvidenceV1.from_frame(
        frame,
        source="base_ohlcv",
        venue="mexc_contract",
        symbol="BTCUSDT",
        venue_symbol="BTC_USDT",
        timeframe="Min60",
        requested_as_of_ts=_CUTOFF,
        request_started_at=started,
        received_at=received,
        source_ts=received,
    )


def test_contract_hash_is_pinned() -> None:
    assert frame_provenance_contract_hash() == (
        "f4004ac933cc1725b2560e93ffbe278c826910424e350059ad29420ed3665dbf"
    )


def test_exact_frame_and_bundle_hash_golden_vectors() -> None:
    frame = _frame(turnover=[1000.0, 1100.0, 1200.0])
    evidence = _evidence(frame, started=_CUTOFF + 1.0, received=_CUTOFF + 2.0)

    assert evidence.frame_hash == (
        "bf038808c3e0d7cb87a63239a93649241068b601a97071192f9a5843a12ff090"
    )
    assert raw_frame_bundle_hash([evidence]) == (
        "4fc99859724d95beea76d174ad54aedbcb485a343e3929360c17331160752fa6"
    )


def test_frame_hash_and_raw_bundle_ignore_operational_latency() -> None:
    frame = _frame(turnover=[1000.0, 1100.0, 1200.0])
    fast = _evidence(frame, started=_CUTOFF + 1.0, received=_CUTOFF + 2.0)
    slow = _evidence(frame, started=_CUTOFF + 10.0, received=_CUTOFF + 20.0)

    assert fast.frame_hash == slow.frame_hash
    assert raw_frame_bundle_hash([fast]) == raw_frame_bundle_hash([slow])
    assert fast.as_dict() != slow.as_dict()


def test_same_available_frame_has_same_bundle_hash_after_failed_refresh() -> None:
    frame = _frame(turnover=[1000.0, 1100.0, 1200.0])
    fresh = _evidence(frame, started=_CUTOFF + 1.0, received=_CUTOFF + 2.0)
    fallback = fresh.with_cache_read(
        requested_as_of_ts=_CUTOFF,
        request_started_at=_CUTOFF + 10.0,
        received_at=_CUTOFF + 11.0,
        source_ts=_CUTOFF + 2.0,
        cache_age_sec=8.0,
        refresh_error_code="MexcOhlcvRequestError",
    )

    assert fallback.outcome == "stale"
    assert fresh.frame_hash == fallback.frame_hash
    assert raw_frame_bundle_hash([fresh]) == raw_frame_bundle_hash([fallback])


@pytest.mark.parametrize(
    "changed",
    [
        _frame(close_shift=0.0001),
        _frame(turnover=[1000.0, 1100.0, 1200.0]),
    ],
)
def test_frame_hash_changes_when_consumed_market_input_changes(changed) -> None:
    baseline = canonical_closed_frame_hash(
        _frame(),
        venue="mexc_contract",
        symbol="BTCUSDT",
        venue_symbol="BTC_USDT",
        timeframe="Min60",
        cutoff_ts=_CUTOFF,
    )
    assert baseline != canonical_closed_frame_hash(
        changed,
        venue="mexc_contract",
        symbol="BTCUSDT",
        venue_symbol="BTC_USDT",
        timeframe="Min60",
        cutoff_ts=_CUTOFF,
    )


def test_turnover_observed_zero_differs_from_missing_turnover() -> None:
    missing = _frame(turnover=[None, None, None])
    observed_zero = _frame(turnover=[0.0, 0.0, 0.0])
    kwargs = {
        "venue": "mexc_contract",
        "symbol": "BTCUSDT",
        "venue_symbol": "BTC_USDT",
        "timeframe": "Min60",
        "cutoff_ts": _CUTOFF,
    }
    assert canonical_closed_frame_hash(missing, **kwargs) != canonical_closed_frame_hash(
        observed_zero, **kwargs
    )


def test_frame_hash_binds_symbol_timeframe_cutoff_and_bar_timestamp() -> None:
    frame = _frame()
    common = {
        "venue": "mexc_contract",
        "symbol": "BTCUSDT",
        "venue_symbol": "BTC_USDT",
        "timeframe": "Min60",
        "cutoff_ts": _CUTOFF,
    }
    baseline = canonical_closed_frame_hash(frame, **common)

    # Aliases are canonicalized before persistence/hash; spelling is not market.
    assert canonical_closed_frame_hash(
        frame, **{**common, "timeframe": "60"}
    ) == baseline

    shifted = frame.copy(deep=True)
    shifted.index = shifted.index - pd.Timedelta(hours=1)
    assert canonical_closed_frame_hash(shifted, **common) != baseline
    assert canonical_closed_frame_hash(
        frame, **{**common, "symbol": "ETHUSDT", "venue_symbol": "ETH_USDT"}
    ) != baseline
    half_hour = frame.copy(deep=True)
    half_hour.index = pd.date_range(
        "2026-01-01T09:00:00Z", periods=len(frame), freq="30min"
    )
    assert canonical_closed_frame_hash(
        half_hour, **{**common, "timeframe": "Min30"}
    ) != baseline
    assert canonical_closed_frame_hash(
        frame, **{**common, "cutoff_ts": _CUTOFF + 3600.0}
    ) != baseline


def test_frame_hash_rejects_reordered_or_non_finite_required_rows() -> None:
    kwargs = {
        "venue": "mexc_contract",
        "symbol": "BTCUSDT",
        "venue_symbol": "BTC_USDT",
        "timeframe": "Min60",
        "cutoff_ts": _CUTOFF,
    }
    with pytest.raises(FrameProvenanceError, match="ordered_and_unique"):
        canonical_closed_frame_hash(_frame().iloc[::-1], **kwargs)
    invalid = _frame()
    invalid.iloc[-1, invalid.columns.get_loc("close")] = float("inf")
    with pytest.raises(FrameProvenanceError, match="finite_number"):
        canonical_closed_frame_hash(invalid, **kwargs)


def test_frame_hash_accepts_pandas_numpy_integer_and_float32_scalars() -> None:
    frame = pd.DataFrame(
        {
            "open": np.asarray([10, 11, 12], dtype=np.int64),
            "high": np.asarray([11, 12, 13], dtype=np.float32),
            "low": np.asarray([9, 10, 11], dtype=np.int32),
            "close": np.asarray([10, 11, 12], dtype=np.float32),
            "volume": np.asarray([100, 110, 120], dtype=np.int64),
            "turnover": np.asarray([1000, 1100, 1200], dtype=np.int64),
        },
        index=pd.date_range("2026-01-01T09:00:00Z", periods=3, freq="h"),
    )
    assert canonical_closed_frame_hash(
        frame,
        venue="mexc_contract",
        symbol="BTCUSDT",
        venue_symbol="BTC_USDT",
        timeframe="Min60",
        cutoff_ts=_CUTOFF,
    )


@pytest.mark.parametrize("invalid_turnover", ["corrupt", float("inf"), -1.0])
def test_explicit_invalid_turnover_is_not_laundered_into_missing(
    invalid_turnover,
) -> None:
    frame = _frame(turnover=[1000.0, 1100.0, invalid_turnover])
    with pytest.raises(FrameProvenanceError, match="turnover"):
        canonical_closed_frame_hash(
            frame,
            venue="mexc_contract",
            symbol="BTCUSDT",
            venue_symbol="BTC_USDT",
            timeframe="Min60",
            cutoff_ts=_CUTOFF,
        )


@pytest.mark.parametrize("case", ["unaligned", "gap", "negative_volume", "bad_ohlc"])
def test_frame_quality_invalid_vectors_cannot_be_called_fresh(case: str) -> None:
    frame = _frame(turnover=[1000.0, 1100.0, 1200.0])
    if case == "unaligned":
        frame.index = frame.index + pd.Timedelta(seconds=1)
    elif case == "gap":
        frame.index = pd.DatetimeIndex(
            [frame.index[0], frame.index[1], frame.index[2] + pd.Timedelta(hours=1)]
        )
    elif case == "negative_volume":
        frame.iloc[-1, frame.columns.get_loc("volume")] = -1.0
    else:
        frame.iloc[-1, frame.columns.get_loc("high")] = 11.0

    with pytest.raises(FrameProvenanceError):
        _evidence(frame, started=_CUTOFF + 1.0, received=_CUTOFF + 2.0)


def test_direct_and_cache_source_timing_must_be_causal() -> None:
    evidence = _evidence(
        _frame(turnover=[1000.0, 1100.0, 1200.0]),
        started=_CUTOFF + 1.0,
        received=_CUTOFF + 2.0,
    )
    with pytest.raises(FrameProvenanceError, match="direct_source_ts"):
        replace(evidence, source_ts=_CUTOFF + 0.5)
    with pytest.raises(FrameProvenanceError, match="request_started_at"):
        replace(evidence, request_started_at=_CUTOFF - 1.0)

    cached = evidence.with_cache_read(
        requested_as_of_ts=_CUTOFF,
        request_started_at=_CUTOFF + 3.0,
        received_at=_CUTOFF + 4.0,
        source_ts=_CUTOFF + 2.0,
        cache_age_sec=1.0,
    )
    with pytest.raises(FrameProvenanceError, match="cache_source_ts"):
        replace(
            cached,
            source_ts=_CUTOFF + 3.5,
            cache_age_sec=0.0,
        )


def test_evidence_round_trip_is_exact_and_frame_read_rechecks_hash() -> None:
    frame = _frame(turnover=[1000.0, 1100.0, 1200.0])
    evidence = _evidence(frame, started=_CUTOFF + 1.0, received=_CUTOFF + 2.0)
    assert parse_source_read_evidence(evidence.as_dict()) == evidence
    FrameRead(frame=frame.copy(deep=True), evidence=evidence)

    extra = deepcopy(evidence.as_dict())
    extra["unexpected"] = True
    with pytest.raises(FrameProvenanceError, match="schema_mismatch"):
        parse_source_read_evidence(extra)

    mutated = frame.copy(deep=True)
    mutated.iloc[-1, mutated.columns.get_loc("close")] += 0.1
    with pytest.raises(FrameProvenanceError, match="hash_mismatch"):
        FrameRead(frame=mutated, evidence=evidence)


def test_frame_read_no_rows_revalidates_empty_frame_contract() -> None:
    canonical_empty = pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"],
        index=pd.DatetimeIndex([], tz="UTC"),
    )
    evidence = SourceReadEvidenceV1.from_frame(
        canonical_empty,
        source="base_ohlcv",
        venue="mexc_contract",
        symbol="BTCUSDT",
        venue_symbol="BTC_USDT",
        timeframe="Min60",
        requested_as_of_ts=_CUTOFF,
        request_started_at=_CUTOFF + 1.0,
        received_at=_CUTOFF + 2.0,
    )
    FrameRead(frame=canonical_empty.copy(deep=True), evidence=evidence)

    malformed_empty_frames = [
        pd.DataFrame(),
        pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([]),
        ),
        pd.DataFrame(
            columns=["open", "high", "low", "close"],
            index=pd.DatetimeIndex([], tz="UTC"),
        ),
    ]
    for malformed in malformed_empty_frames:
        with pytest.raises(FrameProvenanceError):
            FrameRead(frame=malformed, evidence=evidence)


def test_timeframe_alias_is_canonical_before_persistence() -> None:
    frame = _frame(turnover=[1000.0, 1100.0, 1200.0])
    evidence = SourceReadEvidenceV1.from_frame(
        frame,
        source="base_ohlcv",
        venue="mexc_contract",
        symbol="BTCUSDT",
        venue_symbol="BTC_USDT",
        timeframe="60",
        requested_as_of_ts=_CUTOFF,
        request_started_at=_CUTOFF + 1.0,
        received_at=_CUTOFF + 2.0,
    )
    assert evidence.timeframe == "Min60"
    assert evidence.as_dict()["timeframe"] == "Min60"

    noncanonical = evidence.as_dict()
    noncanonical["timeframe"] = "60"
    with pytest.raises(FrameProvenanceError, match="not_canonical"):
        parse_source_read_evidence(noncanonical)


def test_lagging_frame_records_actual_data_through_not_requested_boundary() -> None:
    lagging = _frame().iloc[:2].copy()
    evidence = SourceReadEvidenceV1.from_frame(
        lagging,
        source="higher_timeframe_ohlcv",
        venue="mexc_contract",
        symbol="BTCUSDT",
        venue_symbol="BTC_USDT",
        timeframe="Min60",
        requested_as_of_ts=_CUTOFF,
        request_started_at=_CUTOFF + 1.0,
        received_at=_CUTOFF + 2.0,
    )
    assert evidence.outcome == "stale"
    assert evidence.missing_reason == "data_lag"
    assert evidence.data_through_ts == pd.Timestamp("2026-01-01T11:00:00Z").timestamp()
    assert evidence.data_through_ts < evidence.expected_closed_boundary_ts


class _Response:
    def __init__(self, *, payload=None, http_error=None, json_error=None):
        self.payload = payload
        self.http_error = http_error
        self.json_error = json_error

    def raise_for_status(self):
        if self.http_error is not None:
            raise self.http_error

    def json(self):
        if self.json_error is not None:
            raise self.json_error
        return self.payload


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (_Response(http_error=requests.HTTPError("bad status")), MexcOhlcvRequestError),
        (_Response(json_error=ValueError("bad json")), MexcOhlcvJsonError),
        (_Response(payload={"success": False}), MexcOhlcvApiError),
    ],
)
def test_strict_ohlcv_request_does_not_turn_http_json_or_api_errors_into_empty(
    response, expected
) -> None:
    client = MexcContractClient(max_retries=1)
    with patch.object(client._session, "get", return_value=response):
        with pytest.raises(expected):
            client.fetch_ohlcv("BTCUSDT", "Min60", 3)
    client.close()


def test_strict_ohlcv_transport_error_is_not_empty_no_data() -> None:
    client = MexcContractClient(max_retries=1)
    with patch.object(client._session, "get", side_effect=requests.Timeout("timeout")):
        with pytest.raises(MexcOhlcvRequestError):
            client.fetch_ohlcv("BTCUSDT", "Min60", 3)
    client.close()


def test_true_empty_ohlcv_is_distinct_from_malformed_payload() -> None:
    client = MexcContractClient(max_retries=1)
    empty = {
        "success": True,
        "data": {
            "time": [], "open": [], "high": [], "low": [],
            "close": [], "vol": [], "amount": [],
        },
    }
    with patch.object(client, "_request_public_ohlcv", return_value=empty):
        assert client.fetch_ohlcv("BTCUSDT", "Min60", 3).empty

    malformed = {"success": True, "data": {"time": []}}
    with patch.object(client, "_request_public_ohlcv", return_value=malformed):
        with pytest.raises(MexcOhlcvPayloadError):
            client.fetch_ohlcv("BTCUSDT", "Min60", 3)

    invalid_number = {
        "success": True,
        "data": {
            "time": [1_700_000_000],
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [float("inf")],
            "vol": [10.0],
            "amount": [15.0],
        },
    }
    with patch.object(client, "_request_public_ohlcv", return_value=invalid_number):
        with pytest.raises(MexcOhlcvPayloadError):
            client.fetch_ohlcv("BTCUSDT", "Min60", 3)
    client.close()
