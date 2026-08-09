from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import pytest

from trading.signals.lifecycle_contract import (
    CandidateArmV1,
    CandidateLifecycleEventV1,
    CandidateLifecycleState,
    CandidateSide,
    ConfirmationObservationV1,
    LIFECYCLE_CONTRACT_VERSION,
    LifecycleContractError,
    ProposalObservationBasis,
    ProposalObservationStatus,
    ProposalObservationV1,
    lifecycle_contract_hash,
    lifecycle_contract_payload,
)


_ARM_OPEN = 1_700_002_800.0
_INTERVAL = 3_600
_SPEC_CONTRACT_HASH = "1" * 64
_SPEC_INSTANCE_HASH = "2" * 64
_ARM_INPUT_HASH = "3" * 64
_CONFIRM_INPUT_HASH = "4" * 64

# Replacing this literal requires a new contract version. It is intentionally
# filled from the final declarative schema, not updated to make a drift pass.
_PINNED_CONTRACT_HASH = "012562854856dcb1145eb93066be00f2c68f291e0d94cd59c59b6a2bfef60c31"


def _arm(**overrides) -> CandidateArmV1:
    values = {
        "strategy_spec_version": "mexc_strategy_v2",
        "strategy_spec_contract_hash": _SPEC_CONTRACT_HASH,
        "strategy_spec_instance_hash": _SPEC_INSTANCE_HASH,
        "raw_input_bundle_hash": _ARM_INPUT_HASH,
        "symbol": "ALTUSDT",
        "side": CandidateSide.SHORT,
        "timeframe_seconds": _INTERVAL,
        "arm_bar_open_ts": _ARM_OPEN,
        "arm_candle_cutoff_ts": _ARM_OPEN + _INTERVAL,
        "armed_close": 108.8,
        "invalidate_level": 111.405,
        "arm_trace": {
            "strategy_model": "layered_table_5_softened",
            "failed_layer": "layer_confirmation_pending",
            "layers": {
                "layer1_pump_detection": {
                    "passed": True,
                    "side": "SHORT",
                    "details": {"move_pct": 0.088, "bars_since_peak": 1.0},
                },
                "layer4_fake_filter": {
                    "passed": True,
                    "details": {"funding_available": 1.0, "sentiment": 78.0},
                },
            },
        },
    }
    values.update(overrides)
    return CandidateArmV1(**values)


def _observation(
    arm: CandidateArmV1,
    *,
    state: CandidateLifecycleState = CandidateLifecycleState.WAITING,
    state_epoch: int = 1,
    elapsed_bars: int = 1,
    distinct_observation_count: int = 1,
    input_hash: str = _CONFIRM_INPUT_HASH,
    close: float = 108.0,
) -> ConfirmationObservationV1:
    open_ts = arm.arm_bar_open_ts + elapsed_bars * arm.timeframe_seconds
    if state is CandidateLifecycleState.SAME_BAR:
        open_ts = arm.arm_bar_open_ts
        elapsed_bars = 0
        distinct_observation_count = 0
        input_hash = arm.raw_input_bundle_hash
        close = arm.armed_close
    return ConfirmationObservationV1(
        candidate_id=arm.candidate_id,
        observation_input_bundle_hash=input_hash,
        state=state,
        state_epoch=state_epoch,
        timeframe_seconds=arm.timeframe_seconds,
        observation_bar_open_ts=open_ts,
        observation_candle_cutoff_ts=open_ts + arm.timeframe_seconds,
        observed_high=max(close + 0.5, arm.armed_close if state is CandidateLifecycleState.SAME_BAR else 0.0),
        observed_low=close - 0.5,
        observed_close=close,
        distinct_observation_count=distinct_observation_count,
        elapsed_bars=elapsed_bars,
    )


def _created_proposal(
    arm: CandidateArmV1,
    observation: ConfirmationObservationV1,
    **overrides,
) -> ProposalObservationV1:
    values = {
        "candidate_id": arm.candidate_id,
        "side": arm.side,
        "state_epoch": observation.state_epoch,
        "timeframe_seconds": arm.timeframe_seconds,
        "status": ProposalObservationStatus.CREATED,
        "basis": ProposalObservationBasis.CONFIRMATION,
        "confirmation_observation_id": observation.observation_id,
        "proposal_input_bundle_hash": observation.observation_input_bundle_hash,
        "reference_bar_open_ts": observation.observation_bar_open_ts,
        "reference_candle_cutoff_ts": observation.observation_candle_cutoff_ts,
        "decision_reference_price": 108.0,
        "stop_price": 111.405,
        "take_profit_price": 100.09,
        "details": {"realized_risk_reward": 2.32, "source": ["closed_bar", "arm"]},
        "execution_bound": False,
    }
    values.update(overrides)
    return ProposalObservationV1(**values)


def _bypass_proposal(
    arm: CandidateArmV1,
    *,
    status: ProposalObservationStatus = ProposalObservationStatus.CREATED,
    **overrides,
) -> ProposalObservationV1:
    values = {
        "candidate_id": arm.candidate_id,
        "side": arm.side,
        "state_epoch": 0,
        "timeframe_seconds": arm.timeframe_seconds,
        "status": status,
        "basis": ProposalObservationBasis.ARM_BYPASS,
        "confirmation_observation_id": None,
        "proposal_input_bundle_hash": arm.raw_input_bundle_hash,
        "reference_bar_open_ts": arm.arm_bar_open_ts,
        "reference_candle_cutoff_ts": arm.arm_candle_cutoff_ts,
        "decision_reference_price": 108.8,
        "stop_price": 111.405,
        "take_profit_price": 100.09,
        "details": {"confirmation_enabled": False},
        "execution_bound": False,
    }
    if status is ProposalObservationStatus.REJECTED:
        values.update(
            decision_reference_price=None,
            stop_price=None,
            take_profit_price=None,
            rejection_reason="layer5_stop_too_wide",
        )
    values.update(overrides)
    return ProposalObservationV1(**values)


def _confirmed_event() -> tuple[
    CandidateLifecycleEventV1,
    CandidateLifecycleEventV1,
    CandidateLifecycleEventV1,
]:
    arm = _arm()
    armed = CandidateLifecycleEventV1.armed(arm)
    waiting_observation = _observation(arm)
    waiting = CandidateLifecycleEventV1.transition(
        armed,
        confirmation=waiting_observation,
    )
    confirmation = _observation(
        arm,
        state=CandidateLifecycleState.CONFIRMED,
        state_epoch=2,
        elapsed_bars=2,
        distinct_observation_count=2,
        close=107.7,
    )
    confirmed = CandidateLifecycleEventV1.transition(
        waiting,
        confirmation=confirmation,
        proposal=_created_proposal(
            arm,
            confirmation,
            decision_reference_price=107.7,
        ),
    )
    return armed, waiting, confirmed


def test_contract_hash_is_pinned_and_declares_no_wall_clock_identity() -> None:
    assert LIFECYCLE_CONTRACT_VERSION == "candidate_lifecycle_v1"
    assert lifecycle_contract_hash() == _PINNED_CONTRACT_HASH
    payload = lifecycle_contract_payload()
    assert payload["contract_version"] == LIFECYCLE_CONTRACT_VERSION
    excluded = payload["semantic_identity"]["excludes"]
    assert "decision_completed_ts" in excluded
    assert "proposal_available_ts" in excluded
    assert CandidateLifecycleState.BYPASSED.value == "bypassed"
    assert ProposalObservationBasis.ARM_BYPASS.value == "arm_bypass"


def test_arm_is_deeply_immutable_and_round_trips_canonically() -> None:
    source_trace = {
        "layers": {"layer1": {"passed": True, "values": [1.0, 2.0]}},
    }
    arm = _arm(arm_trace=source_trace)
    encoded = arm.as_dict()

    source_trace["layers"]["layer1"]["values"].append(999.0)
    assert arm.as_dict() == encoded
    with pytest.raises(TypeError):
        arm.arm_trace["new"] = "value"  # type: ignore[index]
    with pytest.raises(TypeError):
        arm.arm_trace["layers"]["layer1"]["passed"] = False  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        arm.symbol = "OTHERUSDT"  # type: ignore[misc]

    rebuilt = CandidateArmV1.from_dict(encoded)
    assert rebuilt == arm
    assert rebuilt.as_dict() == encoded


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"timeframe_seconds": True}, "timeframe_seconds_must_be_an_integer"),
        ({"armed_close": float("nan")}, "armed_close_must_be_finite"),
        ({"armed_close": True}, "armed_close_must_be_a_number"),
        ({"raw_input_bundle_hash": "not-a-hash"}, "raw_input_bundle_hash"),
        ({"arm_candle_cutoff_ts": _ARM_OPEN + 60.0}, "bar_duration_mismatch"),
        ({"arm_bar_open_ts": _ARM_OPEN + 1.0}, "bar_duration_mismatch"),
        ({"invalidate_level": 100.0}, "short_invalidate_level"),
        ({"side": "SHORT"}, "side_must_be_candidate_side"),
        ({"arm_trace": {"bad": float("inf")}}, "non_finite"),
    ],
)
def test_arm_rejects_ambiguous_or_non_finite_inputs(override, message) -> None:
    with pytest.raises(LifecycleContractError, match=message):
        _arm(**override)


def test_candidate_id_binds_every_semantic_arm_namespace() -> None:
    baseline = _arm()
    variants = [
        _arm(strategy_spec_version="mexc_strategy_v3"),
        _arm(strategy_spec_contract_hash="a" * 64),
        _arm(strategy_spec_instance_hash="b" * 64),
        _arm(raw_input_bundle_hash="c" * 64),
        _arm(symbol="OTHERUSDT"),
        _arm(
            side=CandidateSide.LONG,
            armed_close=100.0,
            invalidate_level=98.0,
        ),
        _arm(
            arm_bar_open_ts=_ARM_OPEN + _INTERVAL,
            arm_candle_cutoff_ts=_ARM_OPEN + 2 * _INTERVAL,
        ),
        _arm(armed_close=108.7),
        _arm(arm_trace={"different": {"causal": 1.0}}),
    ]

    assert _arm().candidate_id == baseline.candidate_id
    assert _arm().arm_trace_hash == baseline.arm_trace_hash
    assert len({baseline.candidate_id, *(item.candidate_id for item in variants)}) == len(variants) + 1


def test_confirmation_keeps_observation_count_distinct_from_elapsed_bars() -> None:
    arm = _arm()
    jumped = _observation(
        arm,
        elapsed_bars=3,
        distinct_observation_count=1,
    )
    event = CandidateLifecycleEventV1.transition(
        CandidateLifecycleEventV1.armed(arm),
        confirmation=jumped,
    )

    assert event.confirmation is jumped
    assert jumped.distinct_observation_count == 1
    assert jumped.elapsed_bars == 3
    assert jumped.observation_bar_open_ts == arm.arm_bar_open_ts + 3 * _INTERVAL


def test_confirmation_semantic_id_binds_input_state_bar_prices_and_both_counts() -> None:
    arm = _arm()
    baseline = _observation(arm, elapsed_bars=3, distinct_observation_count=1)
    variants = [
        replace(baseline, observation_input_bundle_hash="a" * 64),
        replace(baseline, state=CandidateLifecycleState.CONFIRMED),
        replace(baseline, state_epoch=2),
        replace(
            baseline,
            observation_bar_open_ts=baseline.observation_bar_open_ts + _INTERVAL,
            observation_candle_cutoff_ts=baseline.observation_candle_cutoff_ts + _INTERVAL,
            elapsed_bars=4,
        ),
        replace(baseline, observed_close=107.9),
        replace(baseline, distinct_observation_count=2),
    ]

    assert len(
        {baseline.observation_id, *(item.observation_id for item in variants)}
    ) == len(variants) + 1


@pytest.mark.parametrize(
    "change",
    [
        {"observed_high": float("nan")},
        {"observed_close": True},
        {"state_epoch": True},
        {"distinct_observation_count": True},
        {"distinct_observation_count": 2, "elapsed_bars": 1},
    ],
)
def test_confirmation_rejects_invalid_numbers_counts_and_types(change) -> None:
    arm = _arm()
    values = {
        "candidate_id": arm.candidate_id,
        "observation_input_bundle_hash": _CONFIRM_INPUT_HASH,
        "state": CandidateLifecycleState.WAITING,
        "state_epoch": 1,
        "timeframe_seconds": _INTERVAL,
        "observation_bar_open_ts": _ARM_OPEN + _INTERVAL,
        "observation_candle_cutoff_ts": _ARM_OPEN + 2 * _INTERVAL,
        "observed_high": 109.0,
        "observed_low": 107.0,
        "observed_close": 108.0,
        "distinct_observation_count": 1,
        "elapsed_bars": 1,
    }
    values.update(change)
    with pytest.raises(LifecycleContractError):
        ConfirmationObservationV1(**values)


def test_same_bar_requires_same_candidate_input_bundle_and_cutoff() -> None:
    arm = _arm()
    same = _observation(arm, state=CandidateLifecycleState.SAME_BAR)
    armed = CandidateLifecycleEventV1.armed(arm)
    repeated = CandidateLifecycleEventV1.transition(armed, confirmation=same)

    assert repeated.state is CandidateLifecycleState.SAME_BAR
    assert repeated.arm is armed.arm

    wrong_bundle = replace(same, observation_input_bundle_hash="d" * 64)
    with pytest.raises(LifecycleContractError, match="same_bar_input_bundle_mismatch"):
        CandidateLifecycleEventV1.transition(armed, confirmation=wrong_bundle)


def test_same_bar_after_waiting_repeats_predecessor_identity_and_counts() -> None:
    arm = _arm()
    armed = CandidateLifecycleEventV1.armed(arm)
    first = _observation(arm, state=CandidateLifecycleState.WAITING)
    waiting = CandidateLifecycleEventV1.transition(armed, confirmation=first)
    repeated_observation = replace(
        first,
        state=CandidateLifecycleState.SAME_BAR,
        state_epoch=2,
    )
    repeated = CandidateLifecycleEventV1.transition(
        waiting,
        confirmation=repeated_observation,
    )
    next_observation = _observation(
        arm,
        state=CandidateLifecycleState.WAITING,
        state_epoch=3,
        elapsed_bars=2,
        distinct_observation_count=2,
    )
    next_waiting = CandidateLifecycleEventV1.transition(
        repeated,
        confirmation=next_observation,
    )

    assert repeated.state is CandidateLifecycleState.SAME_BAR
    assert repeated.confirmation is not None
    assert repeated.confirmation.observation_bar_open_ts == first.observation_bar_open_ts
    assert repeated.confirmation.observation_input_bundle_hash == first.observation_input_bundle_hash
    assert repeated.confirmation.distinct_observation_count == 1
    assert repeated.confirmation.elapsed_bars == 1
    assert next_waiting.state is CandidateLifecycleState.WAITING
    assert next_waiting.confirmation is not None
    assert next_waiting.confirmation.distinct_observation_count == 2
    assert next_waiting.confirmation.elapsed_bars == 2

    wrong_count = replace(repeated_observation, distinct_observation_count=0)
    with pytest.raises(LifecycleContractError, match="same_bar_distinct_count_mismatch"):
        CandidateLifecycleEventV1.transition(waiting, confirmation=wrong_count)


def test_arm_wait_confirm_chain_preserves_arm_and_links_every_transition() -> None:
    armed, waiting, confirmed = _confirmed_event()

    armed.validate_successor(waiting)
    waiting.validate_successor(confirmed)
    assert armed.state_epoch == 0
    assert waiting.state_epoch == 1
    assert confirmed.state_epoch == 2
    assert waiting.previous_event_id == armed.event_id
    assert confirmed.previous_event_id == waiting.event_id
    assert confirmed.arm is armed.arm
    assert confirmed.arm.as_dict() == armed.arm.as_dict()
    assert confirmed.confirmation is not None
    assert confirmed.proposal.confirmation_observation_id == confirmed.confirmation.observation_id
    assert confirmed.proposal.status is ProposalObservationStatus.CREATED
    assert confirmed.proposal.execution_bound is False


def test_confirmation_disabled_is_initial_bypassed_without_fake_observation() -> None:
    arm = _arm()
    proposal = _bypass_proposal(arm)
    bypassed = CandidateLifecycleEventV1.bypassed(arm, proposal=proposal)

    assert bypassed.state is CandidateLifecycleState.BYPASSED
    assert bypassed.state_epoch == 0
    assert bypassed.previous_event_id is None
    assert bypassed.previous_state is None
    assert bypassed.confirmation is None
    assert bypassed.proposal.status is ProposalObservationStatus.CREATED
    assert bypassed.proposal.basis is ProposalObservationBasis.ARM_BYPASS
    assert bypassed.proposal.confirmation_observation_id is None
    assert bypassed.proposal.reference_bar_open_ts == arm.arm_bar_open_ts
    assert bypassed.proposal.reference_candle_cutoff_ts == arm.arm_candle_cutoff_ts
    assert bypassed.proposal.execution_bound is False
    assert CandidateLifecycleEventV1.from_dict(bypassed.as_dict()) == bypassed


def test_bypassed_may_record_rejection_without_becoming_confirmation() -> None:
    arm = _arm()
    rejected = _bypass_proposal(
        arm,
        status=ProposalObservationStatus.REJECTED,
    )
    bypassed = CandidateLifecycleEventV1.bypassed(arm, proposal=rejected)

    assert bypassed.confirmation is None
    assert bypassed.proposal.status is ProposalObservationStatus.REJECTED
    assert bypassed.proposal.rejection_reason == "layer5_stop_too_wide"
    assert bypassed.proposal.decision_reference_price is None


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"proposal_input_bundle_hash": "d" * 64}, "input_bundle_mismatch"),
        (
            {
                "reference_bar_open_ts": _ARM_OPEN + _INTERVAL,
                "reference_candle_cutoff_ts": _ARM_OPEN + 2 * _INTERVAL,
            },
            "reference_bar_mismatch",
        ),
        ({"state_epoch": 1}, "arm_bypass_proposal_requires_epoch_zero"),
        ({"confirmation_observation_id": "e" * 64}, "must_not_link_confirmation"),
    ],
)
def test_bypassed_proposal_fails_closed_outside_exact_arm_basis(change, message) -> None:
    arm = _arm()
    if "state_epoch" in change or "confirmation_observation_id" in change:
        with pytest.raises(LifecycleContractError, match=message):
            _bypass_proposal(arm, **change)
        return
    proposal = _bypass_proposal(arm, **change)
    with pytest.raises(LifecycleContractError, match=message):
        CandidateLifecycleEventV1.bypassed(arm, proposal=proposal)


def test_bypassed_is_not_a_confirmation_observation_and_is_terminal() -> None:
    arm = _arm()
    with pytest.raises(LifecycleContractError, match="follow_up_state"):
        _observation(arm, state=CandidateLifecycleState.BYPASSED)

    bypassed = CandidateLifecycleEventV1.bypassed(
        arm,
        proposal=_bypass_proposal(arm),
    )
    successor_observation = _observation(arm)
    with pytest.raises(LifecycleContractError, match="terminal_event_has_no_successor"):
        CandidateLifecycleEventV1.transition(
            bypassed,
            confirmation=successor_observation,
        )


def test_events_and_all_nested_evidence_round_trip_with_exact_keys() -> None:
    _, _, confirmed = _confirmed_event()
    payload = confirmed.as_dict()
    rebuilt = CandidateLifecycleEventV1.from_dict(payload)

    assert rebuilt == confirmed
    assert rebuilt.as_dict() == payload

    for nested_key, parser in (
        ("arm", CandidateArmV1.from_dict),
        ("confirmation", ConfirmationObservationV1.from_dict),
        ("proposal", ProposalObservationV1.from_dict),
    ):
        tampered = deepcopy(payload[nested_key])
        tampered["unknown"] = 1
        with pytest.raises(LifecycleContractError, match="unknown_keys"):
            parser(tampered)

    extra_event = deepcopy(payload)
    extra_event["unknown"] = 1
    with pytest.raises(LifecycleContractError, match="unknown_keys"):
        CandidateLifecycleEventV1.from_dict(extra_event)

    missing_event = deepcopy(payload)
    del missing_event["previous_event_id"]
    with pytest.raises(LifecycleContractError, match="missing_keys"):
        CandidateLifecycleEventV1.from_dict(missing_event)


@pytest.mark.parametrize(
    ("nested", "identity_key"),
    [
        ("arm", "candidate_id"),
        ("confirmation", "observation_id"),
        ("proposal", "proposal_observation_id"),
    ],
)
def test_parsers_reject_rehashed_content_with_stale_semantic_ids(nested, identity_key) -> None:
    _, _, confirmed = _confirmed_event()
    payload = confirmed.as_dict()
    payload[nested][identity_key] = "f" * 64
    with pytest.raises(LifecycleContractError, match="mismatch"):
        CandidateLifecycleEventV1.from_dict(payload)


def test_transition_rejects_wrong_candidate_or_state_epoch() -> None:
    arm = _arm()
    armed = CandidateLifecycleEventV1.armed(arm)
    other = _arm(symbol="OTHERUSDT")

    wrong_candidate = _observation(other)
    with pytest.raises(LifecycleContractError, match="transition_candidate_id_mismatch"):
        CandidateLifecycleEventV1.transition(armed, confirmation=wrong_candidate)

    wrong_epoch = _observation(arm, state_epoch=2)
    with pytest.raises(LifecycleContractError, match="transition_state_epoch_mismatch"):
        CandidateLifecycleEventV1.transition(armed, confirmation=wrong_epoch)


def test_confirmed_event_requires_same_candidate_epoch_and_observation_link() -> None:
    arm = _arm()
    armed = CandidateLifecycleEventV1.armed(arm)
    confirmation = _observation(
        arm,
        state=CandidateLifecycleState.CONFIRMED,
    )
    proposal = _created_proposal(arm, confirmation)

    wrong_epoch = replace(proposal, state_epoch=2)
    with pytest.raises(LifecycleContractError, match="proposal_state_epoch_mismatch"):
        CandidateLifecycleEventV1.transition(
            armed,
            confirmation=confirmation,
            proposal=wrong_epoch,
        )

    wrong_link = replace(proposal, confirmation_observation_id="e" * 64)
    with pytest.raises(LifecycleContractError, match="proposal_confirmation_link_mismatch"):
        CandidateLifecycleEventV1.transition(
            armed,
            confirmation=confirmation,
            proposal=wrong_link,
        )


def test_confirmed_can_be_created_or_rejected_but_not_not_evaluated() -> None:
    arm = _arm()
    armed = CandidateLifecycleEventV1.armed(arm)
    confirmation = _observation(
        arm,
        state=CandidateLifecycleState.CONFIRMED,
    )

    with pytest.raises(LifecycleContractError, match="confirmed_event_requires_proposal_outcome"):
        CandidateLifecycleEventV1.transition(armed, confirmation=confirmation)

    rejected = ProposalObservationV1(
        candidate_id=arm.candidate_id,
        side=arm.side,
        state_epoch=1,
        timeframe_seconds=arm.timeframe_seconds,
        status=ProposalObservationStatus.REJECTED,
        basis=ProposalObservationBasis.CONFIRMATION,
        confirmation_observation_id=confirmation.observation_id,
        proposal_input_bundle_hash=confirmation.observation_input_bundle_hash,
        reference_bar_open_ts=confirmation.observation_bar_open_ts,
        reference_candle_cutoff_ts=confirmation.observation_candle_cutoff_ts,
        rejection_reason="layer5_stop_too_wide",
        details={"stop_distance_pct": 0.031},
    )
    event = CandidateLifecycleEventV1.transition(
        armed,
        confirmation=confirmation,
        proposal=rejected,
    )
    assert event.state is CandidateLifecycleState.CONFIRMED
    assert event.proposal.status is ProposalObservationStatus.REJECTED
    assert event.proposal.execution_bound is False
    assert event.proposal.decision_reference_price is None


def test_non_confirmed_and_execution_bound_proposals_fail_closed() -> None:
    arm = _arm()
    waiting = _observation(arm)
    created = _created_proposal(arm, waiting)
    with pytest.raises(LifecycleContractError, match="non_confirmed_event_must_not_evaluate_proposal"):
        CandidateLifecycleEventV1.transition(
            CandidateLifecycleEventV1.armed(arm),
            confirmation=waiting,
            proposal=created,
        )

    confirmation = _observation(arm, state=CandidateLifecycleState.CONFIRMED)
    with pytest.raises(LifecycleContractError, match="must_not_be_execution_bound"):
        _created_proposal(arm, confirmation, execution_bound=True)
    with pytest.raises(LifecycleContractError, match="execution_bound_must_be_boolean"):
        _created_proposal(arm, confirmation, execution_bound=0)


def test_created_proposal_geometry_and_semantic_id_are_strict() -> None:
    arm = _arm()
    confirmation = _observation(arm, state=CandidateLifecycleState.CONFIRMED)
    proposal = _created_proposal(arm, confirmation)
    changed = _created_proposal(arm, confirmation, take_profit_price=99.0)

    assert proposal.proposal_observation_id != changed.proposal_observation_id
    with pytest.raises(LifecycleContractError, match="short_proposal_levels_are_invalid"):
        _created_proposal(arm, confirmation, stop_price=107.0)
    with pytest.raises(LifecycleContractError, match="must_be_a_number"):
        _created_proposal(arm, confirmation, stop_price=True)
    with pytest.raises(LifecycleContractError, match="must_be_finite"):
        _created_proposal(arm, confirmation, take_profit_price=float("nan"))


def test_terminal_event_has_no_successor() -> None:
    _, _, confirmed = _confirmed_event()
    next_observation = _observation(
        confirmed.arm,
        state=CandidateLifecycleState.WAITING,
        state_epoch=confirmed.state_epoch + 1,
        elapsed_bars=3,
        distinct_observation_count=3,
    )

    with pytest.raises(LifecycleContractError, match="terminal_event_has_no_successor"):
        CandidateLifecycleEventV1.transition(
            confirmed,
            confirmation=next_observation,
        )


def test_event_ids_bind_transition_links_but_not_any_processing_clock() -> None:
    armed, waiting, confirmed = _confirmed_event()
    rebuilt = CandidateLifecycleEventV1.from_dict(confirmed.as_dict())

    assert rebuilt.event_id == confirmed.event_id
    assert all(
        key not in confirmed.as_dict()
        for key in (
            "request_started_at",
            "received_at",
            "decision_completed_ts",
            "proposal_available_ts",
            "created_at",
        )
    )
    assert len({armed.event_id, waiting.event_id, confirmed.event_id}) == 3
