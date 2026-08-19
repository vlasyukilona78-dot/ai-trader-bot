"""Append-only, hash-chained record of every parameter combination tried.

The number that makes a backtest result meaningful is not the result — it is
how many results were looked at before this one. Without that denominator a
search over noise is indistinguishable from a discovery.

Three rules make the denominator trustworthy:

* The hypothesis, search space and acceptance rule are frozen in a
  :class:`TrialFamily` before the first attempt runs.
* An attempt is written *before* it is evaluated, so a crash or a bad result
  consumes the budget exactly like a good one.
* The log is append-only and hash-chained, so removing an inconvenient run
  breaks the chain instead of quietly shrinking the denominator.

A :class:`LockedTestPlan` reserves a chronological holdout with its thresholds
fixed in advance. It opens exactly once; after that the family is closed and
further tuning needs a new family and a new holdout.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

SCHEMA_VERSION = 1


class RegistryCorruption(RuntimeError):
    """The log's hash chain does not verify."""


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip() or value.strip() != value:
        raise ValueError(f"{name} must be non-empty canonical text")


def _content_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class TrialFamily:
    """Everything frozen before the first attempt of one hypothesis."""

    family_id: str
    hypothesis: str
    label_version: str
    feature_set_id: str
    search_space_id: str
    search_space_hash: str
    primary_metric: str
    secondary_metrics: tuple[str, ...]
    cost_assumptions_version: str
    validation_procedure: str
    acceptance_rule: str

    def __post_init__(self) -> None:
        for name in (
            "family_id",
            "hypothesis",
            "label_version",
            "feature_set_id",
            "search_space_id",
            "search_space_hash",
            "primary_metric",
            "cost_assumptions_version",
            "validation_procedure",
            "acceptance_rule",
        ):
            _require_text(name, getattr(self, name))
        if not self.secondary_metrics:
            raise ValueError("secondary metrics must be pre-registered")
        if len(set(self.secondary_metrics)) != len(self.secondary_metrics):
            raise ValueError("secondary metrics contain duplicates")


@dataclass(frozen=True)
class TrialAttempt:
    """One actually attempted parameter combination."""

    trial_id: str
    family_id: str
    parameters_hash: str
    code_hash: str
    dataset_hash: str

    def __post_init__(self) -> None:
        for name in ("trial_id", "family_id", "parameters_hash", "code_hash", "dataset_hash"):
            _require_text(name, getattr(self, name))


@dataclass(frozen=True)
class LockedTestPlan:
    """A chronological holdout with its thresholds fixed before it is seen."""

    locked_test_id: str
    family_id: str
    dataset_hash: str
    holdout_start: str
    holdout_end: str
    alpha: str
    required_net_expectancy: str
    maximum_drawdown: str
    minimum_independent_episodes: int

    def __post_init__(self) -> None:
        for name in (
            "locked_test_id",
            "family_id",
            "dataset_hash",
            "holdout_start",
            "holdout_end",
            "alpha",
            "required_net_expectancy",
            "maximum_drawdown",
        ):
            _require_text(name, getattr(self, name))
        try:
            alpha = Decimal(self.alpha)
            Decimal(self.required_net_expectancy)
            Decimal(self.maximum_drawdown)
        except InvalidOperation as error:
            raise ValueError("thresholds must be canonical decimal text") from error
        if not Decimal(0) < alpha < Decimal(1):
            raise ValueError("alpha must be strictly between zero and one")
        if self.minimum_independent_episodes <= 0:
            raise ValueError("minimum independent episodes must be positive")
        start = datetime.fromisoformat(self.holdout_start.replace("Z", "+00:00"))
        end = datetime.fromisoformat(self.holdout_end.replace("Z", "+00:00"))
        if end <= start:
            raise ValueError("holdout interval must move forward in time")

    @property
    def plan_hash(self) -> str:
        return _content_hash(asdict(self))


@dataclass(frozen=True)
class RegistryEvent:
    """One verified record read back from the log."""

    sequence: int
    kind: str
    payload: dict
    record_hash: str
    previous_hash: str


class TrialRegistry:
    """Append-only hash-chained trial log backed by one JSONL file."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    # -- reading ----------------------------------------------------------

    def events(self) -> tuple[RegistryEvent, ...]:
        """Read every record, verifying the chain.

        Raises:
            RegistryCorruption: A record was edited, reordered or removed.
        """

        if not self._path.exists():
            return ()

        events: list[RegistryEvent] = []
        previous = ""
        with self._path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RegistryCorruption(f"record {line_no} is not valid JSON") from error
                if record.get("schema_version") != SCHEMA_VERSION:
                    raise RegistryCorruption(f"record {line_no} has an unsupported schema version")
                if record.get("sequence") != len(events):
                    raise RegistryCorruption(
                        f"record {line_no} is out of sequence; a record was removed or reordered"
                    )
                if record.get("previous_hash") != previous:
                    raise RegistryCorruption(f"record {line_no} does not chain to its predecessor")
                body = {k: v for k, v in record.items() if k != "record_hash"}
                if _content_hash(body) != record.get("record_hash"):
                    raise RegistryCorruption(f"record {line_no} was modified after it was written")
                previous = record["record_hash"]
                events.append(
                    RegistryEvent(
                        sequence=record["sequence"],
                        kind=record["kind"],
                        payload=record["payload"],
                        record_hash=record["record_hash"],
                        previous_hash=record["previous_hash"],
                    )
                )
        return tuple(events)

    def attempt_count(self, family_id: str | None = None) -> int:
        """Count every attempt that was *started*, whatever became of it."""

        return sum(
            1
            for event in self.events()
            if event.kind == "attempt_started"
            and (family_id is None or event.payload.get("family_id") == family_id)
        )

    # -- writing ----------------------------------------------------------

    def register_family(self, family: TrialFamily) -> RegistryEvent:
        events = self.events()
        if any(
            e.kind == "family_registered" and e.payload["family_id"] == family.family_id
            for e in events
        ):
            raise ValueError(f"family {family.family_id!r} is already registered")
        payload = asdict(family)
        payload["secondary_metrics"] = list(family.secondary_metrics)
        return self._append("family_registered", payload, events)

    def start_attempt(self, attempt: TrialAttempt) -> RegistryEvent:
        """Record an attempt before it runs, so a crash still consumes budget."""

        events = self.events()
        if not any(
            e.kind == "family_registered" and e.payload["family_id"] == attempt.family_id
            for e in events
        ):
            raise ValueError(f"family {attempt.family_id!r} must be registered first")
        if any(
            e.kind == "locked_test_opened" and e.payload["family_id"] == attempt.family_id
            for e in events
        ):
            raise ValueError(
                "the locked test for this family was already opened; "
                "further tuning requires a new family and a new holdout"
            )
        if any(
            e.kind == "attempt_started" and e.payload["trial_id"] == attempt.trial_id
            for e in events
        ):
            raise ValueError(f"trial {attempt.trial_id!r} already exists")
        return self._append("attempt_started", asdict(attempt), events)

    def finish_attempt(
        self, trial_id: str, *, succeeded: bool, metrics: dict[str, str]
    ) -> RegistryEvent:
        events = self.events()
        started = [
            e for e in events if e.kind == "attempt_started" and e.payload["trial_id"] == trial_id
        ]
        if not started:
            raise ValueError(f"trial {trial_id!r} was never started")
        if any(
            e.kind in ("attempt_succeeded", "attempt_failed")
            and e.payload["trial_id"] == trial_id
            for e in events
        ):
            raise ValueError(f"trial {trial_id!r} already reached a terminal state")
        kind = "attempt_succeeded" if succeeded else "attempt_failed"
        payload = {
            "trial_id": trial_id,
            "family_id": started[0].payload["family_id"],
            "metrics": dict(metrics),
        }
        return self._append(kind, payload, events)

    def register_locked_test(self, plan: LockedTestPlan) -> RegistryEvent:
        events = self.events()
        if not any(
            e.kind == "family_registered" and e.payload["family_id"] == plan.family_id
            for e in events
        ):
            raise ValueError(f"family {plan.family_id!r} must be registered first")
        if any(
            e.kind == "locked_test_registered"
            and e.payload["locked_test_id"] == plan.locked_test_id
            for e in events
        ):
            raise ValueError("locked test is already registered")
        if any(
            e.kind == "locked_test_registered" and e.payload["family_id"] == plan.family_id
            for e in events
        ):
            raise ValueError("this family already reserved a locked test")
        payload = asdict(plan)
        payload["plan_hash"] = plan.plan_hash
        return self._append("locked_test_registered", payload, events)

    def open_locked_test(self, locked_test_id: str) -> LockedTestPlan:
        """Consume the one look at the holdout, closing the family."""

        events = self.events()
        registered = [
            e
            for e in events
            if e.kind == "locked_test_registered"
            and e.payload["locked_test_id"] == locked_test_id
        ]
        if not registered:
            raise ValueError(f"locked test {locked_test_id!r} was never registered")
        if any(
            e.kind == "locked_test_opened" and e.payload["locked_test_id"] == locked_test_id
            for e in events
        ):
            raise ValueError(
                f"locked test {locked_test_id!r} was already opened and cannot be reused"
            )
        payload = dict(registered[0].payload)
        plan_fields = {
            k: v for k, v in payload.items() if k in LockedTestPlan.__dataclass_fields__
        }
        self._append(
            "locked_test_opened",
            {
                "locked_test_id": locked_test_id,
                "family_id": payload["family_id"],
                "plan_hash": payload["plan_hash"],
                "attempts_counted": self.attempt_count(payload["family_id"]),
            },
            events,
        )
        return LockedTestPlan(**plan_fields)

    def _append(self, kind: str, payload: dict, events: tuple[RegistryEvent, ...]) -> RegistryEvent:
        previous = events[-1].record_hash if events else ""
        body = {
            "schema_version": SCHEMA_VERSION,
            "sequence": len(events),
            "kind": kind,
            "payload": payload,
            "previous_hash": previous,
        }
        record = dict(body)
        record["record_hash"] = _content_hash(body)
        line = json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"

        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

        return RegistryEvent(
            sequence=record["sequence"],
            kind=kind,
            payload=payload,
            record_hash=record["record_hash"],
            previous_hash=previous,
        )


@dataclass(frozen=True)
class CorrectedHypothesis:
    """One hypothesis after family-wise error control."""

    hypothesis_id: str
    raw_p_value: Decimal
    adjusted_p_value: Decimal
    rejected: bool


def holm_bonferroni(
    p_values: dict[str, Decimal], *, alpha: Decimal
) -> dict[str, CorrectedHypothesis]:
    """Apply step-down Holm-Bonferroni control across a family of tests.

    Raises:
        ValueError: No hypotheses were supplied, or alpha is out of range.
    """

    if not p_values:
        raise ValueError("multiple-testing correction requires at least one hypothesis")
    if not Decimal(0) < alpha < Decimal(1):
        raise ValueError("alpha must be strictly between zero and one")

    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    total = len(ordered)
    running = Decimal(0)
    may_reject = True
    results: dict[str, CorrectedHypothesis] = {}

    for index, (name, raw) in enumerate(ordered):
        remaining = Decimal(total - index)
        rejected = may_reject and raw <= alpha / remaining
        if not rejected:
            may_reject = False
        running = max(running, min(Decimal(1), raw * remaining))
        results[name] = CorrectedHypothesis(
            hypothesis_id=name,
            raw_p_value=raw,
            adjusted_p_value=running,
            rejected=rejected,
        )
    return {name: results[name] for name in p_values}
