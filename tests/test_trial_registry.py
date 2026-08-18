"""Every parameter combination tried must be counted, including the failures.

Reporting the best of forty runs as though it were the only run is how a search
over noise becomes a discovery. The registry records an attempt before it is
evaluated, so a crash or a bad result consumes the budget exactly like a good
one, and the correction applied at the end knows the true denominator.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from ai.research.trials import (
    LockedTestPlan,
    RegistryCorruption,
    TrialAttempt,
    TrialFamily,
    TrialRegistry,
    holm_bonferroni,
)


def _family(family_id: str = "pump-fade-v1") -> TrialFamily:
    return TrialFamily(
        family_id=family_id,
        hypothesis="Exhaustion after a sharp pump precedes a mean reversion.",
        label_version="target_win@atr2.0-rr1.5",
        feature_set_id="default-25",
        search_space_id="grid-a",
        search_space_hash="deadbeef",
        primary_metric="net_expectancy_after_costs",
        secondary_metrics=("auc", "hit_rate"),
        cost_assumptions_version="unvalidated-2026-08",
        validation_procedure="chronological walk-forward, purge+embargo",
        acceptance_rule="net expectancy > 0 on the locked test at alpha 0.05",
    )


def _attempt(trial_id: str, family_id: str = "pump-fade-v1") -> TrialAttempt:
    return TrialAttempt(
        trial_id=trial_id,
        family_id=family_id,
        parameters_hash=f"params-{trial_id}",
        code_hash="code-abc",
        dataset_hash="data-xyz",
    )


class RegistryLifecycleTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "trials.jsonl"
        self.registry = TrialRegistry(self.path)

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_family_must_be_registered_before_any_attempt(self):
        with self.assertRaises(ValueError):
            self.registry.start_attempt(_attempt("t1"))

    def test_a_family_cannot_be_registered_twice(self):
        self.registry.register_family(_family())

        with self.assertRaises(ValueError):
            self.registry.register_family(_family())

    def test_a_trial_id_cannot_be_reused(self):
        self.registry.register_family(_family())
        self.registry.start_attempt(_attempt("t1"))

        with self.assertRaises(ValueError):
            self.registry.start_attempt(_attempt("t1"))

    def test_an_attempt_counts_as_soon_as_it_starts(self):
        self.registry.register_family(_family())
        self.registry.start_attempt(_attempt("t1"))

        self.assertEqual(self.registry.attempt_count("pump-fade-v1"), 1)

    def test_an_abandoned_attempt_still_counts(self):
        # A crashed run consumed a look at the data exactly like a finished one.
        self.registry.register_family(_family())
        self.registry.start_attempt(_attempt("t1"))
        self.registry.start_attempt(_attempt("t2"))
        self.registry.finish_attempt("t2", succeeded=False, metrics={"auc": "0.44"})

        self.assertEqual(self.registry.attempt_count("pump-fade-v1"), 2)

    def test_a_failed_attempt_counts(self):
        self.registry.register_family(_family())
        self.registry.start_attempt(_attempt("t1"))
        self.registry.finish_attempt("t1", succeeded=False, metrics={})

        self.assertEqual(self.registry.attempt_count("pump-fade-v1"), 1)

    def test_an_attempt_cannot_finish_twice(self):
        self.registry.register_family(_family())
        self.registry.start_attempt(_attempt("t1"))
        self.registry.finish_attempt("t1", succeeded=True, metrics={})

        with self.assertRaises(ValueError):
            self.registry.finish_attempt("t1", succeeded=True, metrics={})

    def test_counts_are_scoped_per_family(self):
        self.registry.register_family(_family("a"))
        self.registry.register_family(_family("b"))
        self.registry.start_attempt(_attempt("t1", "a"))
        self.registry.start_attempt(_attempt("t2", "a"))
        self.registry.start_attempt(_attempt("t3", "b"))

        self.assertEqual(self.registry.attempt_count("a"), 2)
        self.assertEqual(self.registry.attempt_count("b"), 1)

    def test_the_log_survives_reopening(self):
        self.registry.register_family(_family())
        self.registry.start_attempt(_attempt("t1"))

        reopened = TrialRegistry(self.path)

        self.assertEqual(reopened.attempt_count("pump-fade-v1"), 1)


class TamperEvidenceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "trials.jsonl"
        registry = TrialRegistry(self.path)
        registry.register_family(_family())
        registry.start_attempt(_attempt("t1"))
        registry.start_attempt(_attempt("t2"))

    def tearDown(self):
        self._tmp.cleanup()

    def test_deleting_a_record_is_detected(self):
        # Three records were written: the family, then t1, then t2. Dropping the
        # middle one is how an inconvenient attempt would disappear from the
        # denominator.
        lines = self.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 3)
        self.path.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")

        with self.assertRaises(RegistryCorruption):
            TrialRegistry(self.path).events()

    def test_editing_a_record_is_detected(self):
        lines = self.path.read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[1])
        record["payload"]["parameters_hash"] = "rewritten"
        lines[1] = json.dumps(record, sort_keys=True)
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with self.assertRaises(RegistryCorruption):
            TrialRegistry(self.path).events()


class LockedTestTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "trials.jsonl"
        self.registry = TrialRegistry(self.path)
        self.registry.register_family(_family())

    def tearDown(self):
        self._tmp.cleanup()

    def _plan(self) -> LockedTestPlan:
        return LockedTestPlan(
            locked_test_id="lt1",
            family_id="pump-fade-v1",
            dataset_hash="data-xyz",
            holdout_start="2026-06-01T00:00:00Z",
            holdout_end="2026-08-01T00:00:00Z",
            alpha="0.05",
            required_net_expectancy="0.0",
            maximum_drawdown="0.15",
            minimum_independent_episodes=30,
        )

    def test_a_plan_must_be_registered_before_it_is_opened(self):
        with self.assertRaises(ValueError):
            self.registry.open_locked_test("lt1")

    def test_opening_returns_the_frozen_thresholds(self):
        self.registry.register_locked_test(self._plan())

        opened = self.registry.open_locked_test("lt1")

        self.assertEqual(opened.alpha, "0.05")

    def test_a_locked_test_can_only_be_opened_once(self):
        self.registry.register_locked_test(self._plan())
        self.registry.open_locked_test("lt1")

        with self.assertRaises(ValueError):
            self.registry.open_locked_test("lt1")

    def test_no_further_attempts_after_the_holdout_was_viewed(self):
        self.registry.register_locked_test(self._plan())
        self.registry.open_locked_test("lt1")

        with self.assertRaises(ValueError) as ctx:
            self.registry.start_attempt(_attempt("t99"))

        self.assertIn("locked test", str(ctx.exception).lower())

    def test_alpha_must_be_a_probability(self):
        with self.assertRaises(ValueError):
            LockedTestPlan(
                locked_test_id="lt2",
                family_id="pump-fade-v1",
                dataset_hash="d",
                holdout_start="2026-06-01T00:00:00Z",
                holdout_end="2026-08-01T00:00:00Z",
                alpha="1.5",
                required_net_expectancy="0.0",
                maximum_drawdown="0.15",
                minimum_independent_episodes=30,
            )

    def test_the_holdout_interval_must_move_forward(self):
        with self.assertRaises(ValueError):
            LockedTestPlan(
                locked_test_id="lt3",
                family_id="pump-fade-v1",
                dataset_hash="d",
                holdout_start="2026-08-01T00:00:00Z",
                holdout_end="2026-06-01T00:00:00Z",
                alpha="0.05",
                required_net_expectancy="0.0",
                maximum_drawdown="0.15",
                minimum_independent_episodes=30,
            )


class MultipleTestingTests(unittest.TestCase):
    def test_a_lone_significant_result_survives(self):
        results = holm_bonferroni({"h1": Decimal("0.001")}, alpha=Decimal("0.05"))

        self.assertTrue(results["h1"].rejected)

    def test_the_same_p_value_fails_once_many_things_were_tried(self):
        many = {f"h{i}": Decimal("0.04") for i in range(40)}

        results = holm_bonferroni(many, alpha=Decimal("0.05"))

        self.assertFalse(any(r.rejected for r in results.values()))

    def test_step_down_stops_at_the_first_failure(self):
        results = holm_bonferroni(
            {"a": Decimal("0.001"), "b": Decimal("0.40"), "c": Decimal("0.002")},
            alpha=Decimal("0.05"),
        )

        self.assertTrue(results["a"].rejected)
        self.assertTrue(results["c"].rejected)
        self.assertFalse(results["b"].rejected)

    def test_adjusted_p_values_are_monotone(self):
        results = holm_bonferroni(
            {"a": Decimal("0.01"), "b": Decimal("0.02"), "c": Decimal("0.03")},
            alpha=Decimal("0.05"),
        )
        ordered = [results[k].adjusted_p_value for k in ("a", "b", "c")]

        self.assertEqual(ordered, sorted(ordered))

    def test_empty_input_is_refused(self):
        with self.assertRaises(ValueError):
            holm_bonferroni({}, alpha=Decimal("0.05"))


if __name__ == "__main__":
    unittest.main()
