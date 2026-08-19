"""Every numeric assumption carries how it was established.

A guessed slippage constant and a measured one look identical in a report. If
the provenance is not attached to the value, a placeholder silently acquires
the authority of a measurement the moment it appears next to real numbers.
"""

from __future__ import annotations

import unittest

from ai.evidence import Assumption, EvidenceClass, live_gate


class EvidenceClassTests(unittest.TestCase):
    def test_only_empirically_validated_satisfies_the_live_gate(self):
        self.assertTrue(EvidenceClass.EMPIRICALLY_VALIDATED.permits_live())
        self.assertFalse(EvidenceClass.RESEARCH_DERIVED.permits_live())
        self.assertFalse(EvidenceClass.UNVALIDATED.permits_live())

    def test_unvalidated_is_the_weakest_class(self):
        self.assertLess(
            EvidenceClass.UNVALIDATED.strength,
            EvidenceClass.RESEARCH_DERIVED.strength,
        )
        self.assertLess(
            EvidenceClass.RESEARCH_DERIVED.strength,
            EvidenceClass.EMPIRICALLY_VALIDATED.strength,
        )


class AssumptionTests(unittest.TestCase):
    def test_an_assumption_keeps_its_value_and_provenance(self):
        fee = Assumption(
            name="fee_bps_per_side",
            value=5.5,
            evidence=EvidenceClass.UNVALIDATED,
            source="placeholder copied from the backtest default",
        )

        self.assertEqual(fee.value, 5.5)
        self.assertIs(fee.evidence, EvidenceClass.UNVALIDATED)

    def test_an_assumption_requires_a_source_note(self):
        with self.assertRaises(ValueError):
            Assumption(
                name="fee_bps_per_side",
                value=5.5,
                evidence=EvidenceClass.UNVALIDATED,
                source="",
            )

    def test_an_assumption_requires_a_name(self):
        with self.assertRaises(ValueError):
            Assumption(name="", value=1.0, evidence=EvidenceClass.UNVALIDATED, source="x")

    def test_repr_shows_the_evidence_class(self):
        fee = Assumption(
            name="fee_bps_per_side",
            value=5.5,
            evidence=EvidenceClass.UNVALIDATED,
            source="placeholder",
        )

        self.assertIn("UNVALIDATED", repr(fee))


class LiveGateTests(unittest.TestCase):
    def _assumption(self, name: str, evidence: EvidenceClass) -> Assumption:
        return Assumption(name=name, value=1.0, evidence=evidence, source="test")

    def test_gate_passes_when_every_assumption_is_measured(self):
        report = live_gate(
            [
                self._assumption("fee", EvidenceClass.EMPIRICALLY_VALIDATED),
                self._assumption("slippage", EvidenceClass.EMPIRICALLY_VALIDATED),
            ]
        )

        self.assertTrue(report.passed)
        self.assertEqual(report.blocking, ())

    def test_gate_names_every_assumption_that_blocks_it(self):
        report = live_gate(
            [
                self._assumption("fee", EvidenceClass.EMPIRICALLY_VALIDATED),
                self._assumption("slippage", EvidenceClass.UNVALIDATED),
                self._assumption("gap", EvidenceClass.RESEARCH_DERIVED),
            ]
        )

        self.assertFalse(report.passed)
        self.assertEqual(set(report.blocking), {"slippage", "gap"})

    def test_an_empty_set_of_assumptions_does_not_pass(self):
        # Nothing declared means nothing checked, which is not evidence of safety.
        self.assertFalse(live_gate([]).passed)


class ProjectDefaultsTests(unittest.TestCase):
    def test_shipped_cost_assumptions_are_declared_unvalidated(self):
        from ai.evidence import COST_ASSUMPTIONS

        self.assertTrue(COST_ASSUMPTIONS)
        for assumption in COST_ASSUMPTIONS.values():
            with self.subTest(name=assumption.name):
                self.assertFalse(
                    assumption.evidence.permits_live(),
                    "no cost constant has been measured against real fills yet",
                )


if __name__ == "__main__":
    unittest.main()
