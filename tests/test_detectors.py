"""Market episodes get their own lifecycle, separate from the strategy using them.

Today each layer is a boolean recomputed from scratch on every bar, so the
strategy cannot tell "this setup just appeared" from "this setup has been
weakening for six bars". Giving an episode a state machine and an identity adds
the state that matters most: DECAYING, where an open position may still be
exited but a new one may not be opened.
"""

from __future__ import annotations

import unittest

from core.detectors import (
    DetectorSnapshot,
    EpisodeDetector,
    EpisodeState,
    QualityFlag,
)


def _detector(**overrides) -> EpisodeDetector:
    base = dict(
        kind="pump",
        version="v1",
        confirmations_required=2,
        decay_tolerance=2,
        cooldown_bars=3,
    )
    base.update(overrides)
    return EpisodeDetector(**base)


class LifecycleTests(unittest.TestCase):
    def test_a_fresh_detector_is_ready(self):
        self.assertIs(_detector().state, EpisodeState.READY)

    def test_the_first_observation_opens_a_candidate(self):
        detector = _detector()

        snapshot = detector.observe(fired=True, side="SHORT")

        self.assertIs(snapshot.state, EpisodeState.CANDIDATE)

    def test_repeated_evidence_confirms_the_episode(self):
        detector = _detector(confirmations_required=2)

        detector.observe(fired=True, side="SHORT")
        snapshot = detector.observe(fired=True, side="SHORT")

        self.assertIs(snapshot.state, EpisodeState.CONFIRMED)

    def test_evidence_stopping_moves_a_confirmed_episode_to_decaying(self):
        detector = _detector(confirmations_required=1)
        detector.observe(fired=True, side="SHORT")

        snapshot = detector.observe(fired=False, side=None)

        self.assertIs(snapshot.state, EpisodeState.DECAYING)

    def test_sustained_absence_ends_the_episode_in_cooldown(self):
        detector = _detector(confirmations_required=1, decay_tolerance=2)
        detector.observe(fired=True, side="SHORT")
        detector.observe(fired=False, side=None)
        snapshot = detector.observe(fired=False, side=None)

        self.assertIs(snapshot.state, EpisodeState.COOLDOWN)

    def test_returning_evidence_revives_a_decaying_episode(self):
        detector = _detector(confirmations_required=1, decay_tolerance=3)
        detector.observe(fired=True, side="SHORT")
        detector.observe(fired=False, side=None)

        snapshot = detector.observe(fired=True, side="SHORT")

        self.assertIs(snapshot.state, EpisodeState.CONFIRMED)

    def test_cooldown_expires_back_to_ready(self):
        detector = _detector(confirmations_required=1, decay_tolerance=1, cooldown_bars=2)
        detector.observe(fired=True, side="SHORT")
        detector.observe(fired=False, side=None)
        detector.observe(fired=False, side=None)
        snapshot = detector.observe(fired=False, side=None)

        self.assertIs(snapshot.state, EpisodeState.READY)

    def test_an_unconfirmed_candidate_that_disappears_resets(self):
        detector = _detector(confirmations_required=3)
        detector.observe(fired=True, side="SHORT")

        snapshot = detector.observe(fired=False, side=None)

        self.assertIs(snapshot.state, EpisodeState.READY)


class EntryPermissionTests(unittest.TestCase):
    def test_only_a_confirmed_episode_permits_a_new_entry(self):
        for state in EpisodeState:
            with self.subTest(state=state):
                self.assertEqual(
                    state.permits_new_entry(), state is EpisodeState.CONFIRMED
                )

    def test_a_decaying_episode_still_permits_an_exit(self):
        self.assertTrue(EpisodeState.DECAYING.permits_exit())
        self.assertTrue(EpisodeState.CONFIRMED.permits_exit())

    def test_a_decaying_episode_refuses_a_new_entry(self):
        detector = _detector(confirmations_required=1)
        detector.observe(fired=True, side="SHORT")
        snapshot = detector.observe(fired=False, side=None)

        self.assertFalse(snapshot.state.permits_new_entry())
        self.assertTrue(snapshot.state.permits_exit())


class EpisodeIdentityTests(unittest.TestCase):
    def test_an_episode_keeps_one_identity_while_it_lives(self):
        detector = _detector(confirmations_required=1)
        first = detector.observe(fired=True, side="SHORT")
        second = detector.observe(fired=True, side="SHORT")

        self.assertEqual(first.episode_id, second.episode_id)

    def test_a_new_episode_gets_a_new_identity(self):
        detector = _detector(confirmations_required=1, decay_tolerance=1, cooldown_bars=1)
        first = detector.observe(fired=True, side="SHORT").episode_id
        detector.observe(fired=False, side=None)
        detector.observe(fired=False, side=None)
        detector.observe(fired=False, side=None)
        second = detector.observe(fired=True, side="SHORT").episode_id

        self.assertNotEqual(first, second)

    def test_a_ready_detector_has_no_episode(self):
        self.assertIsNone(_detector().snapshot().episode_id)

    def test_the_identity_is_deterministic_across_detectors(self):
        one, two = _detector(), _detector()

        self.assertEqual(
            one.observe(fired=True, side="SHORT").episode_id,
            two.observe(fired=True, side="SHORT").episode_id,
        )

    def test_direction_change_starts_a_new_episode(self):
        detector = _detector(confirmations_required=1)
        first = detector.observe(fired=True, side="SHORT").episode_id

        second = detector.observe(fired=True, side="LONG").episode_id

        self.assertNotEqual(first, second)


class EvidenceTests(unittest.TestCase):
    def test_observations_are_counted(self):
        detector = _detector(confirmations_required=1)
        detector.observe(fired=True, side="SHORT")
        snapshot = detector.observe(fired=True, side="SHORT")

        self.assertEqual(snapshot.evidence_count, 2)

    def test_quality_flags_travel_with_the_snapshot(self):
        detector = _detector()

        snapshot = detector.observe(
            fired=True, side="SHORT", quality_flags=(QualityFlag.INSUFFICIENT_HISTORY,)
        )

        self.assertIn(QualityFlag.INSUFFICIENT_HISTORY, snapshot.quality_flags)

    def test_a_degraded_snapshot_cannot_authorise_an_entry(self):
        detector = _detector(confirmations_required=1)

        snapshot = detector.observe(
            fired=True, side="SHORT", quality_flags=(QualityFlag.STALE,)
        )

        self.assertFalse(snapshot.entry_eligible())

    def test_a_clean_confirmed_snapshot_authorises_an_entry(self):
        detector = _detector(confirmations_required=1)

        snapshot = detector.observe(fired=True, side="SHORT")

        self.assertTrue(snapshot.entry_eligible())

    def test_the_snapshot_records_the_detector_version(self):
        snapshot = _detector(version="v7").observe(fired=True, side="SHORT")

        self.assertEqual(snapshot.version, "v7")

    def test_snapshots_are_immutable(self):
        snapshot = _detector().observe(fired=True, side="SHORT")

        with self.assertRaises(Exception):
            snapshot.state = EpisodeState.CONFIRMED  # type: ignore[misc]


class ConfigurationTests(unittest.TestCase):
    def test_confirmations_required_must_be_positive(self):
        with self.assertRaises(ValueError):
            _detector(confirmations_required=0)

    def test_decay_tolerance_must_be_positive(self):
        with self.assertRaises(ValueError):
            _detector(decay_tolerance=0)

    def test_kind_and_version_are_required(self):
        with self.assertRaises(ValueError):
            _detector(kind="")
        with self.assertRaises(ValueError):
            _detector(version="")


if __name__ == "__main__":
    unittest.main()
