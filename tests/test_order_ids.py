"""Client order IDs must be stable across process restarts.

The exchange rejects a duplicate orderLinkId, which is what stops a retried or
replayed command from opening a second position. That protection only works if
the same logical intent always produces the same ID. Python's built-in hash()
is salted per process, so an ID derived from it changes after every restart and
silently drops the guarantee.
"""

from __future__ import annotations

import subprocess
import sys
import unittest

from trading.execution.order_ids import MAX_ORDER_LINK_ID_LEN, deterministic_order_id


class DeterminismTests(unittest.TestCase):
    def test_same_inputs_give_same_id(self):
        first = deterministic_order_id("v2", "BTCUSDT|LONG_ENTRY", 0.5)
        second = deterministic_order_id("v2", "BTCUSDT|LONG_ENTRY", 0.5)

        self.assertEqual(first, second)

    def test_id_is_stable_in_a_fresh_interpreter(self):
        # A separate process gets a different hash seed. If the ID were derived
        # from hash(), this value would differ from the in-process one.
        here = deterministic_order_id("v2", "BTCUSDT|LONG_ENTRY", 0.5)

        script = (
            "from trading.execution.order_ids import deterministic_order_id;"
            "print(deterministic_order_id('v2', 'BTCUSDT|LONG_ENTRY', 0.5))"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertEqual(result.stdout.strip(), here)

    def test_different_quantity_gives_different_id(self):
        one = deterministic_order_id("v2", "BTCUSDT|LONG_ENTRY", 0.5)
        other = deterministic_order_id("v2", "BTCUSDT|LONG_ENTRY", 0.6)

        self.assertNotEqual(one, other)

    def test_different_prefix_gives_different_id(self):
        entry = deterministic_order_id("v2", "BTCUSDT|LONG_ENTRY", 0.5)
        exit_ = deterministic_order_id("v2-exit", "BTCUSDT|LONG_ENTRY", 0.5)

        self.assertNotEqual(entry, exit_)

    def test_float_parts_do_not_depend_on_repr_drift(self):
        # 0.5 and 0.50 are the same quantity and must not split the identity.
        self.assertEqual(
            deterministic_order_id("v2", "BTCUSDT", 0.5),
            deterministic_order_id("v2", "BTCUSDT", 0.50),
        )


class VenueConstraintTests(unittest.TestCase):
    def test_id_fits_the_venue_limit(self):
        long_key = "X" * 500
        order_id = deterministic_order_id("v2-recover", long_key, 12345.6789)

        self.assertLessEqual(len(order_id), MAX_ORDER_LINK_ID_LEN)

    def test_id_uses_only_characters_the_venue_accepts(self):
        order_id = deterministic_order_id("v2-exit", "BTC/USDT|SHORT_ENTRY", 1.25)

        self.assertRegex(order_id, r"^[A-Za-z0-9_-]+$")

    def test_prefix_stays_readable_in_the_id(self):
        order_id = deterministic_order_id("v2-recover", "BTCUSDT", 1.0)

        self.assertTrue(order_id.startswith("v2-recover-"))


class ValidationTests(unittest.TestCase):
    def test_rejects_empty_prefix(self):
        with self.assertRaises(ValueError):
            deterministic_order_id("", "BTCUSDT", 1.0)

    def test_rejects_prefix_that_leaves_no_room_for_the_digest(self):
        with self.assertRaises(ValueError):
            deterministic_order_id("p" * MAX_ORDER_LINK_ID_LEN, "BTCUSDT", 1.0)


if __name__ == "__main__":
    unittest.main()
