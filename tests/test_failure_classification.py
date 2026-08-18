"""A failed order call must be classified before anything is resent.

A rate-limit rejection never reached the matching engine, so resending it is
safe. A timeout means the response was lost, not that the order was refused —
the venue may hold a live order. Treating the two the same is what turns one
intent into two positions.
"""

from __future__ import annotations

import unittest

from trading.exchange.schemas import OrderResult
from trading.execution.failure_class import FailureClass, classify_failure


def _result(*, error: str = "", ret_code: int | None = None, ret_msg: str = "") -> OrderResult:
    raw: dict = {}
    if ret_code is not None:
        raw["retCode"] = ret_code
    if ret_msg:
        raw["retMsg"] = ret_msg
    return OrderResult(
        success=False,
        order_id="",
        order_link_id="",
        avg_price=0.0,
        filled_qty=0.0,
        status="",
        raw=raw,
        error=error or None,
    )


class RetryableTests(unittest.TestCase):
    def test_rate_limit_code_is_retryable(self):
        self.assertIs(classify_failure(_result(ret_code=10006)), FailureClass.RETRYABLE)

    def test_rate_limit_message_is_retryable(self):
        self.assertIs(
            classify_failure(_result(ret_msg="Too many visits, rate limit")),
            FailureClass.RETRYABLE,
        )


class UnknownOutcomeTests(unittest.TestCase):
    def test_timeout_is_unknown_not_retryable(self):
        self.assertIs(
            classify_failure(_result(error="Read timeout after 5s")),
            FailureClass.UNKNOWN_POSSIBLY_SENT,
        )

    def test_connection_reset_is_unknown(self):
        self.assertIs(
            classify_failure(_result(error="Connection aborted, connection reset by peer")),
            FailureClass.UNKNOWN_POSSIBLY_SENT,
        )

    def test_server_error_code_is_unknown(self):
        # The request reached the venue and the venue failed. Whether the order
        # was created is exactly what we do not know.
        self.assertIs(classify_failure(_result(ret_code=10016)), FailureClass.UNKNOWN_POSSIBLY_SENT)


class TerminalTests(unittest.TestCase):
    def test_insufficient_balance_is_terminal(self):
        self.assertIs(
            classify_failure(_result(ret_code=110007, ret_msg="ab not enough for new order")),
            FailureClass.TERMINAL,
        )

    def test_duplicate_order_link_id_is_terminal(self):
        # The venue already holds this intent; resending must not happen.
        self.assertIs(
            classify_failure(_result(ret_code=110072, ret_msg="orderLinkId exist")),
            FailureClass.TERMINAL,
        )

    def test_unrecognised_failure_defaults_to_terminal(self):
        self.assertIs(classify_failure(_result(ret_msg="something new")), FailureClass.TERMINAL)


class PolicyTests(unittest.TestCase):
    def test_only_retryable_permits_resending(self):
        self.assertTrue(FailureClass.RETRYABLE.permits_resend())
        self.assertFalse(FailureClass.UNKNOWN_POSSIBLY_SENT.permits_resend())
        self.assertFalse(FailureClass.TERMINAL.permits_resend())

    def test_only_unknown_requires_reconciliation(self):
        self.assertTrue(FailureClass.UNKNOWN_POSSIBLY_SENT.requires_reconciliation())
        self.assertFalse(FailureClass.RETRYABLE.requires_reconciliation())
        self.assertFalse(FailureClass.TERMINAL.requires_reconciliation())


if __name__ == "__main__":
    unittest.main()
