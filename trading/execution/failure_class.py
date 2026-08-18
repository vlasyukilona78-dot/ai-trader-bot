"""Classification of a failed order call before any resend decision.

Three outcomes are possible after a call fails, and they demand different
handling:

``RETRYABLE``
    The venue refused the request without acting on it, so the same command may
    be sent again.

``UNKNOWN_POSSIBLY_SENT``
    The response was lost or the venue failed mid-processing. A live order may
    exist. Resending is forbidden; the caller must reconcile against the venue.

``TERMINAL``
    The venue refused the command for a reason resending cannot change.

The default is ``TERMINAL``: an unrecognised failure must not silently become a
resend.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

#: Codes where the request was throttled and never reached matching.
_RATE_LIMIT_CODES = frozenset({10006})

#: Codes where the venue accepted the request and then failed, leaving the
#: order's existence unknown.
_AMBIGUOUS_CODES = frozenset({10016})

#: Preserved from the original engine behaviour: treated as a transient
#: condition that may be resent.
_LEGACY_RETRYABLE_CODES = frozenset({30084})

_RATE_LIMIT_TEXT = ("rate limit", "too many visits", "too many requests")

_AMBIGUOUS_TEXT = (
    "timeout",
    "timed out",
    "connection reset",
    "connection aborted",
    "remote end closed",
    "read error",
)


class FailureClass(Enum):
    """What may be done after a failed order call."""

    RETRYABLE = "retryable"
    UNKNOWN_POSSIBLY_SENT = "unknown_possibly_sent"
    TERMINAL = "terminal"

    def permits_resend(self) -> bool:
        """Whether the identical command may be sent again."""

        return self is FailureClass.RETRYABLE

    def requires_reconciliation(self) -> bool:
        """Whether venue state must be queried before deciding anything."""

        return self is FailureClass.UNKNOWN_POSSIBLY_SENT


def classify_failure(result: Any) -> FailureClass:
    """Classify a failed :class:`OrderResult` conservatively."""

    raw = result.raw if isinstance(getattr(result, "raw", None), dict) else {}
    ret_code = raw.get("retCode")
    text = f"{getattr(result, 'error', '') or ''} {raw.get('retMsg') or ''}".lower()

    if ret_code in _RATE_LIMIT_CODES or any(token in text for token in _RATE_LIMIT_TEXT):
        return FailureClass.RETRYABLE
    if ret_code in _AMBIGUOUS_CODES or any(token in text for token in _AMBIGUOUS_TEXT):
        return FailureClass.UNKNOWN_POSSIBLY_SENT
    if ret_code in _LEGACY_RETRYABLE_CODES:
        return FailureClass.RETRYABLE
    return FailureClass.TERMINAL
