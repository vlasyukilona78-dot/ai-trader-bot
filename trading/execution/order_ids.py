"""Deterministic client order identities.

Bybit rejects a reused ``orderLinkId``, and that rejection is what stops a
retried or replayed command from opening a second position. The guarantee only
holds if one logical intent always maps to one ID, including after a process
restart. ``hash()`` cannot be used here: CPython salts string hashing per
process, so an ID derived from it changes on every restart.
"""

from __future__ import annotations

import hashlib
import re

#: Bybit's documented maximum length for ``orderLinkId``.
MAX_ORDER_LINK_ID_LEN = 36

_SAFE_PREFIX = re.compile(r"[^A-Za-z0-9_-]")


def _canonical(part: object) -> str:
    """Render one identity component so equal values render identically."""

    if isinstance(part, bool):
        return "1" if part else "0"
    if isinstance(part, float):
        # 0.5 and 0.50 are the same quantity; normalise away repr differences.
        return f"{part:.12g}"
    return str(part)


def deterministic_order_id(prefix: str, *parts: object) -> str:
    """Build a stable, venue-safe client order ID.

    Args:
        prefix: Short readable tag kept at the front of the ID, such as
            ``"v2"`` or ``"v2-exit"``.
        *parts: Identity components. Equal components must describe the same
            logical intent, and nothing time-varying belongs here.

    Raises:
        ValueError: The prefix is empty or leaves no room for the digest.
    """

    clean_prefix = _SAFE_PREFIX.sub("", prefix)
    if not clean_prefix:
        raise ValueError("order id prefix must contain usable characters")

    digest_room = MAX_ORDER_LINK_ID_LEN - len(clean_prefix) - 1
    if digest_room < 8:
        raise ValueError(
            f"prefix {prefix!r} leaves {digest_room} characters for the digest; at least 8 are required"
        )

    material = "\x00".join(_canonical(part) for part in parts)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return f"{clean_prefix}-{digest[:digest_room]}"
