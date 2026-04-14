from __future__ import annotations

import os


def _first_nonempty(*names: str) -> str:
    for name in names:
        raw = os.getenv(name)
        if raw is None:
            continue
        value = str(raw).strip()
        if value:
            return value.rstrip("/")
    return ""


def resolve_public_http_base_url(*, testnet: bool = False) -> str:
    if testnet:
        return _first_nonempty("BYBIT_PUBLIC_BASE_URL_TESTNET", "BYBIT_TESTNET_PUBLIC_BASE_URL") or "https://api-testnet.bybit.com"
    return _first_nonempty("BYBIT_PUBLIC_BASE_URL", "BYBIT_MAINNET_PUBLIC_BASE_URL") or "https://api.bybit.com"


def resolve_private_http_base_url(*, testnet: bool = False, demo: bool = False) -> str:
    if testnet:
        return _first_nonempty("BYBIT_PRIVATE_BASE_URL_TESTNET", "BYBIT_TESTNET_PRIVATE_BASE_URL") or "https://api-testnet.bybit.com"
    if demo:
        return _first_nonempty("BYBIT_PRIVATE_BASE_URL_DEMO", "BYBIT_DEMO_PRIVATE_BASE_URL") or "https://api-demo.bybit.com"
    return _first_nonempty("BYBIT_PRIVATE_BASE_URL", "BYBIT_MAINNET_PRIVATE_BASE_URL") or "https://api.bybit.com"


def resolve_public_ws_url(*, testnet: bool = False) -> str:
    if testnet:
        return _first_nonempty("BYBIT_PUBLIC_WS_URL_TESTNET", "BYBIT_TESTNET_PUBLIC_WS_URL") or "wss://stream-testnet.bybit.com/v5/public/linear"
    return _first_nonempty("BYBIT_PUBLIC_WS_URL", "BYBIT_MAINNET_PUBLIC_WS_URL") or "wss://stream.bybit.com/v5/public/linear"


def resolve_private_ws_url(*, testnet: bool = False, demo: bool = False) -> str:
    if testnet:
        return _first_nonempty("BYBIT_PRIVATE_WS_URL_TESTNET", "BYBIT_TESTNET_PRIVATE_WS_URL") or "wss://stream-testnet.bybit.com/v5/private"
    if demo:
        return _first_nonempty("BYBIT_PRIVATE_WS_URL_DEMO", "BYBIT_DEMO_PRIVATE_WS_URL") or "wss://stream-demo.bybit.com/v5/private"
    return _first_nonempty("BYBIT_PRIVATE_WS_URL", "BYBIT_MAINNET_PRIVATE_WS_URL") or "wss://stream.bybit.com/v5/private"
