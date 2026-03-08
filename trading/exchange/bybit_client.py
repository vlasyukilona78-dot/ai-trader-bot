from __future__ import annotations

from typing import Any

try:
    from bybit_client import BybitClient as _BybitClient
except Exception:
    _BybitClient = None


class BybitHttpClient:
    """Thin wrapper over repository BybitClient for V2 adapter usage."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        testnet: bool,
        dry_run: bool,
        recv_window: int = 5000,
        category: str = "linear",
    ):
        if _BybitClient is None:
            raise RuntimeError("bybit_client_dependency_missing")
        self._client = _BybitClient(
            api_key=api_key,
            api_secret=api_secret,
            sandbox=bool(testnet),
            dry_run=bool(dry_run),
            recv_window=recv_window,
            category=category,
        )

    def close(self):
        self._client.close()

    def request_public(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        return self._client._request("GET", path, params=params, private=False)

    def request_private(self, method: str, path: str, *, params: dict[str, Any] | None = None, body: dict[str, Any] | None = None):
        return self._client._request(method.upper(), path, params=params, private=True, json_body=body)

    def get_open_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        return self._client.get_open_positions(symbol=symbol)

    def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        return self._client.get_open_orders(symbol=symbol)

    def place_order_market(self, **kwargs):
        return self._client.place_order_market(**kwargs)

    def place_order_limit(self, **kwargs):
        return self._client.place_order_limit(**kwargs)

    def set_trading_stop(self, **kwargs):
        return self._client.set_trading_stop(**kwargs)

    def cancel_order(self, **kwargs):
        return self._client.cancel_order(**kwargs)

    def get_ticker_meta(self, symbol: str) -> dict[str, Any]:
        return self._client.get_ticker_meta(symbol)


    def get_account_info(self) -> dict[str, Any]:
        return self._client.get_account_info()
