from __future__ import annotations

import importlib
import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


def _import_bybit_client():
    try:
        return importlib.import_module("bybit_client")
    except ModuleNotFoundError as exc:
        if str(getattr(exc, "name", "")) != "requests":
            raise

        fake_requests = types.ModuleType("requests")

        class _DummySession:
            def __init__(self):
                self.headers = {}

            def request(self, *args, **kwargs):
                raise RuntimeError("dummy_session_request_called")

            def close(self):
                return None

        class _Timeout(Exception):
            pass

        class _RequestException(Exception):
            pass

        fake_requests.Session = _DummySession
        fake_requests.exceptions = types.SimpleNamespace(
            Timeout=_Timeout,
            RequestException=_RequestException,
        )
        sys.modules["requests"] = fake_requests
        return importlib.import_module("bybit_client")


_BybitModule = _import_bybit_client()
BybitClient = _BybitModule.BybitClient


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class BybitPostSigningV2Tests(unittest.TestCase):
    def _make_client(self) -> BybitClient:
        client = BybitClient(
            api_key="test_key",
            api_secret="test_secret",
            sandbox=True,
            dry_run=False,
            recv_window=5000,
        )
        client._sess.request = MagicMock(return_value=_FakeResponse({"retCode": 0, "retMsg": "OK", "result": {}}))
        return client

    def test_private_post_signs_and_sends_identical_body(self):
        client = self._make_client()
        try:
            captured: dict[str, str] = {}
            real_sign = client._sign

            def _record_sign(timestamp: str, payload: str) -> str:
                captured["payload"] = payload
                return real_sign(timestamp, payload)

            client._sign = _record_sign  # type: ignore[method-assign]

            body = {"category": "linear", "symbol": "BTCUSDT", "qty": "0.01", "orderType": "Market"}
            with patch("bybit_client.time.time", return_value=1700000000.0):
                _ = client._request("POST", "/v5/order/create", private=True, json_body=body)

            kwargs = client._sess.request.call_args.kwargs
            sent_body = kwargs.get("data")
            self.assertIsInstance(sent_body, str)
            self.assertNotIn("json", kwargs)
            self.assertEqual(captured.get("payload"), sent_body)
            self.assertEqual(sent_body, json.dumps(body, separators=(",", ":"), ensure_ascii=False))
        finally:
            client.close()

    def test_order_creation_path_uses_same_body_for_sign_and_wire(self):
        client = self._make_client()
        try:
            captured: dict[str, str] = {}
            real_sign = client._sign

            def _record_sign(timestamp: str, payload: str) -> str:
                captured["payload"] = payload
                return real_sign(timestamp, payload)

            client._sign = _record_sign  # type: ignore[method-assign]

            with patch("bybit_client.time.time", return_value=1700000000.0):
                _ = client.place_order_market(
                    symbol="BTCUSDT",
                    side="Buy",
                    qty=0.012,
                    reduce_only=False,
                    position_idx=1,
                    order_link_id="cid-123",
                )

            args = client._sess.request.call_args.args
            kwargs = client._sess.request.call_args.kwargs
            self.assertGreaterEqual(len(args), 2)
            self.assertEqual(str(args[0]).upper(), "POST")
            self.assertTrue(str(args[1]).endswith("/v5/order/create"))

            sent_body = kwargs.get("data")
            self.assertIsInstance(sent_body, str)
            self.assertEqual(captured.get("payload"), sent_body)
            self.assertIn('"orderType":"Market"', sent_body)
            self.assertIn('"symbol":"BTCUSDT"', sent_body)
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()