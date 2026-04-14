from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from alerts.telegram_client import TelegramClient


class _FakeResponse:
    def __init__(self, status_code: int = 200, text: str = "ok"):
        self.status_code = status_code
        self.text = text


class _FakeSession:
    def __init__(self, behavior):
        self._behavior = behavior

    def post(self, *args, **kwargs):
        if isinstance(self._behavior, Exception):
            raise self._behavior
        if callable(self._behavior):
            return self._behavior(*args, **kwargs)
        return self._behavior


class TelegramClientV2Tests(unittest.TestCase):
    def test_prefers_proxy_then_direct_when_proxy_configured(self):
        env = {"TELEGRAM_PROXY_URL": "socks5://127.0.0.1:1080"}
        with patch.dict(os.environ, env, clear=False):
            client = TelegramClient(token="t", chat_id="c")
        self.assertEqual([name for name, _ in client._transports], ["proxy", "direct"])

    def test_uses_direct_only_when_proxy_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            client = TelegramClient(token="t", chat_id="c")
        self.assertEqual([name for name, _ in client._transports], ["direct"])

    def test_send_text_falls_back_to_direct_when_proxy_transport_fails(self):
        with patch.dict(os.environ, {"TELEGRAM_PROXY_URL": "socks5://127.0.0.1:1080"}, clear=False):
            client = TelegramClient(token="t", chat_id="c")
        client._transports = [
            ("proxy", _FakeSession(RuntimeError("proxy_down"))),
            ("direct", _FakeSession(_FakeResponse(200, "ok"))),
        ]
        self.assertTrue(client.send_text("hello"))

    def test_send_photo_falls_back_to_direct_when_proxy_transport_fails(self):
        with patch.dict(os.environ, {"TELEGRAM_PROXY_URL": "socks5://127.0.0.1:1080"}, clear=False):
            client = TelegramClient(token="t", chat_id="c")
        client._transports = [
            ("proxy", _FakeSession(RuntimeError("proxy_down"))),
            ("direct", _FakeSession(_FakeResponse(200, "ok"))),
        ]
        self.assertTrue(client.send_photo("caption", b"png-bytes"))


if __name__ == "__main__":
    unittest.main()
