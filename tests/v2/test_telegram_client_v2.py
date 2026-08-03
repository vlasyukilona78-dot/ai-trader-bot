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


class _SequenceSession:
    def __init__(self, behaviors):
        self._behaviors = list(behaviors)

    def post(self, *args, **kwargs):
        if not self._behaviors:
            return _FakeResponse(200, "ok")
        behavior = self._behaviors.pop(0)
        if isinstance(behavior, Exception):
            raise behavior
        if callable(behavior):
            return behavior(*args, **kwargs)
        return behavior


class TelegramClientV2Tests(unittest.TestCase):
    def test_prefers_proxy_then_direct_when_proxy_configured(self):
        env = {"TELEGRAM_PROXY_URL": "socks5://127.0.0.1:1080"}
        with patch.dict(os.environ, env, clear=False), patch.object(TelegramClient, "_proxy_reachable", return_value=True):
            client = TelegramClient(token="t", chat_id="c")
        self.assertEqual([name for name, _ in client._transports], ["proxy", "direct"])

    def test_uses_direct_only_when_proxy_missing(self):
        with patch.dict(os.environ, {"TELEGRAM_AUTO_LOCAL_PROXY": "0"}, clear=True):
            client = TelegramClient(token="t", chat_id="c")
        self.assertEqual([name for name, _ in client._transports], ["direct"])

    def test_auto_detects_local_proxy_when_explicit_proxy_missing(self):
        env = {
            "TELEGRAM_AUTO_LOCAL_PROXY": "1",
            "TELEGRAM_LOCAL_PROXY_PORTS": "10801",
            "TELEGRAM_LOCAL_PROXY_HOSTS": "127.0.0.1",
        }
        with patch.dict(os.environ, env, clear=True), patch.object(TelegramClient, "_proxy_reachable", return_value=True):
            client = TelegramClient(token="t", chat_id="c")
        self.assertEqual(client._configured_proxy_url, "http://127.0.0.1:10801")
        self.assertEqual([name for name, _ in client._transports], ["proxy", "direct"])

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

    def test_send_photo_document_fallback_suppresses_transient_photo_warning(self):
        with patch.dict(os.environ, {"TELEGRAM_AUTO_LOCAL_PROXY": "0"}, clear=True):
            client = TelegramClient(token="t", chat_id="c")
        client._transports = [
            ("direct", _SequenceSession([ConnectionError("reset_once"), _FakeResponse(200, "ok")])),
        ]
        with patch("alerts.telegram_client.logger.warning") as warning_mock:
            self.assertTrue(client.send_photo("caption", b"png-bytes"))
        warning_mock.assert_not_called()

    def test_skips_dead_local_proxy_and_uses_direct(self):
        with patch.dict(os.environ, {"TELEGRAM_PROXY_URL": "http://127.0.0.1:10801"}, clear=False), patch.object(
            TelegramClient, "_proxy_reachable", return_value=False
        ):
            client = TelegramClient(token="t", chat_id="c")
        self.assertEqual([name for name, _ in client._transports], ["direct"])

    def test_failure_log_redacts_telegram_token_and_proxy_credentials(self):
        token = "1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghi"
        proxy = "http://proxy-user:proxy-password@proxy.example:8080"
        with patch.dict(os.environ, {"TELEGRAM_PROXY_URL": proxy}, clear=False), patch.object(
            TelegramClient, "_proxy_reachable", return_value=True
        ):
            client = TelegramClient(token=token, chat_id="c")
        client._transports = [
            (
                "proxy",
                _FakeSession(
                    RuntimeError(
                        f"request failed /bot{token}/sendMessage through {proxy}"
                    )
                ),
            ),
        ]
        with patch("alerts.telegram_client.logger.warning") as warning_mock:
            self.assertFalse(client.send_text("hello"))
        logged = str(warning_mock.call_args_list)
        self.assertNotIn(token, logged)
        self.assertNotIn("proxy-password", logged)
        self.assertIn("[REDACTED_TELEGRAM_TOKEN]", logged)
        self.assertIn("[REDACTED_PROXY_AUTH]", logged)


if __name__ == "__main__":
    unittest.main()
