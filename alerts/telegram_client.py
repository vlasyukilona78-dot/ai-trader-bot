from __future__ import annotations

import json
import logging
import os
import socket
from urllib.parse import urlparse

import requests


logger = logging.getLogger("bot_v2")


class TelegramClient:
    def __init__(self, token: str, chat_id: str, timeout: int = 12):
        self.token = token or ""
        self.chat_id = chat_id or ""
        self.timeout = timeout
        self._configured_proxy_url = self._resolve_proxy_url()
        self._transports = self._build_transports(self._configured_proxy_url)
        self._transport_names = [name for name, _ in self._transports]
        self._has_proxy_transport = "proxy" in self._transport_names
        logger.info(
            "telegram transports configured transports=%s proxy_configured=%s proxy_active=%s",
            ",".join(self._transport_names) or "none",
            bool(self._configured_proxy_url),
            self._has_proxy_transport,
        )

    @staticmethod
    def _create_session(proxy_url: str = "") -> requests.Session:
        session = requests.Session()
        session.trust_env = False
        if proxy_url:
            session.proxies.update({"http": proxy_url, "https": proxy_url})
        return session

    def _build_transports(self, proxy_url: str = "") -> list[tuple[str, requests.Session]]:
        transports: list[tuple[str, requests.Session]] = []
        if proxy_url:
            if self._proxy_reachable(proxy_url):
                transports.append(("proxy", self._create_session(proxy_url)))
            else:
                logger.warning("telegram proxy is not reachable at startup: %s", proxy_url)
            if self._env_truthy("TELEGRAM_DIRECT_FALLBACK", True) or not transports:
                transports.append(("direct", self._create_session()))
            return transports
        transports.append(("direct", self._create_session()))
        return transports

    @staticmethod
    def _env_truthy(name: str, default: bool = False) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        value = str(raw).strip().lower()
        if not value:
            return default
        return value in {"1", "true", "yes", "on"}

    @staticmethod
    def _csv_env(name: str, default: tuple[str, ...]) -> list[str]:
        raw = os.getenv(name)
        if raw is None:
            return list(default)
        values = [part.strip() for part in str(raw).split(",")]
        return [value for value in values if value]

    @classmethod
    def _detect_local_proxy_url(cls) -> str:
        scheme = str(os.getenv("TELEGRAM_LOCAL_PROXY_SCHEME") or "http").strip().lower() or "http"
        if scheme not in {"http", "socks5", "socks5h"}:
            scheme = "http"
        hosts = cls._csv_env("TELEGRAM_LOCAL_PROXY_HOSTS", ("127.0.0.1", "localhost"))
        ports = cls._csv_env("TELEGRAM_LOCAL_PROXY_PORTS", ("10801", "10809", "7890", "8080"))
        for host in hosts:
            for port_raw in ports:
                try:
                    port = int(str(port_raw).strip())
                except (TypeError, ValueError):
                    continue
                if port <= 0:
                    continue
                candidate = f"{scheme}://{host}:{port}"
                if cls._proxy_reachable(candidate):
                    logger.info("telegram auto local proxy detected: %s", candidate)
                    return candidate
        return ""

    @classmethod
    def _resolve_proxy_url(cls) -> str:
        for name in (
            "TELEGRAM_PROXY_URL",
            "BOT_TELEGRAM_PROXY_URL",
            "TELEGRAM_HTTP_PROXY",
            "TELEGRAM_HTTPS_PROXY",
        ):
            raw = os.getenv(name)
            if raw is None:
                continue
            value = str(raw).strip()
            if value:
                return value
        if cls._env_truthy("TELEGRAM_AUTO_LOCAL_PROXY", True):
            return cls._detect_local_proxy_url()
        return ""

    @staticmethod
    def _proxy_reachable(proxy_url: str) -> bool:
        try:
            parsed = urlparse(proxy_url)
            host = str(parsed.hostname or "").strip()
            port = int(parsed.port or 0)
        except Exception:
            return True
        if not host or port <= 0:
            return True
        if host not in {"127.0.0.1", "localhost", "::1"}:
            return True
        try:
            with socket.create_connection((host, port), timeout=0.35):
                return True
        except OSError:
            return False

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    @staticmethod
    def _serialize_reply_markup(reply_markup: dict | None) -> str | None:
        if not isinstance(reply_markup, dict) or not reply_markup:
            return None
        return json.dumps(reply_markup, ensure_ascii=False, separators=(",", ":"))

    def _post_with_failover(
        self,
        url: str,
        *,
        data: dict,
        files: dict | None = None,
        timeout=None,
        attempts_per_transport: int = 1,
    ) -> tuple[bool, str]:
        last_status = None
        last_text = ""
        last_exc = None
        last_transport = ""
        for transport_name, session in self._transports:
            for _ in range(max(1, int(attempts_per_transport))):
                try:
                    resp = session.post(url, data=data, files=files, timeout=timeout)
                    if resp.status_code == 200:
                        return True, transport_name
                    last_transport = transport_name
                    last_status = resp.status_code
                    last_text = resp.text[:300]
                except Exception as exc:
                    last_transport = transport_name
                    last_exc = exc
                    continue
        if last_status is not None:
            logger.warning("telegram request HTTP %s via %s: %s", last_status, last_transport or "unknown", last_text)
        elif last_exc is not None:
            logger.warning("telegram request failed via %s: %s", last_transport or "unknown", last_exc)
        return False, last_transport

    def send_text(self, text: str, reply_markup: dict | None = None) -> bool:
        if not self.enabled:
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
        reply_markup_json = self._serialize_reply_markup(reply_markup)
        if reply_markup_json:
            payload["reply_markup"] = reply_markup_json
        delivered, transport = self._post_with_failover(
            url,
            data=payload,
            timeout=(3, max(6, int(self.timeout))) if self._has_proxy_transport else (4, max(6, int(self.timeout))),
            attempts_per_transport=1,
        )
        if delivered:
            logger.info("telegram sendMessage ok via %s", transport or "unknown")
            return True
        return False

    def _send_document_fallback(
        self,
        *,
        caption: str,
        image_bytes: bytes,
        filename: str,
        reply_markup: dict | None = None,
    ) -> bool:
        url = f"https://api.telegram.org/bot{self.token}/sendDocument"
        data = {"chat_id": self.chat_id, "caption": caption, "parse_mode": "HTML"}
        reply_markup_json = self._serialize_reply_markup(reply_markup)
        if reply_markup_json:
            data["reply_markup"] = reply_markup_json
        files = {"document": (filename, image_bytes, "image/png")}
        delivered, transport = self._post_with_failover(
            url,
            data=data,
            files=files,
            timeout=(3, max(int(self.timeout), 18)) if self._has_proxy_transport else (5, max(int(self.timeout), 25)),
            attempts_per_transport=1,
        )
        if delivered:
            logger.info("telegram sendDocument ok via %s", transport or "unknown")
            return True
        return False

    def send_photo(
        self,
        caption: str,
        image_bytes: bytes,
        filename: str = "signal.png",
        reply_markup: dict | None = None,
    ) -> bool:
        if not self.enabled:
            return False
        if not image_bytes:
            return False

        url = f"https://api.telegram.org/bot{self.token}/sendPhoto"
        data = {"chat_id": self.chat_id, "caption": caption, "parse_mode": "HTML"}
        reply_markup_json = self._serialize_reply_markup(reply_markup)
        if reply_markup_json:
            data["reply_markup"] = reply_markup_json
        files = {"photo": (filename, image_bytes, "image/png")}
        delivered, transport = self._post_with_failover(
            url,
            data=data,
            files=files,
            timeout=(3, max(int(self.timeout), 15)) if self._has_proxy_transport else (6, max(int(self.timeout), 25)),
            attempts_per_transport=1,
        )
        if delivered:
            logger.info("telegram sendPhoto ok via %s", transport or "unknown")
            return True
        return self._send_document_fallback(
            caption=caption,
            image_bytes=image_bytes,
            filename=filename,
            reply_markup=reply_markup,
        )
