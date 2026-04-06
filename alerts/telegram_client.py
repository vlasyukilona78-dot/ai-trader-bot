from __future__ import annotations

import json
import logging
import time

import requests


logger = logging.getLogger("bot_v2")


class TelegramClient:
    def __init__(self, token: str, chat_id: str, timeout: int = 12):
        self.token = token or ""
        self.chat_id = chat_id or ""
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    @staticmethod
    def _serialize_reply_markup(reply_markup: dict | None) -> str | None:
        if not isinstance(reply_markup, dict) or not reply_markup:
            return None
        return json.dumps(reply_markup, ensure_ascii=False, separators=(",", ":"))

    def send_text(self, text: str, reply_markup: dict | None = None) -> bool:
        if not self.enabled:
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
        reply_markup_json = self._serialize_reply_markup(reply_markup)
        if reply_markup_json:
            payload["reply_markup"] = reply_markup_json
        last_status = None
        last_text = ""
        last_exc = None
        for attempt in range(2):
            try:
                resp = requests.post(url, data=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    logger.info("telegram sendMessage ok")
                    return True
                last_status = resp.status_code
                last_text = resp.text[:300]
            except Exception as exc:
                last_exc = exc
            if attempt == 0:
                time.sleep(1.0)
        if last_status is not None:
            logger.warning("telegram sendMessage HTTP %s: %s", last_status, last_text)
        elif last_exc is not None:
            logger.warning("telegram sendMessage failed: %s", last_exc)
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
        last_status = None
        last_text = ""
        last_exc = None
        for attempt in range(2):
            try:
                resp = requests.post(url, data=data, files=files, timeout=max(self.timeout, 20))
                if resp.status_code == 200:
                    logger.info("telegram sendPhoto ok")
                    return True
                last_status = resp.status_code
                last_text = resp.text[:300]
            except Exception as exc:
                last_exc = exc
            if attempt == 0:
                time.sleep(1.0)
        if last_status is not None:
            logger.warning("telegram sendPhoto HTTP %s: %s", last_status, last_text)
        elif last_exc is not None:
            logger.warning("telegram sendPhoto failed: %s", last_exc)
        return False
