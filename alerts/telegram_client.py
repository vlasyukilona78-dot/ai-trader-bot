from __future__ import annotations

import requests


class TelegramClient:
    def __init__(self, token: str, chat_id: str, timeout: int = 12):
        self.token = token or ""
        self.chat_id = chat_id or ""
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def send_text(self, text: str) -> bool:
        if not self.enabled:
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
        try:
            resp = requests.post(url, data=payload, timeout=self.timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def send_photo(self, caption: str, image_bytes: bytes, filename: str = "signal.png") -> bool:
        if not self.enabled:
            return False
        if not image_bytes:
            return False

        url = f"https://api.telegram.org/bot{self.token}/sendPhoto"
        data = {"chat_id": self.chat_id, "caption": caption, "parse_mode": "HTML"}
        files = {"photo": (filename, image_bytes, "image/png")}
        try:
            resp = requests.post(url, data=data, files=files, timeout=max(self.timeout, 20))
            return resp.status_code == 200
        except Exception:
            return False
