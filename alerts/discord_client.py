from __future__ import annotations

import json

import requests


class DiscordClient:
    def __init__(self, webhook_url: str, timeout: int = 12):
        self.webhook_url = webhook_url or ""
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.webhook_url)

    def send_text(self, text: str) -> bool:
        if not self.enabled:
            return False
        try:
            resp = requests.post(self.webhook_url, json={"content": text}, timeout=self.timeout)
            return resp.status_code in (200, 204)
        except Exception:
            return False

    def send_embed(self, title: str, description: str, color: int = 0x00AAFF) -> bool:
        if not self.enabled:
            return False
        payload = {
            "embeds": [
                {
                    "title": title,
                    "description": description,
                    "color": int(color),
                }
            ]
        }
        try:
            resp = requests.post(self.webhook_url, json=payload, timeout=self.timeout)
            return resp.status_code in (200, 204)
        except Exception:
            return False

    def send_image(self, text: str, image_bytes: bytes, filename: str = "signal.png") -> bool:
        if not self.enabled:
            return False
        if not image_bytes:
            return False

        payload = {
            "content": text,
            "embeds": [
                {
                    "title": "Signal Chart",
                    "image": {"url": f"attachment://{filename}"},
                }
            ],
        }

        files = {
            "payload_json": (None, json.dumps(payload), "application/json"),
            "files[0]": (filename, image_bytes, "image/png"),
        }
        try:
            resp = requests.post(self.webhook_url, files=files, timeout=max(self.timeout, 20))
            return resp.status_code in (200, 204)
        except Exception:
            return False
