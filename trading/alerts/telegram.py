from __future__ import annotations

from alerts.telegram_client import TelegramClient


class TelegramAlerter:
    def __init__(self, token: str, chat_id: str):
        self._client = TelegramClient(token=token, chat_id=chat_id)

    def send(self, text: str):
        self._client.send_text(text)
