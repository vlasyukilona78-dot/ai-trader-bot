from __future__ import annotations

from alerts.telegram_client import TelegramClient


class TelegramAlerter:
    def __init__(self, token: str, chat_id: str):
        self._client = TelegramClient(token=token, chat_id=chat_id)

    def send(self, text: str, reply_markup: dict | None = None):
        return self._client.send_text(text, reply_markup=reply_markup)

    def send_photo(
        self,
        caption: str,
        image_bytes: bytes,
        filename: str = "signal.png",
        reply_markup: dict | None = None,
    ):
        return self._client.send_photo(
            caption=caption,
            image_bytes=image_bytes,
            filename=filename,
            reply_markup=reply_markup,
        )
