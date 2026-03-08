from __future__ import annotations

from alerts.discord_client import DiscordClient


class DiscordAlerter:
    def __init__(self, webhook_url: str):
        self._client = DiscordClient(webhook_url=webhook_url)

    def send(self, text: str):
        self._client.send_text(text)
