from __future__ import annotations

from trading.alerts.telegram import TelegramAlerter


class _Client:
    def __init__(self, result: bool) -> None:
        self.result = result
        self.messages: list[str] = []

    def send_text(self, text: str) -> bool:
        self.messages.append(text)
        return self.result


def _alerter(result: bool) -> tuple[TelegramAlerter, _Client]:
    client = _Client(result)
    alerter = object.__new__(TelegramAlerter)
    alerter._client = client
    return alerter, client


def test_send_returns_confirmed_client_success() -> None:
    alerter, client = _alerter(True)

    assert alerter.send("signal") is True
    assert client.messages == ["signal"]


def test_send_does_not_report_delivery_when_client_returns_false() -> None:
    alerter, _ = _alerter(False)

    assert alerter.send("signal") is False
