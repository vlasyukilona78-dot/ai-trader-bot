from types import SimpleNamespace
from unittest.mock import patch

from app.main import _build_alerters, _build_ultra_early_alerters


def _config():
    return SimpleNamespace(
        alerts=SimpleNamespace(
            telegram_token="configured-token",
            telegram_chat_id="configured-chat",
            discord_webhook_url="https://example.invalid/webhook",
        )
    )


def test_global_alert_switch_disables_all_transports():
    with patch.dict("os.environ", {"ALERTS_ENABLED": "false"}, clear=False):
        assert _build_alerters(_config()) == []
        assert _build_ultra_early_alerters(_config()) == []


def test_ultra_alert_switch_does_not_disable_main_alerts():
    with patch.dict(
        "os.environ",
        {
            "ALERTS_ENABLED": "true",
            "ULTRA_ALERTS_ENABLED": "false",
            "TELEGRAM_AUTO_LOCAL_PROXY": "false",
        },
        clear=False,
    ):
        assert len(_build_alerters(_config())) == 2
        assert _build_ultra_early_alerters(_config()) == []
