from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.bootstrap import ConfigError, load_runtime_config


class BootstrapConfigV2Tests(unittest.TestCase):
    def test_conflicting_runtime_modes_rejected(self):
        env = {
            "BOT_RUNTIME_MODE": "paper",
            "BOT_MODE": "testnet",
            "WS_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ConfigError):
                load_runtime_config()

    def test_live_requires_explicit_flags_and_cap(self):
        env = {
            "BOT_RUNTIME_MODE": "live",
            "BYBIT_API_KEY": "k",
            "BYBIT_API_SECRET": "s",
            "LIVE_TRADING_ENABLED": "true",
            "LIVE_STARTUP_MAX_NOTIONAL_USDT": "500",
            "RISK_MAX_TOTAL_NOTIONAL_PCT": "0.20",
            "FEATURE_RUNTIME_ENABLED": "false",
            "WS_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = load_runtime_config()
            self.assertEqual(cfg.mode, "live")
            self.assertFalse(cfg.adapter.testnet)
            self.assertFalse(cfg.adapter.dry_run)

    def test_live_rejected_without_live_toggle(self):
        env = {
            "BOT_RUNTIME_MODE": "live",
            "BYBIT_API_KEY": "k",
            "BYBIT_API_SECRET": "s",
            "LIVE_STARTUP_MAX_NOTIONAL_USDT": "500",
            "WS_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ConfigError):
                load_runtime_config()

    def test_testnet_requires_api_secrets(self):
        env = {
            "BOT_RUNTIME_MODE": "testnet",
            "LIVE_TRADING_ENABLED": "false",
            "WS_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ConfigError):
                load_runtime_config()


    def test_feature_runtime_dependency_mismatch_rejected(self):
        env = {
            "BOT_RUNTIME_MODE": "paper",
            "FEATURE_RUNTIME_ENABLED": "true",
            "WS_ENABLED": "false",
        }

        import importlib

        real_import_module = importlib.import_module

        def _side_effect(name, package=None):
            if name == "numpy":
                raise ImportError("missing_numpy")
            return real_import_module(name, package)

        with patch.dict(os.environ, env, clear=True):
            with patch("app.bootstrap.importlib.import_module", side_effect=_side_effect):
                with self.assertRaises(ConfigError):
                    load_runtime_config()
    def test_non_live_rejects_live_toggle(self):
        env = {
            "BOT_RUNTIME_MODE": "paper",
            "LIVE_TRADING_ENABLED": "true",
            "WS_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ConfigError):
                load_runtime_config()


if __name__ == "__main__":
    unittest.main()



