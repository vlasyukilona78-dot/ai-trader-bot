from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.bootstrap import ConfigError, _quality_symbol_allowed, load_runtime_config


class BootstrapConfigV2Tests(unittest.TestCase):
    def test_quality_symbol_filter_rejects_wrappers_and_metals(self):
        self.assertFalse(_quality_symbol_allowed("1000PEPEUSDT"))
        self.assertFalse(_quality_symbol_allowed("SHIB1000USDT"))
        self.assertFalse(_quality_symbol_allowed("XAUUSDT"))
        self.assertFalse(_quality_symbol_allowed("XAUTUSDT"))
        self.assertFalse(_quality_symbol_allowed("PAXGUSDT"))
        self.assertFalse(_quality_symbol_allowed("USD1USDT"))
        self.assertFalse(_quality_symbol_allowed("USUSDT"))
        self.assertFalse(_quality_symbol_allowed("ETHBTCUSDT"))
        self.assertTrue(_quality_symbol_allowed("AAVEUSDT"))
        self.assertTrue(_quality_symbol_allowed("1INCHUSDT"))

    def test_quality_symbol_filter_rejects_too_new_listings_when_requested(self):
        now_ts_ms = 1_800_000_000_000
        listed_5_days_ago = now_ts_ms - 5 * 86_400_000
        listed_45_days_ago = now_ts_ms - 45 * 86_400_000
        self.assertFalse(
            _quality_symbol_allowed(
                "NEWTUSDT",
                launch_time_ms=listed_5_days_ago,
                min_days_listed=21,
                now_ts_ms=now_ts_ms,
            )
        )
        self.assertTrue(
            _quality_symbol_allowed(
                "AAVEUSDT",
                launch_time_ms=listed_45_days_ago,
                min_days_listed=21,
                now_ts_ms=now_ts_ms,
            )
        )

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

    def test_live_allows_online_retrain_when_enabled(self):
        env = {
            "BOT_RUNTIME_MODE": "live",
            "BYBIT_API_KEY": "k",
            "BYBIT_API_SECRET": "s",
            "LIVE_TRADING_ENABLED": "true",
            "LIVE_STARTUP_MAX_NOTIONAL_USDT": "500",
            "RISK_MAX_TOTAL_NOTIONAL_PCT": "0.20",
            "ML_ONLINE_RETRAIN_ENABLED": "true",
            "FEATURE_RUNTIME_ENABLED": "false",
            "WS_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = load_runtime_config()
            self.assertEqual(cfg.mode, "live")
            self.assertTrue(cfg.flags.online_retraining_enabled)

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

    def test_demo_mode_uses_mainnet_public_and_non_dry_private_trading(self):
        env = {
            "BOT_RUNTIME_MODE": "demo",
            "BYBIT_TESTNET": "false",
            "BYBIT_API_KEY": "k",
            "BYBIT_API_SECRET": "s",
            "FEATURE_RUNTIME_ENABLED": "false",
            "WS_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = load_runtime_config()
            self.assertEqual(cfg.mode, "demo")
            self.assertFalse(cfg.adapter.testnet)
            self.assertFalse(cfg.adapter.dry_run)
            self.assertTrue(cfg.adapter.demo)

    def test_demo_mode_accepts_generic_key_aliases_for_runtime_learning(self):
        env = {
            "BOT_RUNTIME_MODE": "demo",
            "BYBIT_TESTNET": "false",
            "BYBIT_KEY": "k",
            "BYBIT_SECRET": "s",
            "FEATURE_RUNTIME_ENABLED": "false",
            "WS_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = load_runtime_config()
            self.assertEqual(cfg.mode, "demo")
            self.assertEqual(cfg.adapter.api_key, "k")
            self.assertEqual(cfg.adapter.api_secret, "s")
            self.assertTrue(cfg.adapter.demo)

    def test_demo_mode_accepts_demo_alias_key_names_for_runtime_learning(self):
        env = {
            "BOT_RUNTIME_MODE": "demo",
            "BYBIT_TESTNET": "false",
            "DEMO_API_KEY": "k",
            "DEMO_API_SECRET": "s",
            "FEATURE_RUNTIME_ENABLED": "false",
            "WS_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = load_runtime_config()
            self.assertEqual(cfg.mode, "demo")
            self.assertEqual(cfg.adapter.api_key, "k")
            self.assertEqual(cfg.adapter.api_secret, "s")
            self.assertTrue(cfg.adapter.demo)

    def test_demo_mode_uses_profile_specific_demo_credentials_for_main(self):
        env = {
            "BOT_RUNTIME_MODE": "demo",
            "BOT_SIGNAL_PROFILE": "main",
            "BYBIT_TESTNET": "false",
            "BYBIT_DEMO_API_KEY_MAIN": "main_k",
            "BYBIT_DEMO_API_SECRET_MAIN": "main_s",
            "FEATURE_RUNTIME_ENABLED": "false",
            "WS_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = load_runtime_config()
            self.assertEqual(cfg.adapter.api_key, "main_k")
            self.assertEqual(cfg.adapter.api_secret, "main_s")
            self.assertEqual(cfg.runtime_db_path, "data/runtime/v2_demo_runtime_main.db")

    def test_demo_mode_uses_profile_specific_demo_credentials_for_early(self):
        env = {
            "BOT_RUNTIME_MODE": "demo",
            "BOT_SIGNAL_PROFILE": "early",
            "BYBIT_TESTNET": "false",
            "BYBIT_DEMO_API_KEY_EARLY": "early_k",
            "BYBIT_DEMO_API_SECRET_EARLY": "early_s",
            "FEATURE_RUNTIME_ENABLED": "false",
            "WS_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = load_runtime_config()
            self.assertEqual(cfg.adapter.api_key, "early_k")
            self.assertEqual(cfg.adapter.api_secret, "early_s")
            self.assertEqual(cfg.runtime_db_path, "data/runtime/v2_demo_runtime_early.db")

    def test_demo_mode_profiles_generic_runtime_db_path_for_main(self):
        env = {
            "BOT_RUNTIME_MODE": "demo",
            "BOT_SIGNAL_PROFILE": "main",
            "BYBIT_TESTNET": "false",
            "BYBIT_API_KEY": "k",
            "BYBIT_API_SECRET": "s",
            "RUNTIME_DB_PATH": "data/runtime/v2_demo_runtime.db",
            "FEATURE_RUNTIME_ENABLED": "false",
            "WS_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = load_runtime_config()
            self.assertEqual(cfg.runtime_db_path, "data/runtime/v2_demo_runtime_main.db")

    def test_demo_mode_profiles_generic_runtime_db_path_for_early(self):
        env = {
            "BOT_RUNTIME_MODE": "demo",
            "BOT_SIGNAL_PROFILE": "early",
            "BYBIT_TESTNET": "false",
            "BYBIT_API_KEY": "k",
            "BYBIT_API_SECRET": "s",
            "RUNTIME_DB_PATH": "data/runtime/v2_demo_runtime.db",
            "FEATURE_RUNTIME_ENABLED": "false",
            "WS_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = load_runtime_config()
            self.assertEqual(cfg.runtime_db_path, "data/runtime/v2_demo_runtime_early.db")

    def test_legacy_demo_env_infers_demo_mode_when_runtime_mode_missing(self):
        env = {
            "BYBIT_ENV": "demo",
            "BYBIT_TESTNET": "false",
            "BYBIT_API_KEY": "k",
            "BYBIT_API_SECRET": "s",
            "FEATURE_RUNTIME_ENABLED": "false",
            "WS_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = load_runtime_config()
            self.assertEqual(cfg.mode, "demo")
            self.assertTrue(cfg.adapter.demo)
            self.assertFalse(cfg.adapter.dry_run)


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



