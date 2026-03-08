from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.bootstrap import ConfigError
from app.testnet_validation import STATUS_BLOCKED, STATUS_FAIL, classify_config_error_status, main


class TestnetValidationBlockedV2Tests(unittest.TestCase):
    def test_classify_missing_credentials_as_blocked(self):
        status = classify_config_error_status("BYBIT_API_KEY and BYBIT_API_SECRET are required for testnet/live modes")
        self.assertEqual(status, STATUS_BLOCKED)

    def test_classify_non_credentials_error_as_fail(self):
        status = classify_config_error_status("RISK_MAX_LEVERAGE must be > 0")
        self.assertEqual(status, STATUS_FAIL)

    def test_main_writes_blocked_report_on_missing_credentials(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = str(Path(tmpdir) / "report.json")
            args = Namespace(
                symbol="BTCUSDT",
                max_notional_usdt=20.0,
                execute_orders=False,
                soak_seconds=0,
                chaos_cycles=0,
                run_full_suite=False,
                artifacts_root=str(Path(tmpdir) / "artifacts"),
                deployment_constraints_out=str(Path(tmpdir) / "constraints.lock.txt"),
                report_out=report_path,
            )
            with patch("app.testnet_validation.parse_args", return_value=args), patch(
                "app.testnet_validation.setup_logging", return_value=MagicMock()
            ), patch(
                "app.testnet_validation.load_runtime_config",
                side_effect=ConfigError("BYBIT_API_KEY and BYBIT_API_SECRET are required for testnet/live modes"),
            ):
                code = main()

            self.assertEqual(code, 4)
            report = json.loads(Path(report_path).read_text(encoding="utf-8"))
            self.assertEqual(report["scenarios"][0]["status"], STATUS_BLOCKED)
            self.assertEqual(report["status_counts"].get(STATUS_BLOCKED), 1)

    def test_main_writes_fail_report_on_generic_config_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = str(Path(tmpdir) / "report.json")
            args = Namespace(
                symbol="BTCUSDT",
                max_notional_usdt=20.0,
                execute_orders=False,
                soak_seconds=0,
                chaos_cycles=0,
                run_full_suite=False,
                artifacts_root=str(Path(tmpdir) / "artifacts"),
                deployment_constraints_out=str(Path(tmpdir) / "constraints.lock.txt"),
                report_out=report_path,
            )
            with patch("app.testnet_validation.parse_args", return_value=args), patch(
                "app.testnet_validation.setup_logging", return_value=MagicMock()
            ), patch(
                "app.testnet_validation.load_runtime_config",
                side_effect=ConfigError("BOT_RUNTIME_MODE conflicts with BOT_MODE"),
            ):
                code = main()

            self.assertEqual(code, 2)
            report = json.loads(Path(report_path).read_text(encoding="utf-8"))
            self.assertEqual(report["scenarios"][0]["status"], STATUS_FAIL)
            self.assertEqual(report["status_counts"].get(STATUS_FAIL), 1)


if __name__ == "__main__":
    unittest.main()