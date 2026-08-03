from __future__ import annotations

import logging
import unittest
from unittest.mock import patch

from trading.metrics.logging import CompactConsoleFormatter, _BelowWarningFilter


class MetricsLoggingV2Tests(unittest.TestCase):
    def test_compact_formatter_puts_message_before_structured_noise(self):
        record = logging.LogRecord(
            name="bot_v2",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="market_parallel_plan symbols=%s workers=%s",
            args=(413, 16),
            exc_info=None,
        )

        rendered = CompactConsoleFormatter().format(record)

        self.assertIn("INFO", rendered)
        self.assertTrue(rendered.endswith("market_parallel_plan symbols=413 workers=16"))
        self.assertNotIn('"logger"', rendered)

    def test_stdout_filter_keeps_info_out_of_error_stream(self):
        filter_ = _BelowWarningFilter()
        info = logging.LogRecord("bot_v2", logging.INFO, __file__, 1, "ok", (), None)
        warning = logging.LogRecord("bot_v2", logging.WARNING, __file__, 1, "warn", (), None)

        self.assertTrue(filter_.filter(info))
        self.assertFalse(filter_.filter(warning))


if __name__ == "__main__":
    unittest.main()
