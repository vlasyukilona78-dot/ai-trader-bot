from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.main import _analysis_worker_count


class MarketParallelismV2Tests(unittest.TestCase):
    def test_worker_count_uses_configured_parallelism_for_full_market(self):
        with patch.dict(
            os.environ,
            {
                "CONCURRENT_TASKS": "16",
                "MARKET_ANALYSIS_MAX_WORKERS": "32",
            },
            clear=False,
        ):
            self.assertEqual(_analysis_worker_count(413), 16)

    def test_worker_count_is_bounded_by_safety_cap_and_market_size(self):
        with patch.dict(
            os.environ,
            {
                "CONCURRENT_TASKS": "413",
                "MARKET_ANALYSIS_MAX_WORKERS": "32",
            },
            clear=False,
        ):
            self.assertEqual(_analysis_worker_count(413), 32)
            self.assertEqual(_analysis_worker_count(7), 7)


if __name__ == "__main__":
    unittest.main()
