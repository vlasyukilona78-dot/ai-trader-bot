import tempfile
import unittest
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    pd = None
    HAS_PANDAS = False

if HAS_PANDAS:
    from engine.schema import SIGNAL_COLUMNS, append_row_csv, validate_signal_row


@unittest.skipUnless(HAS_PANDAS, "pandas is not installed in test runtime")
class SchemaTests(unittest.TestCase):
    def test_validate_signal_row_required(self):
        row = validate_signal_row(
            {
                "symbol": "BTC/USDT",
                "direction": "SHORT",
                "entry": 100,
                "tp": 90,
                "sl": 110,
                "strategy": "pump_short_profile",
            }
        )
        self.assertEqual(row["symbol"], "BTC/USDT")
        self.assertEqual(row["direction"], "SHORT")

    def test_append_row_csv_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "signals.csv"
            row = validate_signal_row(
                {
                    "symbol": "BTC/USDT",
                    "direction": "SHORT",
                    "entry": 100,
                    "tp": 90,
                    "sl": 110,
                    "strategy": "pump_short_profile",
                }
            )
            append_row_csv(path, row, SIGNAL_COLUMNS)
            df = pd.read_csv(path)
            self.assertIn("symbol", df.columns)
            self.assertEqual(len(df), 1)


if __name__ == "__main__":
    unittest.main()
