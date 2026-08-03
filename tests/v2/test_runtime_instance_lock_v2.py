from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.main import (
    _acquire_runtime_instance_lock_file,
    _release_runtime_instance_lock_file,
)


class RuntimeInstanceLockV2Tests(unittest.TestCase):
    def test_blocks_duplicate_and_releases_for_next_start(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bot_runtime.lock"
            first = _acquire_runtime_instance_lock_file(path)
            try:
                with self.assertRaises(OSError):
                    _acquire_runtime_instance_lock_file(path)
            finally:
                _release_runtime_instance_lock_file(first)

            restarted = _acquire_runtime_instance_lock_file(path)
            _release_runtime_instance_lock_file(restarted)


if __name__ == "__main__":
    unittest.main()
