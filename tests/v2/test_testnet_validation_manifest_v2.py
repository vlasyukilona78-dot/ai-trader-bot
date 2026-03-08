from __future__ import annotations

import unittest

from app.testnet_validation import collect_runtime_manifest


class TestnetValidationManifestV2Tests(unittest.TestCase):
    def test_runtime_manifest_shape(self):
        manifest = collect_runtime_manifest()
        self.assertIn("python", manifest)
        self.assertIn("packages", manifest)
        self.assertIn("version", manifest["python"])
        self.assertIn("executable", manifest["python"])
        for name in ("numpy", "pandas", "websockets", "requests"):
            self.assertIn(name, manifest["packages"])


if __name__ == "__main__":
    unittest.main()
