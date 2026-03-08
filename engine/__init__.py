"""Legacy engine modules are quarantined.

These modules are retained only for archive/reference and are blocked from runtime usage.
Use V2 modules under trading/* with app/main.py entrypoint.
"""

import os

if os.getenv("ALLOW_LEGACY_RUNTIME", "false").strip().lower() not in ("1", "true", "yes"):
    raise RuntimeError("Legacy engine package is quarantined. Use V2 runtime path app/main.py")
