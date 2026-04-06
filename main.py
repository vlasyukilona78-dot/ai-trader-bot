import os
import sys

from app.main import main


if __name__ == "__main__":
    if os.getenv("PYCHARM_HOSTED") == "1" and len(sys.argv) == 1:
        os.environ.setdefault("BOT_SIGNAL_PROFILE", "main")
        sys.argv.append("--loop")
    raise SystemExit(main())
