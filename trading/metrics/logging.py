from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "event"):
            payload["event"] = record.event
        return json.dumps(payload, ensure_ascii=False)


class CompactConsoleFormatter(logging.Formatter):
    """Readable IDE output while structured JSON remains available for collectors."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
        return f"{timestamp} | {record.levelname:<7} | {record.getMessage()}"


class _BelowWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < logging.WARNING


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("bot_v2")
    logger.setLevel(level.upper())
    if logger.handlers:
        return logger

    output_format = str(os.getenv("BOT_CONSOLE_LOG_FORMAT", "json")).strip().lower()
    if output_format in {"compact", "human", "readable"}:
        formatter = CompactConsoleFormatter()

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.addFilter(_BelowWarningFilter())
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger
