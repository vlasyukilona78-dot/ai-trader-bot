# fix_logs.py — normalize logs/signals_log.csv structure, dedupe, sort
import pandas as pd
import os
from logger import logger

LOG_PATH = "logs/signals_log.csv"
EXPECTED_COLS = ["time","symbol","direction","entry","tp","sl","rsi","vol_change","ai_prob","ai_horizon","strategy"]

def fix_logs(path=LOG_PATH):
    if not os.path.exists(path):
        logger.warning("Log file not found: %s", path)
        return False
    try:
        # try safe read
        try:
            df = pd.read_csv(path, on_bad_lines="skip")
        except TypeError:
            # older pandas
            df = pd.read_csv(path, error_bad_lines=False, warn_bad_lines=True)  # type: ignore
        # ensure columns exist
        for c in EXPECTED_COLS:
            if c not in df.columns:
                df[c] = None
        df = df[EXPECTED_COLS].copy()
        # convert numeric
        for c in ["entry","tp","sl","rsi","vol_change","ai_prob","ai_horizon"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # drop rows missing essential fields
        df = df.dropna(subset=["symbol","direction","entry"])
        # dedupe by time+symbol+direction
        df = df.drop_duplicates(subset=["time","symbol","direction"])
        # sort by time if parseable
        try:
            df["time_parsed"] = pd.to_datetime(df["time"], errors="coerce")
            df = df.sort_values("time_parsed").drop(columns=["time_parsed"])
        except Exception:
            pass
        df.to_csv(path, index=False)
        logger.info("Logs fixed: %s rows remain", len(df))
        return True
    except Exception as e:
        logger.exception("fix_logs failed: %s", e)
        return False

if __name__ == "__main__":
    fix_logs()