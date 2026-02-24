# reinforce_trainer.py (updated)
import pandas as pd
import os
import joblib
import time
from datetime import datetime
from logger import logger
from config import *
from codex_trainer import backup_model_files

TRADE_LOG = 'logs/trades_log.csv'
BACKUP_DIR = 'backups'
MIN_TRADES_TO_RETRAIN = int(os.getenv("MIN_TRADES_TO_RETRAIN", "20"))
RETRAIN_HISTORY = "logs/retrain_history.csv"

def append_trade_to_dataset(trade_result: dict):
    """Append a trade row to TRADE_LOG (CSV)."""
    os.makedirs('logs', exist_ok=True)
    row = {
        'time': trade_result.get('closed_at') or datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
        'symbol': trade_result.get('symbol'),
        'profit': float(trade_result.get('profit') or 0.0),
        'duration': float(trade_result.get('duration') or 0.0),
        'side': trade_result.get('side'),
    }
    df_row = pd.DataFrame([row])
    if not os.path.exists(TRADE_LOG):
        df_row.to_csv(TRADE_LOG, index=False)
    else:
        df_row.to_csv(TRADE_LOG, mode='a', index=False, header=False)
    logger.info("Appended trade to %s: %s", TRADE_LOG, row)

def maybe_retrain_models(train_callable, min_trades: int = MIN_TRADES_TO_RETRAIN):
    """
    If enough trades collected -> make backup, call train_callable(force=True),
    and write retrain history.
    """
    if not os.path.exists(TRADE_LOG):
        logger.debug("No trade log found")
        return False
    try:
        df = pd.read_csv(TRADE_LOG)
    except Exception as e:
        logger.exception("Failed to read TRADE_LOG: %s", e)
        return False
    if len(df) >= min_trades:
        logger.info("Found %d trades -> retraining models", len(df))
        # backup models
        try:
            backup_model_files(".")
        except Exception:
            logger.exception("backup_model_files failed")
        # call training
        started = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        success = False
        try:
            res = train_callable(force=True)
            success = bool(res)
            logger.info("train_callable returned %s", res)
        except Exception as e:
            logger.exception("train_callable exception: %s", e)
            success = False
        # write retrain history
        os.makedirs("logs", exist_ok=True)
        hist_row = {
            "time": started,
            "num_trades": len(df),
            "success": int(success)
        }
        hist_df = pd.DataFrame([hist_row])
        if not os.path.exists(RETRAIN_HISTORY):
            hist_df.to_csv(RETRAIN_HISTORY, index=False)
        else:
            hist_df.to_csv(RETRAIN_HISTORY, mode='a', index=False, header=False)
        return success
    return False