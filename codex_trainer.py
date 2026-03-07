import os
import shutil
import time
from datetime import datetime
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from config import LOG_PATH, RETRAIN_INTERVAL, TRADE_LOG_PATH
from logger import logger

LOGS_PATH = LOG_PATH or "logs/signals_log.csv"
TRADES_PATH = TRADE_LOG_PATH or "logs/trades_log.csv"
BACKUP_DIR = "backups"
MODEL_FILES = ("ai_model_win.pkl", "ai_model_horizon.pkl", "scaler.pkl", "ai_calibrator.pkl")
DEFAULT_HORIZON = 8.48
MIN_TRAIN_ROWS = int(os.getenv("MIN_TRAIN_ROWS", "30"))

FEATURE_COLUMNS = [
    "rsi",
    "rsi_5mean",
    "rsi_20mean",
    "price_change",
    "vol_change",
    "ema20",
    "ema50",
    "atr",
    "close",
    "ema_diff_norm",
    "atr_norm",
]


SIDE_TO_DIRECTION = {
    "buy": "LONG",
    "long": "LONG",
    "sell": "SHORT",
    "short": "SHORT",
}


def backup_model_files(save_dir: str = "."):
    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    for name in MODEL_FILES:
        source = os.path.join(save_dir, name)
        if not os.path.exists(source):
            continue
        target = os.path.join(BACKUP_DIR, f"{ts}_{name}")
        shutil.copy2(source, target)
        logger.info("Model backup created: %s", target)


def safe_read_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
    except TypeError:
        df = pd.read_csv(path, error_bad_lines=False, warn_bad_lines=True)  # type: ignore[arg-type]

    if df.empty:
        return df

    df.columns = [str(col).strip() for col in df.columns]
    return df


def _to_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _normalize_direction(value: str) -> str:
    val = str(value or "").strip().lower()
    if val in SIDE_TO_DIRECTION:
        return SIDE_TO_DIRECTION[val]
    up = str(value or "").strip().upper()
    if up in ("LONG", "SHORT"):
        return up
    return ""


def _prepare_signals_df(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if "time" not in data.columns:
        raise ValueError("signals log must contain time column")
    if "symbol" not in data.columns or "direction" not in data.columns:
        raise ValueError("signals log must contain symbol and direction")

    data["time"] = _to_dt(data["time"])
    data = data.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    data["direction"] = data["direction"].map(_normalize_direction)
    data = data[data["direction"].isin(["LONG", "SHORT"])].copy()
    return data


def _prepare_trades_df(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    required = {"time", "symbol", "profit"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"trades log missing required columns: {sorted(missing)}")

    data["time"] = _to_dt(data["time"])
    data["profit"] = pd.to_numeric(data["profit"], errors="coerce")
    data["duration"] = pd.to_numeric(data.get("duration", DEFAULT_HORIZON), errors="coerce")
    data["side"] = data.get("side", "").map(_normalize_direction)
    data = data.dropna(subset=["time", "profit"]).sort_values("time").reset_index(drop=True)
    data = data[data["side"].isin(["LONG", "SHORT"])].copy()
    return data


def _merge_labels(signals_df: pd.DataFrame, trades_df: pd.DataFrame, max_delay_hours: int = 12) -> pd.DataFrame:
    labeled_parts = []

    for symbol in sorted(set(signals_df["symbol"]).intersection(set(trades_df["symbol"]))):
        sig_symbol = signals_df[signals_df["symbol"] == symbol].copy()
        tr_symbol = trades_df[trades_df["symbol"] == symbol].copy()

        for direction in ("LONG", "SHORT"):
            sig = sig_symbol[sig_symbol["direction"] == direction].copy()
            tr = tr_symbol[tr_symbol["side"] == direction].copy()
            if sig.empty or tr.empty:
                continue

            sig = sig.sort_values("time")
            tr = tr.sort_values("time")

            merged = pd.merge_asof(
                tr,
                sig,
                on="time",
                direction="backward",
                tolerance=pd.Timedelta(hours=max_delay_hours),
                suffixes=("_trade", ""),
            )
            merged = merged.dropna(subset=["entry", "tp", "sl"]) if set(["entry", "tp", "sl"]).issubset(merged.columns) else merged
            if merged.empty:
                continue

            merged["profit"] = pd.to_numeric(merged["profit"], errors="coerce")
            merged["duration"] = pd.to_numeric(merged["duration"], errors="coerce")
            merged = merged.dropna(subset=["profit"])
            if merged.empty:
                continue

            labeled_parts.append(merged)

    if not labeled_parts:
        return pd.DataFrame()

    labeled = pd.concat(labeled_parts, ignore_index=True)
    labeled = labeled.sort_values("time").reset_index(drop=True)
    return labeled


def update_dataset() -> pd.DataFrame:
    if not os.path.exists(LOGS_PATH):
        logger.warning("Training log file not found: %s", LOGS_PATH)
        return pd.DataFrame()

    try:
        signals_raw = safe_read_csv(LOGS_PATH)
    except Exception as exc:
        logger.exception("Failed to read signal CSV: %s", exc)
        return pd.DataFrame()

    if signals_raw.empty:
        return pd.DataFrame()

    signals = _prepare_signals_df(signals_raw)

    # Preferred path: labels already present in same dataset.
    if "profit" in signals.columns and signals["profit"].notna().any():
        data = signals.copy()
        data["profit"] = pd.to_numeric(data["profit"], errors="coerce")
        data["duration"] = pd.to_numeric(data.get("duration", DEFAULT_HORIZON), errors="coerce")
        data = data.dropna(subset=["profit"])
        if not data.empty:
            logger.info("Loaded labeled rows directly from signal log: %d", len(data))
            return data

    # Strict mode: do not self-label from ai_prob; require real trade outcomes.
    if not os.path.exists(TRADES_PATH):
        logger.warning("No trades log for labels: %s", TRADES_PATH)
        return pd.DataFrame()

    try:
        trades_raw = safe_read_csv(TRADES_PATH)
    except Exception as exc:
        logger.exception("Failed to read trades CSV: %s", exc)
        return pd.DataFrame()

    if trades_raw.empty:
        logger.warning("Trades log is empty, cannot train without true labels")
        return pd.DataFrame()

    try:
        trades = _prepare_trades_df(trades_raw)
    except Exception as exc:
        logger.exception("Trades data prep failed: %s", exc)
        return pd.DataFrame()

    data = _merge_labels(signals, trades)
    if data.empty:
        logger.warning("Could not map trades to signals for supervised labels")
        return pd.DataFrame()

    logger.info("Labeled dataset assembled via signals+trades merge: %d rows", len(data))
    return data


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    data = df.copy()

    for col in ["rsi", "vol_change", "ema20", "ema50", "atr", "close"]:
        if col not in data.columns:
            data[col] = 0.0
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)

    data["rsi_5mean"] = data["rsi"].rolling(5, min_periods=1).mean()
    data["rsi_20mean"] = data["rsi"].rolling(20, min_periods=1).mean()
    data["price_change"] = data["close"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    data["ema_diff_norm"] = (data["ema20"] - data["ema50"]) / (data["close"].abs() + 1e-9)
    data["atr_norm"] = data["atr"] / (data["close"].abs() + 1e-9)

    if "profit" not in data.columns or not data["profit"].notna().any():
        raise ValueError("Training requires true profit labels; ai_prob self-labeling is disabled")

    y_win = (pd.to_numeric(data["profit"], errors="coerce").fillna(0.0) > 0).astype(int)

    if "duration" in data.columns and data["duration"].notna().any():
        y_horizon = pd.to_numeric(data["duration"], errors="coerce").fillna(DEFAULT_HORIZON)
    else:
        y_horizon = pd.Series(np.full(len(data), DEFAULT_HORIZON), index=data.index)

    X = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ts = _to_dt(data["time"]).fillna(pd.Timestamp.utcnow())
    return X, y_win, y_horizon.astype(float), ts


def _fit_classifier(X_scaled: np.ndarray, y: pd.Series):
    model = RandomForestClassifier(
        n_estimators=350,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_scaled, y)
    return model


def _fit_horizon_model(X_scaled: np.ndarray, y_horizon: pd.Series):
    model = RandomForestRegressor(
        n_estimators=240,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled, y_horizon)
    return model


def walk_forward_validation(X: pd.DataFrame, y: pd.Series, y_horizon: pd.Series):
    n_splits = max(2, min(5, len(X) // 20))
    if len(X) < 40:
        logger.info("Walk-forward skipped: too few rows (%d)", len(X))
        return

    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        h_train = y_horizon.iloc[train_idx]
        h_test = y_horizon.iloc[test_idx]

        if y_train.nunique() < 2:
            logger.info("Fold %d skipped for classifier (single class)", fold)
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = _fit_classifier(X_train_scaled, y_train)
        reg = _fit_horizon_model(X_train_scaled, h_train)

        pred = clf.predict(X_test_scaled)
        if hasattr(clf, "predict_proba") and y_test.nunique() > 1:
            proba = clf.predict_proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_test, proba)
        else:
            auc = float("nan")

        hpred = reg.predict(X_test_scaled)

        fold_rows.append(
            {
                "fold": fold,
                "test_rows": len(test_idx),
                "acc": accuracy_score(y_test, pred),
                "auc": auc,
                "horizon_mae": mean_absolute_error(h_test, hpred),
            }
        )

    if fold_rows:
        val_df = pd.DataFrame(fold_rows)
        logger.info("Walk-forward validation:\n%s", val_df.to_string(index=False))
    else:
        logger.info("Walk-forward validation produced no evaluable folds")


def fit_calibrator(X: pd.DataFrame, y: pd.Series):
    if len(X) < 80 or y.nunique() < 2:
        logger.info("Calibration skipped: rows=%d classes=%d", len(X), y.nunique())
        return None

    split = int(len(X) * 0.8)
    X_train = X.iloc[:split]
    X_calib = X.iloc[split:]
    y_train = y.iloc[:split]
    y_calib = y.iloc[split:]

    if len(X_calib) < 20 or y_calib.nunique() < 2 or y_train.nunique() < 2:
        logger.info("Calibration skipped due to class imbalance in temporal holdout")
        return None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_calib_scaled = scaler.transform(X_calib)

    clf = _fit_classifier(X_train_scaled, y_train)
    probs = clf.predict_proba(X_calib_scaled)[:, 1]

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(probs, y_calib)

    calibrated_probs = calibrator.transform(probs)
    try:
        auc_before = roc_auc_score(y_calib, probs)
        auc_after = roc_auc_score(y_calib, calibrated_probs)
        logger.info("Calibration AUC before=%.4f after=%.4f", auc_before, auc_after)
    except Exception:
        logger.info("Calibration fitted (AUC unavailable)")

    return calibrator


def train_models(save_dir: str = ".") -> bool:
    df = update_dataset()
    if df.empty:
        logger.warning("No labeled data available for training")
        return False

    if len(df) < MIN_TRAIN_ROWS:
        logger.warning("Not enough rows for training: %d < %d", len(df), MIN_TRAIN_ROWS)
        return False

    X, y, y_horizon, ts = prepare_features(df)

    order = np.argsort(ts.to_numpy())
    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    y_horizon = y_horizon.iloc[order].reset_index(drop=True)

    if y.nunique() < 2:
        logger.warning("Training aborted: only one class in labels")
        return False

    walk_forward_validation(X, y, y_horizon)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model_win = _fit_classifier(X_scaled, y)
    model_horizon = _fit_horizon_model(X_scaled, y_horizon)

    pred_full = model_win.predict(X_scaled)
    logger.info("Classifier in-sample report:\n%s", classification_report(y, pred_full, digits=3))

    hpred_full = model_horizon.predict(X_scaled)
    logger.info("Horizon in-sample R2: %.4f | MAE: %.4f", r2_score(y_horizon, hpred_full), mean_absolute_error(y_horizon, hpred_full))

    calibrator = fit_calibrator(X, y)

    os.makedirs(save_dir, exist_ok=True)
    backup_model_files(save_dir)

    joblib.dump(model_win, os.path.join(save_dir, "ai_model_win.pkl"))
    joblib.dump(model_horizon, os.path.join(save_dir, "ai_model_horizon.pkl"))
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
    if calibrator is not None:
        joblib.dump(calibrator, os.path.join(save_dir, "ai_calibrator.pkl"))
    elif os.path.exists(os.path.join(save_dir, "ai_calibrator.pkl")):
        os.remove(os.path.join(save_dir, "ai_calibrator.pkl"))

    logger.info("Training completed and models saved to %s", save_dir)
    return True


def train_if_needed(retrain_interval: int = RETRAIN_INTERVAL, force: bool = False, save_dir: str = ".") -> bool:
    try:
        model_path = os.path.join(save_dir, "ai_model_win.pkl")
        if not force and os.path.exists(model_path):
            age_seconds = time.time() - os.path.getmtime(model_path)
            if age_seconds < retrain_interval:
                logger.info(
                    "Model is still fresh (%0.1fs < %0.1fs), skipping retrain",
                    age_seconds,
                    retrain_interval,
                )
                return False

        return train_models(save_dir=save_dir)
    except Exception as exc:
        logger.exception("train_if_needed failed: %s", exc)
        return False


if __name__ == "__main__":
    success = train_models()
    if success:
        logger.info("Training finished successfully")
    else:
        logger.warning("Training did not produce new models")
