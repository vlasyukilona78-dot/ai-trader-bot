from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import classification_report, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from ai.utils import DEFAULT_FEATURE_NAMES, save_feature_names

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:
    LGBMClassifier = None
    LGBMRegressor = None


def _select_model_type(model_type: str) -> str:
    mt = model_type.lower().strip()
    if mt == "xgboost" and XGBClassifier is not None and XGBRegressor is not None:
        return "xgboost"
    if mt == "lightgbm" and LGBMClassifier is not None and LGBMRegressor is not None:
        return "lightgbm"
    if mt in ("auto", "xgboost", "lightgbm"):
        if XGBClassifier is not None and XGBRegressor is not None:
            return "xgboost"
        if LGBMClassifier is not None and LGBMRegressor is not None:
            return "lightgbm"
    return "sklearn"


def _make_models(model_type: str):
    if model_type == "xgboost":
        clf = XGBClassifier(
            n_estimators=350,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        )
        reg = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
        return clf, reg

    if model_type == "lightgbm":
        clf = LGBMClassifier(
            n_estimators=350,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        reg = LGBMRegressor(
            n_estimators=320,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        return clf, reg

    clf = GradientBoostingClassifier(random_state=42)
    reg = GradientBoostingRegressor(random_state=42)
    return clf, reg


def _prepare_xy(df: pd.DataFrame, features: list[str]):
    required_labels = {"target_win", "target_horizon"}
    missing = required_labels - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing target columns: {sorted(missing)}")

    for col in features:
        if col not in df.columns:
            df[col] = 0.0

    X = df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_win = pd.to_numeric(df["target_win"], errors="coerce").fillna(0.0).astype(int)
    y_horizon = pd.to_numeric(df["target_horizon"], errors="coerce").fillna(8.0)
    return X, y_win, y_horizon


def _fit_calibrator(y_true: pd.Series, probs: np.ndarray) -> IsotonicRegression | None:
    if len(y_true) < 100 or y_true.nunique() < 2:
        return None
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(probs, y_true)
    return cal


def _temporal_split(X: pd.DataFrame, y: pd.Series, y_h: pd.Series):
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    h_train = y_h.iloc[:split_idx]
    h_test = y_h.iloc[split_idx:]
    return X_train, X_test, y_train, y_test, h_train, h_test


def train_models(
    dataset_path: str,
    model_dir: str = "ai/models",
    model_type: str = "auto",
    regime: str | None = None,
):
    df = pd.read_csv(dataset_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp")

    if regime:
        if "market_regime" not in df.columns:
            raise ValueError("Dataset does not have market_regime column")
        df = df[df["market_regime"].astype(str).str.upper() == regime.upper()]

    df = df.reset_index(drop=True)
    if len(df) < 120:
        raise ValueError(f"Not enough rows for training: {len(df)}")

    features = [f for f in DEFAULT_FEATURE_NAMES if f in df.columns]
    if not features:
        features = DEFAULT_FEATURE_NAMES

    X, y, y_h = _prepare_xy(df, features)
    if y.nunique() < 2:
        raise ValueError("target_win has only one class")

    selected_model_type = _select_model_type(model_type)
    clf, reg = _make_models(selected_model_type)

    X_train, X_test, y_train, y_test, h_train, h_test = _temporal_split(X, y, y_h)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf.fit(X_train_s, y_train)
    reg.fit(X_train_s, h_train)

    pred = clf.predict(X_test_s)
    if hasattr(clf, "predict_proba") and y_test.nunique() > 1:
        probs = clf.predict_proba(X_test_s)[:, 1]
        auc = roc_auc_score(y_test, probs)
    else:
        probs = pred.astype(float)
        auc = float("nan")

    h_pred = reg.predict(X_test_s)

    print("Model type:", selected_model_type)
    print("Classifier report:")
    print(classification_report(y_test, pred, digits=3))
    print("AUC:", round(float(auc), 4) if np.isfinite(auc) else "n/a")
    print("Horizon MAE:", round(float(mean_absolute_error(h_test, h_pred)), 4))
    print("Horizon R2:", round(float(r2_score(h_test, h_pred)), 4))

    calibrator = _fit_calibrator(y_test, probs)

    os.makedirs(model_dir, exist_ok=True)
    suffix = f"_{regime.lower()}" if regime else ""

    joblib.dump(clf, Path(model_dir) / f"model_win{suffix}.pkl")
    joblib.dump(reg, Path(model_dir) / f"model_horizon{suffix}.pkl")
    joblib.dump(scaler, Path(model_dir) / f"scaler{suffix}.pkl")
    if calibrator is not None:
        joblib.dump(calibrator, Path(model_dir) / f"calibrator{suffix}.pkl")

    save_feature_names(features, model_dir=model_dir, regime=regime)

    # walk-forward diagnostics
    tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(X) // 80)))
    wf_rows: list[dict[str, float]] = []
    for fold_idx, (tr_idx, te_idx) in enumerate(tscv.split(X), start=1):
        X_tr = X.iloc[tr_idx]
        X_te = X.iloc[te_idx]
        y_tr = y.iloc[tr_idx]
        y_te = y.iloc[te_idx]
        if y_tr.nunique() < 2 or y_te.nunique() < 2:
            continue

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
        clf_fold, _ = _make_models(selected_model_type)
        clf_fold.fit(X_tr_s, y_tr)
        if hasattr(clf_fold, "predict_proba"):
            p = clf_fold.predict_proba(X_te_s)[:, 1]
            fold_auc = roc_auc_score(y_te, p)
        else:
            fold_auc = float("nan")

        wf_rows.append({"fold": float(fold_idx), "auc": float(fold_auc)})

    if wf_rows:
        wf = pd.DataFrame(wf_rows)
        print("Walk-forward AUC mean:", round(float(wf["auc"].mean()), 4))


def parse_args():
    parser = argparse.ArgumentParser(description="Train ML confirmation models for crypto bot")
    parser.add_argument("--dataset", required=True, help="Path to training CSV")
    parser.add_argument("--model-dir", default="ai/models", help="Output directory for .pkl models")
    parser.add_argument("--model-type", default="auto", choices=["auto", "xgboost", "lightgbm", "sklearn"])
    parser.add_argument("--regime", default="", help="Optional market regime filter: TREND/RANGE/PUMP/PANIC")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        train_models(
            dataset_path=args.dataset,
            model_dir=args.model_dir,
            model_type=args.model_type,
            regime=args.regime if args.regime else None,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    except ValueError as exc:
        raise SystemExit(str(exc))



