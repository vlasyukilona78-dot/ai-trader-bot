from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
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

from ai.inference.governance import compute_feature_schema_hash, load_registry, save_registry
from ai.missing import MissingnessPolicy
from ai.ood import fit_envelope
from ai.splitting import temporal_split_3
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
    """Extract features and labels, preserving absence as absence.

    A feature that was not computed stays NaN rather than becoming 0.0. Zero is
    a meaningful reading for the divergence and spike features, so filling with
    it would make "not measured" indistinguishable from "measured as neutral".
    Imputation happens later, fitted on training rows only.
    """

    required_labels = {"target_win", "target_horizon"}
    missing = required_labels - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing target columns: {sorted(missing)}")

    for col in features:
        if col not in df.columns:
            df[col] = np.nan

    X = df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y_win = pd.to_numeric(df["target_win"], errors="coerce").fillna(0.0).astype(int)
    y_horizon = pd.to_numeric(df["target_horizon"], errors="coerce").fillna(8.0)
    return X, y_win, y_horizon


def _fit_calibrator(y_true: pd.Series, probs: np.ndarray) -> IsotonicRegression | None:
    if len(y_true) < 100 or y_true.nunique() < 2:
        return None
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(probs, y_true)
    return cal


def _reliability(y_true: pd.Series, probs: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: population-weighted gap between predicted
    probability and realised rate, measured on equal-width bins."""

    outcomes = np.asarray(y_true, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if outcomes.size == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, bins + 1)
    assigned = np.clip(np.digitize(probs, edges[1:-1]), 0, bins - 1)

    weighted_gap = 0.0
    for b in range(bins):
        in_bin = assigned == b
        count = int(in_bin.sum())
        if count == 0:
            continue
        weighted_gap += count * abs(probs[in_bin].mean() - outcomes[in_bin].mean())
    return float(weighted_gap / outcomes.size)


def _write_artifact_manifest(
    *,
    model_dir: str,
    version: str,
    selected_model_type: str,
    dataset_path: str,
    features: list[str],
    rows: int,
    split: dict | None = None,
):
    manifest = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_type": selected_model_type,
        "dataset_path": str(dataset_path),
        "rows": int(rows),
        "feature_names": list(features),
        "feature_schema_hash": compute_feature_schema_hash(list(features)),
        "split": split or {},
    }
    path = Path(model_dir) / ("manifest.json" if version == "default" else f"manifest_{version}.json")
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    registry = load_registry(model_dir)
    registry.setdefault("auto_promotion", False)
    registry.setdefault("history", [])
    if version == "default" and not registry.get("champion"):
        registry["champion"] = "default"
    elif version != registry.get("champion"):
        registry["challenger"] = version
    registry["history"].append(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": "register_artifact",
            "version": version,
            "model_type": selected_model_type,
        }
    )
    save_registry(model_dir, registry)


def train_models(
    dataset_path: str,
    model_dir: str = "ai/models",
    model_type: str = "auto",
    regime: str | None = None,
    train_frac: float = 0.70,
    calib_frac: float = 0.15,
    embargo: int = 0,
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

    split = temporal_split_3(
        n=len(X),
        horizons=y_h.tolist(),
        train_frac=train_frac,
        calib_frac=calib_frac,
        embargo=embargo,
    )
    train_rows = list(split.train_idx)
    calib_rows = list(split.calib_idx)
    test_rows = list(split.test_idx)

    X_train, y_train, h_train = X.iloc[train_rows], y.iloc[train_rows], y_h.iloc[train_rows]
    X_calib, y_calib = X.iloc[calib_rows], y.iloc[calib_rows]
    X_test, y_test, h_test = X.iloc[test_rows], y.iloc[test_rows], y_h.iloc[test_rows]

    # The support envelope records the raw observed ranges, before imputation
    # replaces gaps with medians that were never actually observed.
    envelope_regimes = tuple(
        sorted(df.loc[train_rows, "market_regime"].astype(str).str.upper().unique())
    ) if "market_regime" in df.columns else ("UNKNOWN",)
    ood_envelope = fit_envelope(
        X_train.dropna(axis=1, how="all"),
        version=f"{regime.lower() if regime else 'default'}-{len(train_rows)}",
        regimes=envelope_regimes,
    )

    # Imputation is learned from training rows only; a median over the whole
    # dataset would carry later information back into training.
    missing_policy = MissingnessPolicy(add_indicators=True).fit(X_train)
    X_train = missing_policy.transform(X_train)
    X_calib = missing_policy.transform(X_calib)
    X_test = missing_policy.transform(X_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_calib_s = scaler.transform(X_calib)
    X_test_s = scaler.transform(X_test)

    clf.fit(X_train_s, y_train)
    reg.fit(X_train_s, h_train)

    has_proba = hasattr(clf, "predict_proba")

    # The calibrator is fitted on its own interval so the test interval stays
    # untouched and can measure how well the calibrated probabilities hold up.
    calib_probs = clf.predict_proba(X_calib_s)[:, 1] if has_proba else clf.predict(X_calib_s).astype(float)
    calibrator = _fit_calibrator(y_calib, calib_probs)

    pred = clf.predict(X_test_s)
    if has_proba and y_test.nunique() > 1:
        probs = clf.predict_proba(X_test_s)[:, 1]
        auc = roc_auc_score(y_test, probs)
    else:
        probs = pred.astype(float)
        auc = float("nan")

    h_pred = reg.predict(X_test_s)

    ece_raw = _reliability(y_test, probs)
    ece_calibrated = (
        _reliability(y_test, calibrator.predict(probs)) if calibrator is not None else float("nan")
    )

    print("Model type:", selected_model_type)
    print(
        f"Split: train={len(split.train_idx)} calib={len(split.calib_idx)} "
        f"test={len(split.test_idx)} | purged {split.purged_train}/{split.purged_calib} "
        f"| embargo {split.embargo}"
    )
    print("Classifier report:")
    print(classification_report(y_test, pred, digits=3))
    print("AUC:", round(float(auc), 4) if np.isfinite(auc) else "n/a")
    print("Horizon MAE:", round(float(mean_absolute_error(h_test, h_pred)), 4))
    print("Horizon R2:", round(float(r2_score(h_test, h_pred)), 4))
    print("ECE raw:", round(ece_raw, 4))
    print("ECE calibrated:", round(ece_calibrated, 4) if np.isfinite(ece_calibrated) else "n/a")

    os.makedirs(model_dir, exist_ok=True)
    suffix = f"_{regime.lower()}" if regime else ""

    joblib.dump(clf, Path(model_dir) / f"model_win{suffix}.pkl")
    joblib.dump(reg, Path(model_dir) / f"model_horizon{suffix}.pkl")
    joblib.dump(scaler, Path(model_dir) / f"scaler{suffix}.pkl")
    joblib.dump(missing_policy, Path(model_dir) / f"missing_policy{suffix}.pkl")
    joblib.dump(ood_envelope, Path(model_dir) / f"ood_envelope{suffix}.pkl")
    if calibrator is not None:
        joblib.dump(calibrator, Path(model_dir) / f"calibrator{suffix}.pkl")

    summary = {
        "train_rows": len(split.train_idx),
        "calibration_rows": len(split.calib_idx),
        "test_rows": len(split.test_idx),
        "calibration_fit_rows": len(y_calib),
        "metrics_rows": len(y_test),
        "purged_train": split.purged_train,
        "purged_calibration": split.purged_calib,
        "embargo": split.embargo,
        "calibrator_fitted": calibrator is not None,
        "train_missing_rate": missing_policy.train_missing_rate,
        "ood_envelope_version": ood_envelope.version,
        "ood_regimes": list(ood_envelope.valid_regimes),
        "auc": float(auc),
        "ece_raw": float(ece_raw),
        "ece_calibrated": float(ece_calibrated),
    }

    save_feature_names(features, model_dir=model_dir, regime=regime)
    _write_artifact_manifest(
        model_dir=model_dir,
        version=(regime.lower() if regime else "default"),
        selected_model_type=selected_model_type,
        dataset_path=dataset_path,
        features=features,
        rows=len(df),
        split=summary,
    )

    tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(X) // 80)))
    wf_rows: list[dict[str, float]] = []
    for fold_idx, (tr_idx, te_idx) in enumerate(tscv.split(X), start=1):
        X_tr = X.iloc[tr_idx]
        X_te = X.iloc[te_idx]
        y_tr = y.iloc[tr_idx]
        y_te = y.iloc[te_idx]
        if y_tr.nunique() < 2 or y_te.nunique() < 2:
            continue

        # Each fold learns its own imputation from its own training rows.
        fold_policy = MissingnessPolicy(add_indicators=True).fit(X_tr)
        X_tr = fold_policy.transform(X_tr)
        X_te = fold_policy.transform(X_te)

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
        walk_forward_auc = float(wf["auc"].mean())
        print("Walk-forward AUC mean:", round(walk_forward_auc, 4))
        summary["walk_forward_auc"] = walk_forward_auc

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Train ML confirmation models for crypto bot")
    parser.add_argument("--dataset", required=True, help="Path to training CSV")
    parser.add_argument("--model-dir", default="ai/models", help="Output directory for .pkl models")
    parser.add_argument("--model-type", default="auto", choices=["auto", "xgboost", "lightgbm", "sklearn"])
    parser.add_argument("--regime", default="", help="Optional market regime filter: TREND/RANGE/PUMP/PANIC")
    parser.add_argument("--train-frac", type=float, default=0.70, help="Share of rows used for training")
    parser.add_argument("--calib-frac", type=float, default=0.15, help="Share of rows used to fit the calibrator")
    parser.add_argument(
        "--embargo",
        type=int,
        default=0,
        help="Extra bars of separation at each interval boundary, on top of the label horizon",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        train_models(
            dataset_path=args.dataset,
            model_dir=args.model_dir,
            model_type=args.model_type,
            regime=args.regime if args.regime else None,
            train_frac=args.train_frac,
            calib_frac=args.calib_frac,
            embargo=args.embargo,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    except ValueError as exc:
        raise SystemExit(str(exc))
