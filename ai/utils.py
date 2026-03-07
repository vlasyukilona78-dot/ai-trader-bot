from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


@dataclass
class ModelBundle:
    classifier: Any | None
    regressor: Any | None
    scaler: Any | None
    calibrator: Any | None
    feature_names: list[str]


DEFAULT_FEATURE_NAMES = [
    "rsi",
    "ema20",
    "ema50",
    "atr",
    "vwap_dist",
    "bb_position",
    "volume_spike",
    "obv_div_short",
    "obv_div_long",
    "cvd_div_short",
    "cvd_div_long",
    "poc_dist",
    "vah_dist",
    "val_dist",
    "regime_num",
    "funding_rate",
    "open_interest",
    "long_short_ratio",
    "sentiment_index",
    "liq_high_dist",
    "liq_low_dist",
    "close_ret_5",
    "close_ret_20",
    "mtf_rsi_5m",
    "mtf_rsi_15m",
    "mtf_atr_norm_5m",
    "mtf_atr_norm_15m",
    "mtf_trend_5m",
    "mtf_trend_15m",
]


def _safe_load(path: Path):
    try:
        if path.exists():
            return joblib.load(path)
    except Exception:
        return None
    return None


def load_model_bundle(model_dir: str = "ai/models", regime: str | None = None) -> ModelBundle:
    base = Path(model_dir)
    suffix = f"_{regime.lower()}" if regime else ""

    cls_path = base / f"model_win{suffix}.pkl"
    reg_path = base / f"model_horizon{suffix}.pkl"
    scaler_path = base / f"scaler{suffix}.pkl"
    cal_path = base / f"calibrator{suffix}.pkl"
    feat_path = base / f"features{suffix}.pkl"

    classifier = _safe_load(cls_path)
    regressor = _safe_load(reg_path)
    scaler = _safe_load(scaler_path)
    calibrator = _safe_load(cal_path)
    feature_names = _safe_load(feat_path)

    if not isinstance(feature_names, list):
        if hasattr(scaler, "feature_names_in_"):
            feature_names = [str(x) for x in scaler.feature_names_in_]
        else:
            feature_names = DEFAULT_FEATURE_NAMES

    return ModelBundle(
        classifier=classifier,
        regressor=regressor,
        scaler=scaler,
        calibrator=calibrator,
        feature_names=feature_names,
    )


def _heuristic_prediction(feature_vector: np.ndarray) -> tuple[float, float]:
    rsi = float(feature_vector[0]) if len(feature_vector) > 0 else 50.0
    volume_spike = float(feature_vector[6]) if len(feature_vector) > 6 else 1.0
    regime_num = float(feature_vector[14]) if len(feature_vector) > 14 else 0.0

    prob = 0.50
    prob += 0.15 * np.tanh((volume_spike - 1.0) / 4.0)
    prob += 0.10 * np.tanh(abs(rsi - 50.0) / 30.0)
    prob += 0.08 * np.tanh(abs(regime_num) / 2.0)
    prob = float(max(0.01, min(0.99, prob)))

    horizon = 8.0 + 8.0 * abs(regime_num)
    horizon = float(max(1.0, min(96.0, horizon)))
    return prob, horizon


def predict_with_bundle(bundle: ModelBundle, feature_row: dict[str, float]) -> tuple[float, float]:
    values = [float(feature_row.get(name, 0.0)) for name in bundle.feature_names]
    arr = np.array([values], dtype=float)

    if bundle.classifier is None or bundle.regressor is None:
        return _heuristic_prediction(arr[0])

    try:
        if bundle.scaler is not None:
            if hasattr(bundle.scaler, "feature_names_in_"):
                frame = pd.DataFrame(arr, columns=bundle.scaler.feature_names_in_)
                arr_scaled = bundle.scaler.transform(frame)
            else:
                arr_scaled = bundle.scaler.transform(arr)
        else:
            arr_scaled = arr

        if hasattr(bundle.classifier, "predict_proba"):
            p = bundle.classifier.predict_proba(arr_scaled)[0]
            prob = float(p[-1]) if len(p) > 1 else float(p[0])
        else:
            prob = float(bundle.classifier.predict(arr_scaled)[0])

        horizon = float(bundle.regressor.predict(arr_scaled)[0])

        if bundle.calibrator is not None and hasattr(bundle.calibrator, "transform"):
            try:
                prob = float(bundle.calibrator.transform([prob])[0])
            except Exception:
                pass

        if not np.isfinite(prob):
            prob = 0.5
        if not np.isfinite(horizon):
            horizon = 8.0

        prob = float(max(0.0, min(1.0, prob)))
        horizon = float(max(1.0, min(240.0, horizon)))
        return prob, horizon
    except Exception:
        return _heuristic_prediction(arr[0])


def save_feature_names(feature_names: list[str], model_dir: str = "ai/models", regime: str | None = None):
    os.makedirs(model_dir, exist_ok=True)
    suffix = f"_{regime.lower()}" if regime else ""
    path = Path(model_dir) / f"features{suffix}.pkl"
    joblib.dump(feature_names, path)
