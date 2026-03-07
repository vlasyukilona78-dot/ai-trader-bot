from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.feature_engineering import REQUIRED_MODEL_FEATURES, build_feature_row
from core.indicators import compute_indicators
from core.market_regime import detect_market_regime
from core.volume_profile import compute_volume_profile


def load_ohlcv(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"OHLCV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        dt = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    elif "time" in df.columns:
        raw = pd.to_numeric(df["time"], errors="coerce")
        unit = "ms" if raw.dropna().median() > 10_000_000_000 else "s"
        dt = pd.to_datetime(raw, unit=unit, utc=True, errors="coerce")
    else:
        raise ValueError("Need datetime/timestamp/time column")

    df["datetime"] = dt
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close", "volume"])


def build_dataset(df: pd.DataFrame, lookahead: int = 24, atr_mult: float = 1.0, rr: float = 2.0) -> pd.DataFrame:
    enriched = compute_indicators(df)
    rows: list[dict] = []

    for i in range(80, len(enriched) - lookahead - 1):
        hist = enriched.iloc[: i + 1]
        future = enriched.iloc[i + 1 : i + 1 + lookahead]
        if len(future) < lookahead:
            continue

        vp = compute_volume_profile(hist)
        regime = detect_market_regime(hist)

        feat = build_feature_row(
            symbol="DATA/USDT",
            df=hist,
            volume_profile=vp,
            regime=regime,
            extras={},
        )
        if feat is None:
            continue

        entry = float(hist.iloc[-1]["close"])
        atr = float(hist.iloc[-1].get("atr", entry * 0.01) or entry * 0.01)
        sl_long = entry - atr * atr_mult
        tp_long = entry + (entry - sl_long) * rr

        exit_idx = None
        exit_price = None
        for j in range(len(future)):
            hi = float(future.iloc[j]["high"])
            lo = float(future.iloc[j]["low"])
            if hi >= tp_long:
                exit_idx = j + 1
                exit_price = tp_long
                break
            if lo <= sl_long:
                exit_idx = j + 1
                exit_price = sl_long
                break

        if exit_idx is None:
            exit_idx = lookahead
            exit_price = float(future.iloc[-1]["close"])

        pnl = (exit_price - entry) / max(entry, 1e-9)
        target_win = 1 if pnl > 0 else 0
        target_horizon = float(exit_idx)

        row = {
            "timestamp": hist.index[-1],
            "market_regime": regime.value,
            "target_win": target_win,
            "target_horizon": target_horizon,
            "future_return": pnl,
        }
        for name in REQUIRED_MODEL_FEATURES:
            row[name] = float(feat.values.get(name, 0.0))

        rows.append(row)

    out = pd.DataFrame(rows)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Build ML dataset from OHLCV")
    parser.add_argument("--input", required=True, help="Path to OHLCV CSV")
    parser.add_argument("--output", default="data/processed/training_dataset.csv")
    parser.add_argument("--lookahead", type=int, default=24)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        df = load_ohlcv(args.input)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))

    dataset = build_dataset(df, lookahead=int(args.lookahead))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(out, index=False)
    print(f"rows={len(dataset)}")
    print(f"saved={out}")

