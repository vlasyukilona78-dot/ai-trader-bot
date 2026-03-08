from __future__ import annotations

import pandas as pd

from ai.train import train_models
from ai.training.validate import chronological_split, validate_no_feature_leakage


def train_with_validation(dataset_path: str, model_dir: str, model_type: str = "auto"):
    df = pd.read_csv(dataset_path)
    validate_no_feature_leakage(df)
    train_df, val_df, test_df = chronological_split(df)
    _ = (val_df, test_df)  # reserved for metric gates in next iterations

    tmp_dataset = train_df
    tmp_path = dataset_path + ".train_only.tmp.csv"
    tmp_dataset.to_csv(tmp_path, index=False)
    try:
        train_models(dataset_path=tmp_path, model_dir=model_dir, model_type=model_type, regime=None)
    finally:
        import os

        if os.path.exists(tmp_path):
            os.remove(tmp_path)
