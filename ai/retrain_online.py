from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from ai.train import train_models


@dataclass
class OnlineRetrainConfig:
    dataset_path: str = "data/processed/training_dataset.csv"
    model_dir: str = "ai/models"
    retrain_interval_sec: int = 6 * 3600
    min_new_rows: int = 200


class OnlineRetrainer:
    def __init__(self, config: OnlineRetrainConfig):
        self.config = config
        self.last_train_ts: float = 0.0
        self.last_row_count: int = 0

    def _row_count(self) -> int:
        if not os.path.exists(self.config.dataset_path):
            return 0
        try:
            df = pd.read_csv(self.config.dataset_path)
            return len(df)
        except Exception:
            return 0

    def maybe_retrain(self, model_type: str = "auto") -> bool:
        now = time.time()
        rows = self._row_count()

        if rows <= 0:
            return False

        interval_ok = (now - self.last_train_ts) >= self.config.retrain_interval_sec
        growth_ok = (rows - self.last_row_count) >= self.config.min_new_rows

        if not (interval_ok and growth_ok):
            return False

        try:
            train_models(
                dataset_path=self.config.dataset_path,
                model_dir=self.config.model_dir,
                model_type=model_type,
                regime=None,
            )
        except Exception:
            return False

        self.last_train_ts = now
        self.last_row_count = rows
        return True


