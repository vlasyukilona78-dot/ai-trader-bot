from __future__ import annotations

from pathlib import Path

from ai.build_dataset import build_dataset, load_ohlcv


def build_training_dataset(input_csv: str, output_csv: str, lookahead: int = 24) -> int:
    df = load_ohlcv(input_csv)
    out = build_dataset(df, lookahead=lookahead)
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return len(out)
