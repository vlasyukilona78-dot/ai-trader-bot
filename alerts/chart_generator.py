from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


from core.volume_profile import VolumeProfileLevels


def build_signal_chart(
    symbol: str,
    df: pd.DataFrame,
    side: str,
    entry: float,
    tp: float,
    sl: float,
    volume_profile: VolumeProfileLevels | None = None,
) -> bytes | None:
    if df.empty or len(df) < 10:
        return None

    frame = df.tail(120).copy()
    x = range(len(frame))

    fig = plt.figure(figsize=(10, 6), facecolor="#0b1220")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#0b1220")
    ax.plot(x, frame["close"], color="#d7e3ff", linewidth=1.6, label="Close")

    if "ema20" in frame.columns:
        ax.plot(x, frame["ema20"], color="#00d4ff", linewidth=1.2, label="EMA20")
    if "ema50" in frame.columns:
        ax.plot(x, frame["ema50"], color="#ffad00", linewidth=1.2, label="EMA50")
    if "vwap" in frame.columns:
        ax.plot(x, frame["vwap"], color="#7cf29a", linewidth=1.0, label="VWAP")

    ax.axhline(entry, color="#4ea8de", linestyle="--", linewidth=1.2, label="Entry")
    ax.axhline(tp, color="#2dc653", linestyle="--", linewidth=1.2, label="TP")
    ax.axhline(sl, color="#ef476f", linestyle="--", linewidth=1.2, label="SL")

    if volume_profile is not None:
        ax.axhline(volume_profile.poc, color="#ffd166", linestyle=":", linewidth=1.2, label="POC")
        ax.axhline(volume_profile.vah, color="#f9844a", linestyle=":", linewidth=1.0, label="VAH")
        ax.axhline(volume_profile.val, color="#90be6d", linestyle=":", linewidth=1.0, label="VAL")

    ax.set_title(f"{symbol} | {side}", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#33415c")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(loc="best", fontsize=8)

    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()
