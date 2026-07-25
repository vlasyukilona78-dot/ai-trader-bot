from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


from core.levels import find_horizontal_levels, liquidation_histogram, nearest_level_above
from core.volume_profile import VolumeProfileLevels


def build_signal_chart_with_liquidations(
    symbol: str,
    df: pd.DataFrame,
    side: str,
    entry: float,
    tp: float,
    sl: float,
    volume_profile: VolumeProfileLevels | None = None,
    *,
    show_levels: bool = True,
    max_levels: int = 4,
) -> bytes | None:
    """Signal chart with an estimated liquidation-density panel beside it.

    The density panel shows where long liquidations are likely stacked: those
    clusters are the fuel that turns a fading pump into a cascade, so seeing them
    next to the entry makes the setup judgeable at a glance.
    """
    if df.empty or len(df) < 10:
        return None

    frame = df.tail(120).copy()
    x = range(len(frame))

    fig = plt.figure(figsize=(12, 6), facecolor="#0b1220")
    grid = fig.add_gridspec(1, 5, wspace=0.04)
    ax = fig.add_subplot(grid[0, :4])
    ax_liq = fig.add_subplot(grid[0, 4], sharey=ax)

    for axis in (ax, ax_liq):
        axis.set_facecolor("#0b1220")
        axis.tick_params(colors="white")
        for spine in axis.spines.values():
            spine.set_color("#33415c")

    ax.plot(x, frame["close"], color="#d7e3ff", linewidth=1.6, label="Close")
    for col, colour, label in (("ema20", "#00d4ff", "EMA20"), ("ema50", "#ffad00", "EMA50")):
        if col in frame.columns:
            ax.plot(x, frame[col], color=colour, linewidth=1.1, label=label)

    ax.axhline(entry, color="#4ea8de", linestyle="--", linewidth=1.3, label="Entry")
    ax.axhline(tp, color="#2dc653", linestyle="--", linewidth=1.3, label="TP")
    ax.axhline(sl, color="#ef476f", linestyle="--", linewidth=1.3, label="SL")

    if volume_profile is not None:
        ax.axhline(volume_profile.poc, color="#ffd166", linestyle=":", linewidth=1.1, label="POC")

    if show_levels:
        levels = find_horizontal_levels(frame)
        overhead = nearest_level_above(levels, entry)
        strongest = sorted(levels, key=lambda lv: -lv.strength)[:max_levels]
        for lv in strongest:
            is_key = overhead is not None and abs(lv.price - overhead.price) < 1e-12
            ax.axhline(
                lv.price,
                color="#c77dff" if is_key else "#4a4e69",
                linestyle="-" if is_key else ":",
                linewidth=1.4 if is_key else 0.9,
                alpha=0.9 if is_key else 0.6,
                label="Key level" if is_key else None,
            )

    centres, density = liquidation_histogram(frame)
    if len(centres):
        ax_liq.barh(centres, density, height=(centres[1] - centres[0]) if len(centres) > 1 else None,
                    color="#ff5d8f", alpha=0.75)
        peak = centres[int(density.argmax())]
        ax_liq.axhline(peak, color="#ffd166", linestyle="--", linewidth=1.2)
        ax_liq.text(0.5, peak, " peak", color="#ffd166", fontsize=7, va="bottom")
    ax_liq.set_title("Liq. map\n(estimated)", color="#ff5d8f", fontsize=8)
    ax_liq.set_xticks([])
    plt.setp(ax_liq.get_yticklabels(), visible=False)

    ax.set_title(f"{symbol} | {side}", color="white")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


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
