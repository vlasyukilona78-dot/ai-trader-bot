from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.feature_engineering import compute_mtf_feature_snapshot
from core.indicators import compute_indicators
from core.market_data import MarketDataClient


RUNTIME_DIR = ROOT / "data" / "runtime"
BAR_HORIZONS: tuple[int, ...] = (3, 5, 10, 20)


@dataclass
class ExitEvent:
    profile: str
    symbol: str
    action: str
    exec_status: str
    ts: float
    exit_price: float | None
    entry_price: float | None
    tp: float | None
    sl: float | None
    realized_pnl: float
    stopped_out: bool
    exit_type: str
    managed_exit_reason: str
    order_link_id: str
    raw: dict[str, Any]


def _iso_utc(ts: float | None) -> str:
    if ts is None:
        return ""
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(out):
        return default
    return out


def _round(value: Any, ndigits: int = 3) -> float:
    return round(_safe_float(value), ndigits)


def _normalize_action(value: str) -> str:
    return str(value or "").strip().upper()


def _extract_float(raw: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key not in raw:
            continue
        try:
            value = float(raw.get(key))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _load_db_rows(db_path: Path) -> list[ExitEvent]:
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT id, symbol, action, exec_status, ts, order_link_id, raw_json
        FROM order_decisions
        ORDER BY id ASC
        """
    ).fetchall()
    conn.close()

    out: list[ExitEvent] = []
    for row in rows:
        try:
            raw = json.loads(str(row["raw_json"]))
            if not isinstance(raw, dict):
                raw = {}
        except Exception:
            raw = {}

        action = str(row["action"])
        exec_status = str(row["exec_status"])
        if _normalize_action(action) != "EXIT_SHORT":
            continue
        if _normalize_action(exec_status) != "FILLED":
            continue

        raw_execution = raw.get("execution_context", {})
        raw_intent = raw.get("intent_context", {})
        exit_type = str(raw.get("exit_type") or "")
        managed_exit_reason = str(raw.get("managed_exit_reason") or raw_intent.get("reason") or "")
        out.append(
            ExitEvent(
                profile="main" if "main" in db_path.name else "early",
                symbol=str(row["symbol"]),
                action=action,
                exec_status=exec_status,
                ts=float(row["ts"]),
                exit_price=_extract_float(raw, "exit_price", "avg_price", "price"),
                entry_price=_extract_float(raw, "entry_price", "entry", "entry_px"),
                tp=_extract_float(raw, "tp_price", "take_profit", "tp", "take_profit_price"),
                sl=_extract_float(raw, "sl_price", "stop_loss", "sl", "stop_loss_price"),
                realized_pnl=_safe_float(raw.get("realized_pnl"), _safe_float(raw_execution.get("realized_pnl"), 0.0)),
                stopped_out=bool(raw.get("stopped_out")) or exit_type.lower() == "stop_loss",
                exit_type=exit_type,
                managed_exit_reason=managed_exit_reason,
                order_link_id=str(row["order_link_id"] or ""),
                raw=raw,
            )
        )
    return out


def _select_recent_exits(rows: list[ExitEvent], limit: int) -> list[ExitEvent]:
    return rows[-max(1, int(limit)) :]


def _timeframe_seconds(interval: str) -> int:
    text = str(interval).strip().upper()
    if text.isdigit():
        return max(int(text), 1) * 60
    mapping = {
        "D": 86400,
        "1D": 86400,
        "W": 7 * 86400,
        "1W": 7 * 86400,
    }
    return mapping.get(text, 60)


def _fetch_symbol_ohlcv(
    client: MarketDataClient,
    symbol: str,
    interval: str,
    *,
    limit: int = 1000,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> pd.DataFrame:
    df = client.fetch_ohlcv(
        symbol=symbol,
        interval=interval,
        limit=limit,
        start_ms=start_ms,
        end_ms=end_ms,
    )
    if df.empty:
        return df
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def _fetch_exit_window(client: MarketDataClient, event: ExitEvent, timeframe: str) -> pd.DataFrame:
    tf_sec = _timeframe_seconds(timeframe)
    pre_sec = max(180 * 60, tf_sec * 120)
    post_sec = max(120 * 60, tf_sec * 120)
    start_ms = int((float(event.ts) - pre_sec) * 1000)
    end_ms = int((float(event.ts) + post_sec) * 1000)
    candles = int((pre_sec + post_sec) / tf_sec) + 20
    return _fetch_symbol_ohlcv(
        client,
        event.symbol,
        timeframe,
        limit=min(max(candles, 160), 1000),
        start_ms=start_ms,
        end_ms=end_ms,
    )


def _exit_context(ind_df: pd.DataFrame, bar_ts: pd.Timestamp, exit_price: float) -> dict[str, Any]:
    pre = ind_df.loc[ind_df.index <= bar_ts].copy()
    if pre.empty:
        return {}

    bar = pre.iloc[-1]
    mtf = compute_mtf_feature_snapshot(pre[["open", "high", "low", "close", "volume"]])
    vwap_dist = _safe_float(bar.get("vwap_dist"), 0.0)
    ema20 = _safe_float(bar.get("ema20"), exit_price)
    ema50 = _safe_float(bar.get("ema50"), exit_price)

    return {
        "rsi": _round(bar.get("rsi"), 2),
        "volume_spike": _round(bar.get("volume_spike"), 2),
        "adx": _round(bar.get("adx"), 2),
        "vwap_dist_pct": _round(vwap_dist * 100.0, 3),
        "ema20_gap_pct": _round((exit_price - ema20) / exit_price * 100.0 if exit_price > 0 else 0.0, 3),
        "ema50_gap_pct": _round((exit_price - ema50) / exit_price * 100.0 if exit_price > 0 else 0.0, 3),
        "mtf_rsi_5m": _round(mtf.get("mtf_rsi_5m"), 2),
        "mtf_rsi_15m": _round(mtf.get("mtf_rsi_15m"), 2),
        "mtf_trend_5m_pct": _round(_safe_float(mtf.get("mtf_trend_5m"), 0.0) * 100.0, 3),
        "mtf_trend_15m_pct": _round(_safe_float(mtf.get("mtf_trend_15m"), 0.0) * 100.0, 3),
        "mtf_trend_1h_pct": _round(_safe_float(mtf.get("mtf_trend_1h"), 0.0) * 100.0, 3),
    }


def _bars_until_short_exit_move_pct(
    window: pd.DataFrame,
    *,
    exit_price: float,
    direction: str,
    threshold_pct: float,
) -> int | None:
    if exit_price <= 0 or window.empty or threshold_pct <= 0:
        return None
    direction_key = str(direction).strip().lower()
    if direction_key not in {"further_down", "rebound"}:
        return None

    for idx, (_, row) in enumerate(window.iterrows(), start=1):
        high_px = _safe_float(row.get("high"), exit_price)
        low_px = _safe_float(row.get("low"), exit_price)
        if direction_key == "further_down":
            move_pct = (exit_price - low_px) / exit_price * 100.0
        else:
            move_pct = (high_px - exit_price) / exit_price * 100.0
        if move_pct >= threshold_pct:
            return idx
    return None


def _minutes_from_bars(bars: int | None, *, timeframe: str) -> float | None:
    if bars is None:
        return None
    tf_sec = _timeframe_seconds(timeframe)
    return round((float(bars) * tf_sec) / 60.0, 3)


def _window_exit_metrics_short(window: pd.DataFrame, *, exit_price: float) -> dict[str, float]:
    if exit_price <= 0 or window.empty:
        return {
            "bars_observed": 0.0,
            "further_favorable_pct": 0.0,
            "rebound_pct": 0.0,
            "close_move_pct": 0.0,
        }

    max_high = _safe_float(window["high"].max(), exit_price)
    min_low = _safe_float(window["low"].min(), exit_price)
    close_px = _safe_float(window.iloc[-1].get("close"), exit_price)
    further_favorable = (exit_price - min_low) / exit_price * 100.0
    rebound = (max_high - exit_price) / exit_price * 100.0
    close_move = (exit_price - close_px) / exit_price * 100.0
    return {
        "bars_observed": float(len(window)),
        "further_favorable_pct": round(further_favorable, 3),
        "rebound_pct": round(rebound, 3),
        "close_move_pct": round(close_move, 3),
    }


def _classify_short_exit_quality(
    *,
    realized_pnl: float,
    stopped_out: bool,
    further_15: float,
    rebound_15: float,
    further_60: float,
    rebound_60: float,
    bars_to_further_down: int | None,
    bars_to_rebound: int | None,
) -> str:
    if stopped_out:
        if further_60 >= max(0.75, rebound_60 * 1.1):
            return "stopped_then_reversed"
        return "protective_exit"

    if realized_pnl < 0:
        if rebound_60 >= max(0.5, further_60 * 1.1):
            return "late_or_bad"
        if further_60 >= max(0.75, rebound_60 * 1.2):
            return "cut_before_reversal"
        return "mixed"

    if (
        further_60 >= max(0.8, rebound_60 * 1.2)
        and (bars_to_rebound is None or (bars_to_further_down is not None and bars_to_further_down + 2 < bars_to_rebound))
    ):
        return "too_early"

    if rebound_60 >= max(0.5, further_60 * 1.05):
        return "timely"

    if further_15 < 0.25 and rebound_15 < 0.25:
        return "neutral"

    return "mixed"


def _summarize_results(items: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"count": len(items), "verdict_counts": {}}
    if not items:
        return summary

    verdict_counts: dict[str, int] = {}
    exit_type_counts: dict[str, int] = {}
    managed_reason_counts: dict[str, int] = {}
    realized_pnl: list[float] = []
    further_15: list[float] = []
    rebound_15: list[float] = []
    further_60: list[float] = []
    rebound_60: list[float] = []
    minutes_to_further_down: list[float] = []
    minutes_to_rebound: list[float] = []
    stopped_out_count = 0
    horizon_accumulator: dict[str, dict[str, list[float]]] = {
        str(horizon): {
            "further_favorable_pct": [],
            "rebound_pct": [],
            "close_move_pct": [],
        }
        for horizon in BAR_HORIZONS
    }

    for item in items:
        verdict = str(item.get("verdict") or "unknown")
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        exit_type = str(item.get("exit_type") or "")
        if exit_type:
            exit_type_counts[exit_type] = exit_type_counts.get(exit_type, 0) + 1
        managed_reason = str(item.get("managed_exit_reason") or "")
        if managed_reason:
            managed_reason_counts[managed_reason] = managed_reason_counts.get(managed_reason, 0) + 1
        realized_pnl.append(_safe_float(item.get("realized_pnl"), 0.0))
        further_15.append(_safe_float(item.get("further_favorable_15m_pct"), 0.0))
        rebound_15.append(_safe_float(item.get("rebound_15m_pct"), 0.0))
        further_60.append(_safe_float(item.get("further_favorable_60m_pct"), 0.0))
        rebound_60.append(_safe_float(item.get("rebound_60m_pct"), 0.0))
        if bool(item.get("stopped_out")):
            stopped_out_count += 1
        first_down = item.get("minutes_to_further_down_035pct")
        first_up = item.get("minutes_to_rebound_035pct")
        if first_down is not None:
            minutes_to_further_down.append(_safe_float(first_down, 0.0))
        if first_up is not None:
            minutes_to_rebound.append(_safe_float(first_up, 0.0))
        raw_horizons = item.get("bar_horizons", {})
        if isinstance(raw_horizons, dict):
            for horizon in BAR_HORIZONS:
                horizon_key = str(horizon)
                payload = raw_horizons.get(horizon_key, {})
                if not isinstance(payload, dict):
                    continue
                for metric_key in ("further_favorable_pct", "rebound_pct", "close_move_pct"):
                    if metric_key in payload:
                        horizon_accumulator[horizon_key][metric_key].append(_safe_float(payload.get(metric_key), 0.0))

    def _avg(values: list[float]) -> float:
        if not values:
            return 0.0
        return round(sum(values) / len(values), 3)

    total = max(len(items), 1)
    summary.update(
        {
            "verdict_counts": verdict_counts,
            "exit_type_counts": dict(sorted(exit_type_counts.items(), key=lambda item: (-item[1], item[0]))),
            "managed_exit_reason_counts": dict(sorted(managed_reason_counts.items(), key=lambda item: (-item[1], item[0]))),
            "avg_realized_pnl": _avg(realized_pnl),
            "stopped_out_rate": round(stopped_out_count / total, 3),
            "avg_further_favorable_15m_pct": _avg(further_15),
            "avg_rebound_15m_pct": _avg(rebound_15),
            "avg_further_favorable_60m_pct": _avg(further_60),
            "avg_rebound_60m_pct": _avg(rebound_60),
            "avg_minutes_to_further_down_035pct": _avg(minutes_to_further_down),
            "avg_minutes_to_rebound_035pct": _avg(minutes_to_rebound),
            "bar_horizons": {
                horizon_key: {
                    f"avg_{metric_key}": _avg(values)
                    for metric_key, values in metrics.items()
                }
                for horizon_key, metrics in horizon_accumulator.items()
            },
        }
    )
    return summary


def _analyze_exit(client: MarketDataClient, event: ExitEvent, *, timeframe: str = "1") -> dict[str, Any]:
    df = _fetch_exit_window(client, event, timeframe)
    if df.empty:
        return {
            "profile": event.profile,
            "symbol": event.symbol,
            "exit_ts": _iso_utc(event.ts),
            "status": "no_market_data",
        }

    ind_df = compute_indicators(df)
    exit_dt = pd.Timestamp(datetime.fromtimestamp(event.ts, tz=timezone.utc))
    at_or_after = ind_df.loc[ind_df.index >= exit_dt]
    if at_or_after.empty:
        return {
            "profile": event.profile,
            "symbol": event.symbol,
            "exit_ts": _iso_utc(event.ts),
            "status": "exit_after_available_data",
        }

    bar_ts = at_or_after.index[0]
    exit_price = float(event.exit_price or at_or_after.iloc[0]["close"])

    window_15 = at_or_after.iloc[:15]
    window_60 = at_or_after.iloc[:60]
    first10 = at_or_after.iloc[:10]

    metrics_15 = _window_exit_metrics_short(window_15, exit_price=exit_price)
    metrics_60 = _window_exit_metrics_short(window_60, exit_price=exit_price)

    first_reaction = "flat"
    if not first10.empty:
        first_down = metrics_15["further_favorable_pct"] if len(first10) == len(window_15) else _window_exit_metrics_short(first10, exit_price=exit_price)["further_favorable_pct"]
        first_up = metrics_15["rebound_pct"] if len(first10) == len(window_15) else _window_exit_metrics_short(first10, exit_price=exit_price)["rebound_pct"]
        if first_down >= first_up * 1.1 and first_down >= 0.2:
            first_reaction = "down"
        elif first_up >= first_down * 1.1 and first_up >= 0.2:
            first_reaction = "up"

    bars_to_further_down = _bars_until_short_exit_move_pct(
        at_or_after,
        exit_price=exit_price,
        direction="further_down",
        threshold_pct=0.35,
    )
    bars_to_rebound = _bars_until_short_exit_move_pct(
        at_or_after,
        exit_price=exit_price,
        direction="rebound",
        threshold_pct=0.35,
    )

    horizon_metrics: dict[str, dict[str, float]] = {}
    for horizon in BAR_HORIZONS:
        horizon_metrics[str(horizon)] = _window_exit_metrics_short(
            at_or_after.iloc[:horizon],
            exit_price=exit_price,
        )

    verdict = _classify_short_exit_quality(
        realized_pnl=float(event.realized_pnl),
        stopped_out=bool(event.stopped_out),
        further_15=metrics_15["further_favorable_pct"],
        rebound_15=metrics_15["rebound_pct"],
        further_60=metrics_60["further_favorable_pct"],
        rebound_60=metrics_60["rebound_pct"],
        bars_to_further_down=bars_to_further_down,
        bars_to_rebound=bars_to_rebound,
    )

    return {
        "profile": event.profile,
        "symbol": event.symbol,
        "action": event.action,
        "exit_ts": _iso_utc(event.ts),
        "exit_price": round(exit_price, 8),
        "entry_price": round(float(event.entry_price or 0.0), 8) if event.entry_price else None,
        "tp": round(float(event.tp or 0.0), 8) if event.tp else None,
        "sl": round(float(event.sl or 0.0), 8) if event.sl else None,
        "realized_pnl": round(float(event.realized_pnl), 6),
        "stopped_out": bool(event.stopped_out),
        "exit_type": event.exit_type,
        "managed_exit_reason": event.managed_exit_reason,
        "first_reaction": first_reaction,
        "further_favorable_15m_pct": metrics_15["further_favorable_pct"],
        "rebound_15m_pct": metrics_15["rebound_pct"],
        "further_favorable_60m_pct": metrics_60["further_favorable_pct"],
        "rebound_60m_pct": metrics_60["rebound_pct"],
        "bars_to_further_down_035pct": bars_to_further_down,
        "bars_to_rebound_035pct": bars_to_rebound,
        "minutes_to_further_down_035pct": _minutes_from_bars(bars_to_further_down, timeframe=timeframe),
        "minutes_to_rebound_035pct": _minutes_from_bars(bars_to_rebound, timeframe=timeframe),
        "verdict": verdict,
        "bar_horizons": horizon_metrics,
        "context": _exit_context(ind_df, bar_ts, exit_price),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze recent short exit quality from runtime DB against Bybit OHLCV.")
    parser.add_argument("--main-db", default=str(RUNTIME_DIR / "v2_demo_runtime_main.db"))
    parser.add_argument("--early-db", default=str(RUNTIME_DIR / "v2_demo_runtime_early.db"))
    parser.add_argument("--main-limit", type=int, default=12)
    parser.add_argument("--early-limit", type=int, default=12)
    parser.add_argument("--timeframe", default="1")
    args = parser.parse_args()

    main_rows = _load_db_rows(Path(args.main_db))
    early_rows = _load_db_rows(Path(args.early_db))

    main_exits = _select_recent_exits(main_rows, int(args.main_limit))
    early_exits = _select_recent_exits(early_rows, int(args.early_limit))

    client = MarketDataClient(timeout=12, max_retries=2)
    try:
        main_analyzed = [_analyze_exit(client, row, timeframe=str(args.timeframe)) for row in main_exits]
        early_analyzed = [_analyze_exit(client, row, timeframe=str(args.timeframe)) for row in early_exits]
    finally:
        client.close()

    analyzed = {
        "main": main_analyzed,
        "early": early_analyzed,
        "summary": {
            "main": _summarize_results(main_analyzed),
            "early": _summarize_results(early_analyzed),
        },
    }
    print(json.dumps(analyzed, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
