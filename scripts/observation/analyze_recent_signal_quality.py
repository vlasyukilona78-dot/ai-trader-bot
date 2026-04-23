from __future__ import annotations

import argparse
import json
import re
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
from core.volume_profile import compute_volume_profile


RUNTIME_DIR = ROOT / "data" / "runtime"
LOG_DIR = ROOT / "logs" / "runtime"


@dataclass
class SignalEvent:
    profile: str
    kind: str
    symbol: str
    action: str
    exec_status: str
    ts: float
    delivery_ts: float | None
    entry_price: float | None
    tp: float | None
    sl: float | None
    risk_reason: str
    exec_reason: str
    order_link_id: str
    raw: dict[str, Any]


def _iso_utc(ts: float | None) -> str:
    if ts is None:
        return ""
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()


def _latest_live_log() -> Path:
    candidates = sorted(LOG_DIR.glob("bot_supervisor_live_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    candidates = sorted(LOG_DIR.glob("bot_supervisor_*.stdout.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No runtime log found in {LOG_DIR}")
    return candidates[0]


def _load_db_rows(db_path: Path) -> list[SignalEvent]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT id, symbol, action, state_before, risk_reason, exec_status, exec_reason,
               order_id, order_link_id, ts, raw_json
        FROM order_decisions
        ORDER BY id ASC
        """
    ).fetchall()
    conn.close()

    events: list[SignalEvent] = []
    for row in rows:
        try:
            raw = json.loads(str(row["raw_json"]))
            if not isinstance(raw, dict):
                raw = {}
        except Exception:
            raw = {}

        entry_price = _extract_float(raw, "entry_price", "entry", "entry_px", "price")
        tp = _extract_float(raw, "tp_price", "take_profit", "tp", "take_profit_price")
        sl = _extract_float(raw, "sl_price", "stop_loss", "sl", "stop_loss_price")

        events.append(
            SignalEvent(
                profile="main" if "main" in db_path.name else "early",
                kind="db_decision",
                symbol=str(row["symbol"]),
                action=str(row["action"]),
                exec_status=str(row["exec_status"]),
                ts=float(row["ts"]),
                delivery_ts=None,
                entry_price=entry_price,
                tp=tp,
                sl=sl,
                risk_reason=str(row["risk_reason"]),
                exec_reason=str(row["exec_reason"]),
                order_link_id=str(row["order_link_id"]),
                raw=raw,
            )
        )
    return events


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


def _parse_log_for_early_alerts(log_path: Path) -> list[SignalEvent]:
    line_re = re.compile(r"^\[(?P<profile>[^\]]+)\]\s+(?P<payload>\{.*\})$")
    pending: dict[str, list[SignalEvent]] = {}
    alerts: list[SignalEvent] = []

    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            match = line_re.match(raw_line.strip())
            if not match:
                continue
            profile = match.group("profile")
            if profile != "early":
                continue
            try:
                payload = json.loads(match.group("payload"))
            except Exception:
                continue
            msg = str(payload.get("msg") or "")
            ts_raw = str(payload.get("ts") or "")
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).timestamp()
            except Exception:
                continue

            if "reason=early_profile_monitor_only" in msg and "symbol=" in msg:
                symbol = _extract_token(msg, "symbol")
                if not symbol:
                    continue
                queue = pending.setdefault(profile, [])
                queue.append(
                    SignalEvent(
                    profile="early",
                    kind="early_alert",
                    symbol=symbol,
                    action="SHORT_ENTRY",
                    exec_status="IGNORED",
                    ts=ts,
                    delivery_ts=None,
                    entry_price=None,
                    tp=None,
                    sl=None,
                    risk_reason="approved",
                    exec_reason="early_profile_monitor_only",
                    order_link_id="",
                    raw={"source_log": str(log_path)},
                    )
                )
                if len(queue) > 24:
                    pending[profile] = queue[-24:]
                continue

            if payload.get("event") == "early_signal_alert_delivery" and "attempted=2 sent=2" in msg:
                queue = pending.get(profile) or []
                event = None
                while queue:
                    candidate = queue.pop()
                    if 0 <= ts - candidate.ts <= 1800:
                        event = candidate
                        break
                if event is None:
                    continue
                event.delivery_ts = ts
                alerts.append(event)

    return alerts


def _extract_token(msg: str, key: str) -> str:
    marker = f"{key}="
    idx = msg.find(marker)
    if idx < 0:
        return ""
    value = msg[idx + len(marker):].split(" ", 1)[0].strip()
    return value


def _select_recent_main_entries(rows: list[SignalEvent], limit: int) -> list[SignalEvent]:
    entries = [
        row
        for row in rows
        if row.profile == "main"
        and _normalize_action(row.action) == "SHORT_ENTRY"
        and _normalize_action(row.exec_status) == "FILLED"
    ]
    return entries[-limit:]


def _select_recent_early_alerts(rows: list[SignalEvent], limit: int) -> list[SignalEvent]:
    return rows[-limit:]


def _normalize_action(value: str) -> str:
    return str(value or "").strip().upper()


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


def _timeframe_seconds(interval: str) -> int:
    text = str(interval).strip().upper()
    if text.isdigit():
        return max(int(text), 1) * 60
    mapping = {
        "D": 86400,
        "1D": 86400,
        "W": 7 * 86400,
        "1W": 7 * 86400,
        "M": 30 * 86400,
        "1M": 30 * 86400,
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


def _fetch_signal_window(client: MarketDataClient, event: SignalEvent, timeframe: str) -> pd.DataFrame:
    tf_sec = _timeframe_seconds(timeframe)
    pre_sec = max(240 * 60, tf_sec * 180)
    post_sec = max(90 * 60, tf_sec * 90)
    start_ms = int((float(event.ts) - pre_sec) * 1000)
    end_ms = int((float(event.ts) + post_sec) * 1000)
    candles = int((pre_sec + post_sec) / tf_sec) + 20
    return _fetch_symbol_ohlcv(
        client,
        event.symbol,
        timeframe,
        limit=min(max(candles, 180), 1000),
        start_ms=start_ms,
        end_ms=end_ms,
    )


def _signal_context(ind_df: pd.DataFrame, bar_ts: pd.Timestamp, entry: float) -> dict[str, Any]:
    pre = ind_df.loc[ind_df.index <= bar_ts].copy()
    if pre.empty:
        return {}

    bar = pre.iloc[-1]
    open_px = _safe_float(bar.get("open"), entry)
    high_px = _safe_float(bar.get("high"), entry)
    low_px = _safe_float(bar.get("low"), entry)
    close_px = _safe_float(bar.get("close"), entry)
    candle_range = max(high_px - low_px, 1e-12)
    upper_wick = max(high_px - max(open_px, close_px), 0.0)
    lower_wick = max(min(open_px, close_px) - low_px, 0.0)

    recent_20 = pre.tail(20)
    recent_60 = pre.tail(60)
    recent_high_20 = _safe_float(recent_20["high"].max(), close_px) if not recent_20.empty else close_px
    recent_high_60 = _safe_float(recent_60["high"].max(), close_px) if not recent_60.empty else close_px
    recent_low_60 = _safe_float(recent_60["low"].min(), close_px) if not recent_60.empty else close_px

    vp = compute_volume_profile(pre[["open", "high", "low", "close", "volume"]], window=120, bins=48)
    liq_high, liq_low = MarketDataClient.estimate_liquidation_clusters(
        pre[["open", "high", "low", "close", "volume"]],
        window=80,
    )
    mtf = compute_mtf_feature_snapshot(pre[["open", "high", "low", "close", "volume"]])

    bb_upper = _safe_float(bar.get("bb_upper"), close_px)
    kc_upper = _safe_float(bar.get("kc_upper"), close_px)
    ema20 = _safe_float(bar.get("ema20"), close_px)
    ema50 = _safe_float(bar.get("ema50"), close_px)
    atr = _safe_float(bar.get("atr"), 0.0)

    return {
        "rsi": _round(bar.get("rsi"), 2),
        "volume_spike": _round(bar.get("volume_spike"), 2),
        "vwap_dist_pct": _round(_safe_float(bar.get("vwap_dist"), 0.0) * 100.0, 3),
        "bb_position": _round(bar.get("bb_position"), 3),
        "above_bb_upper": close_px > bb_upper,
        "above_kc_upper": close_px > kc_upper,
        "atr_norm_pct": _round((atr / close_px * 100.0) if close_px > 0 else 0.0, 3),
        "ema20_gap_pct": _round((close_px - ema20) / close_px * 100.0 if close_px > 0 else 0.0, 3),
        "ema50_gap_pct": _round((close_px - ema50) / close_px * 100.0 if close_px > 0 else 0.0, 3),
        "adx": _round(bar.get("adx"), 2),
        "body_pct": _round((close_px - open_px) / open_px * 100.0 if open_px > 0 else 0.0, 3),
        "upper_wick_pct": _round(upper_wick / close_px * 100.0 if close_px > 0 else 0.0, 3),
        "lower_wick_pct": _round(lower_wick / close_px * 100.0 if close_px > 0 else 0.0, 3),
        "close_position_in_candle": _round((close_px - low_px) / candle_range, 3),
        "dist_to_recent_high_20_pct": _round((recent_high_20 - close_px) / close_px * 100.0 if close_px > 0 else 0.0, 3),
        "dist_to_recent_high_60_pct": _round((recent_high_60 - close_px) / close_px * 100.0 if close_px > 0 else 0.0, 3),
        "dist_from_recent_low_60_pct": _round((close_px - recent_low_60) / close_px * 100.0 if close_px > 0 else 0.0, 3),
        "poc_dist_pct": _round((close_px - vp.poc) / close_px * 100.0 if vp and close_px > 0 else 0.0, 3),
        "vah_dist_pct": _round((close_px - vp.vah) / close_px * 100.0 if vp and close_px > 0 else 0.0, 3),
        "val_dist_pct": _round((close_px - vp.val) / close_px * 100.0 if vp and close_px > 0 else 0.0, 3),
        "liq_high_dist_pct": _round((close_px - liq_high) / close_px * 100.0 if liq_high and close_px > 0 else 0.0, 3),
        "liq_low_dist_pct": _round((close_px - liq_low) / close_px * 100.0 if liq_low and close_px > 0 else 0.0, 3),
        "mtf_rsi_5m": _round(mtf.get("mtf_rsi_5m"), 2),
        "mtf_rsi_15m": _round(mtf.get("mtf_rsi_15m"), 2),
        "mtf_trend_5m_pct": _round(_safe_float(mtf.get("mtf_trend_5m"), 0.0) * 100.0, 3),
        "mtf_trend_15m_pct": _round(_safe_float(mtf.get("mtf_trend_15m"), 0.0) * 100.0, 3),
    }


def _analyze_signal(client: MarketDataClient, event: SignalEvent, *, timeframe: str = "1") -> dict[str, Any]:
    df = _fetch_signal_window(client, event, timeframe)
    if df.empty:
        return {
            "profile": event.profile,
            "kind": event.kind,
            "symbol": event.symbol,
            "signal_ts": _iso_utc(event.ts),
            "status": "no_market_data",
        }

    ind_df = compute_indicators(df)
    signal_dt = pd.Timestamp(datetime.fromtimestamp(event.ts, tz=timezone.utc))
    at_or_after = ind_df.loc[ind_df.index >= signal_dt]
    if at_or_after.empty:
        return {
            "profile": event.profile,
            "kind": event.kind,
            "symbol": event.symbol,
            "signal_ts": _iso_utc(event.ts),
            "status": "signal_after_available_data",
        }

    bar = at_or_after.iloc[0]
    bar_ts = at_or_after.index[0]
    entry = float(event.entry_price or bar["close"])

    window_15 = at_or_after.iloc[:15]
    window_30 = at_or_after.iloc[:30]
    window_60 = at_or_after.iloc[:60]

    max_high_15 = float(window_15["high"].max()) if not window_15.empty else entry
    min_low_15 = float(window_15["low"].min()) if not window_15.empty else entry
    max_high_60 = float(window_60["high"].max()) if not window_60.empty else entry
    min_low_60 = float(window_60["low"].min()) if not window_60.empty else entry

    up_15 = (max_high_15 - entry) / entry * 100 if entry > 0 else 0.0
    down_15 = (entry - min_low_15) / entry * 100 if entry > 0 else 0.0
    up_60 = (max_high_60 - entry) / entry * 100 if entry > 0 else 0.0
    down_60 = (entry - min_low_60) / entry * 100 if entry > 0 else 0.0

    first10 = at_or_after.iloc[:10]
    first_reaction = "flat"
    if not first10.empty:
        first_min = float(first10["low"].min())
        first_max = float(first10["high"].max())
        first_down = (entry - first_min) / entry * 100 if entry > 0 else 0.0
        first_up = (first_max - entry) / entry * 100 if entry > 0 else 0.0
        if first_down >= max(0.35, first_up + 0.1):
            first_reaction = "down"
        elif first_up >= max(0.35, first_down + 0.1):
            first_reaction = "up"

    tp_hit_60 = bool(event.tp and min_low_60 <= float(event.tp))
    sl_hit_60 = bool(event.sl and max_high_60 >= float(event.sl))

    verdict = "worked"
    if sl_hit_60:
        verdict = "failed"
    elif event.kind == "early_alert" and first_reaction == "up" and up_15 >= max(1.0, down_15 * 1.1) and up_60 >= max(1.5, down_60 * 1.15):
        verdict = "too_early"
    elif first_reaction == "up" and up_15 >= max(0.55, down_15 * 1.15) and up_60 >= max(0.75, down_60 * 1.10):
        verdict = "late_or_weak"
    elif first_reaction == "flat" and down_15 < 0.25:
        verdict = "weak"
    elif down_60 < 0.35 and up_15 > 0.35:
        verdict = "weak"
    elif event.kind == "early_alert" and first_reaction != "down" and down_60 < 0.5:
        verdict = "too_early"

    return {
        "profile": event.profile,
        "kind": event.kind,
        "symbol": event.symbol,
        "signal_ts": _iso_utc(event.ts),
        "delivery_ts": _iso_utc(event.delivery_ts),
        "signal_bar_ts": str(bar_ts.isoformat()),
        "entry_ref": round(entry, 8),
        "tp": event.tp,
        "sl": event.sl,
        "first_reaction_10m": first_reaction,
        "upside_15m_pct": round(up_15, 3),
        "downside_15m_pct": round(down_15, 3),
        "upside_60m_pct": round(up_60, 3),
        "downside_60m_pct": round(down_60, 3),
        "tp_hit_60m": tp_hit_60,
        "sl_hit_60m": sl_hit_60,
        "verdict": verdict,
        "context": _signal_context(ind_df, bar_ts, entry),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze recent signal quality from runtime DB/logs against Bybit OHLCV.")
    parser.add_argument("--main-db", default=str(RUNTIME_DIR / "v2_demo_runtime_main.db"))
    parser.add_argument("--early-db", default=str(RUNTIME_DIR / "v2_demo_runtime_early.db"))
    parser.add_argument("--log", default=str(_latest_live_log()))
    parser.add_argument("--main-limit", type=int, default=12)
    parser.add_argument("--early-limit", type=int, default=12)
    parser.add_argument("--timeframe", default="1")
    args = parser.parse_args()

    main_rows = _load_db_rows(Path(args.main_db))
    early_log_rows = _parse_log_for_early_alerts(Path(args.log))

    main_entries = _select_recent_main_entries(main_rows, int(args.main_limit))
    early_entries = early_log_rows[-int(args.early_limit):]

    client = MarketDataClient(timeout=12, max_retries=2)
    try:
        analyzed = {
            "main": [_analyze_signal(client, row, timeframe=str(args.timeframe)) for row in main_entries],
            "early": [_analyze_signal(client, row, timeframe=str(args.timeframe)) for row in early_entries],
        }
    finally:
        client.close()

    print(json.dumps(analyzed, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
