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
BAR_HORIZONS: tuple[int, ...] = (3, 5, 10, 20)
LOCAL_OBSERVATIONS_PATH = RUNTIME_DIR / "signal_observations_completed.jsonl"


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
    patterns = (
        "bot_supervisor_live_*.log",
        "bot_supervisor_*.stdout.log",
        "bot_*.stderr.log",
        "bot_*.stdout.log",
    )
    candidates = sorted(
        {
            path
            for pattern in patterns
            for path in LOG_DIR.glob(pattern)
            if path.is_file() and path.stat().st_size > 0
        },
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No runtime log found in {LOG_DIR}")
    return candidates[0]


def _load_db_rows(db_path: Path) -> list[SignalEvent]:
    if not db_path.exists() or db_path.stat().st_size <= 0:
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='order_decisions'"
        ).fetchone()
        if table_exists is None:
            return []
        rows = conn.execute(
            """
            SELECT id, symbol, action, state_before, risk_reason, exec_status, exec_reason,
                   order_id, order_link_id, ts, raw_json
            FROM order_decisions
            ORDER BY id ASC
            """
        ).fetchall()
    finally:
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


def _load_signal_event_log(path: Path) -> list[SignalEvent]:
    if not path.exists():
        return []

    events: list[SignalEvent] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            try:
                raw = json.loads(text)
            except Exception:
                continue
            if not isinstance(raw, dict):
                continue

            side = str(raw.get("side") or "").upper()
            if side != "SHORT":
                continue
            if int(_safe_float(raw.get("delivery_sent"), 0.0)) <= 0:
                continue

            phase = str(raw.get("phase") or "").upper()
            reason = str(raw.get("reason") or "")
            is_ultra = phase.startswith("ULTRA")
            profile = "early" if phase in {"WATCH", "SETUP"} or is_ultra or "early" in reason else "main"
            if is_ultra:
                kind = "ultra_alert"
            elif phase in {"WATCH", "SETUP"}:
                kind = "early_alert"
            else:
                kind = "signal_alert"

            events.append(
                SignalEvent(
                    profile=profile,
                    kind=kind,
                    symbol=str(raw.get("symbol") or "").upper(),
                    action="SHORT_ENTRY",
                    exec_status="ALERT_SENT",
                    ts=_safe_float(raw.get("ts"), 0.0),
                    delivery_ts=_safe_float(raw.get("ts"), 0.0),
                    entry_price=_extract_float(raw, "entry", "entry_price", "price"),
                    tp=_extract_float(raw, "tp", "take_profit", "take_profit_price"),
                    sl=_extract_float(raw, "sl", "stop_loss", "stop_loss_price"),
                    risk_reason="alert",
                    exec_reason=reason,
                    order_link_id=str(raw.get("signal_id") or ""),
                    raw=raw,
                )
            )

    return [event for event in events if event.symbol and event.ts > 0]


def _load_completed_signal_observations(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except (TypeError, ValueError):
                continue
            if not isinstance(row, dict):
                continue
            signal_id = str(row.get("signal_id") or "").strip()
            if signal_id and str(row.get("status") or "") == "completed":
                rows[signal_id] = row
    return rows


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


def _dedupe_recent_alerts(rows: list[SignalEvent], *, cooldown_sec: int = 900) -> list[SignalEvent]:
    """Remove exact cross-process alert duplicates while preserving later unique re-tests."""
    deduped: list[SignalEvent] = []
    last_seen: dict[tuple[str, str, str, float, float, float], float] = {}

    for row in sorted(rows, key=lambda item: float(item.ts)):
        phase = str(row.raw.get("phase") or row.kind).upper() if isinstance(row.raw, dict) else row.kind.upper()
        key = (
            str(row.symbol).upper(),
            str(row.profile).lower(),
            phase,
            round(float(row.entry_price or 0.0), 8),
            round(float(row.tp or 0.0), 8),
            round(float(row.sl or 0.0), 8),
        )
        last_ts = last_seen.get(key)
        if last_ts is not None and float(row.ts) - last_ts <= float(cooldown_sec):
            continue
        last_seen[key] = float(row.ts)
        deduped.append(row)

    return deduped


def _is_early_kind(kind: str) -> bool:
    return str(kind or "").lower() in {"early_alert", "ultra_alert", "watch", "setup", "ultra"}


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


def _select_filled_short_entries(rows: list[SignalEvent], *, profile: str | None = None) -> list[SignalEvent]:
    out: list[SignalEvent] = []
    normalized_profile = str(profile or "").strip().lower()
    for row in rows:
        if normalized_profile and str(row.profile).strip().lower() != normalized_profile:
            continue
        if _normalize_action(row.action) != "SHORT_ENTRY":
            continue
        if _normalize_action(row.exec_status) != "FILLED":
            continue
        out.append(row)
    return out


def _enrich_early_alerts_with_db_execution(
    alerts: list[SignalEvent],
    early_rows: list[SignalEvent],
    *,
    max_match_delay_sec: int = 1800,
) -> list[SignalEvent]:
    if not alerts or not early_rows:
        return alerts

    by_symbol: dict[str, list[SignalEvent]] = {}
    for row in _select_filled_short_entries(early_rows, profile="early"):
        by_symbol.setdefault(str(row.symbol), []).append(row)

    for symbol_rows in by_symbol.values():
        symbol_rows.sort(key=lambda item: float(item.ts))

    used_keys: set[tuple[str, float, str]] = set()
    for alert in alerts:
        symbol_rows = by_symbol.get(str(alert.symbol), [])
        best: SignalEvent | None = None
        best_lag: float | None = None
        for candidate in symbol_rows:
            candidate_key = (str(candidate.symbol), float(candidate.ts), str(candidate.order_link_id))
            if candidate_key in used_keys:
                continue
            lag = float(candidate.ts) - float(alert.ts)
            if lag < -60.0 or lag > float(max_match_delay_sec):
                continue
            if best is None or lag < float(best_lag):
                best = candidate
                best_lag = lag
        if best is None:
            continue

        used_keys.add((str(best.symbol), float(best.ts), str(best.order_link_id)))
        alert.entry_price = alert.entry_price or best.entry_price
        alert.tp = alert.tp or best.tp
        alert.sl = alert.sl or best.sl
        alert.action = best.action or alert.action
        alert.exec_status = best.exec_status or alert.exec_status
        alert.risk_reason = best.risk_reason or alert.risk_reason
        alert.exec_reason = best.exec_reason or alert.exec_reason
        alert.order_link_id = best.order_link_id or alert.order_link_id
        enriched_raw = dict(alert.raw)
        enriched_raw["matched_early_decision_ts"] = _iso_utc(best.ts)
        if best.order_link_id:
            enriched_raw["matched_early_order_link_id"] = best.order_link_id
        alert.raw = enriched_raw

    return alerts


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


def _bars_until_move_pct(
    window: pd.DataFrame,
    *,
    entry: float,
    direction: str,
    threshold_pct: float,
) -> int | None:
    if entry <= 0 or window.empty or threshold_pct <= 0:
        return None
    direction_key = str(direction).strip().lower()
    if direction_key not in {"up", "down"}:
        return None

    for idx, (_, row) in enumerate(window.iterrows(), start=1):
        high_px = _safe_float(row.get("high"), entry)
        low_px = _safe_float(row.get("low"), entry)
        if direction_key == "down":
            move_pct = (entry - low_px) / entry * 100.0
        else:
            move_pct = (high_px - entry) / entry * 100.0
        if move_pct >= threshold_pct:
            return idx
    return None


def _minutes_from_bars(bars: int | None, *, timeframe: str) -> float | None:
    if bars is None:
        return None
    tf_sec = _timeframe_seconds(timeframe)
    return round((float(bars) * tf_sec) / 60.0, 3)


def _window_outcome_metrics(window: pd.DataFrame, *, entry: float) -> dict[str, float]:
    if entry <= 0 or window.empty:
        return {
            "bars_observed": 0.0,
            "favorable_excursion_pct": 0.0,
            "adverse_excursion_pct": 0.0,
            "close_move_pct": 0.0,
        }

    max_high = _safe_float(window["high"].max(), entry)
    min_low = _safe_float(window["low"].min(), entry)
    close_px = _safe_float(window.iloc[-1].get("close"), entry)
    favorable = (entry - min_low) / entry * 100.0
    adverse = (max_high - entry) / entry * 100.0
    close_move = (entry - close_px) / entry * 100.0
    return {
        "bars_observed": float(len(window)),
        "favorable_excursion_pct": round(favorable, 3),
        "adverse_excursion_pct": round(adverse, 3),
        "close_move_pct": round(close_move, 3),
    }


def _classify_short_signal_outcome(
    *,
    kind: str,
    first_reaction: str,
    up_15: float,
    down_15: float,
    up_60: float,
    down_60: float,
    tp_hit_60: bool,
    sl_hit_60: bool,
    bars_to_first_down_move: int | None,
    bars_to_first_up_move: int | None,
) -> str:
    if sl_hit_60:
        return "failed"

    if tp_hit_60 and down_15 >= max(0.35, up_15 * 0.8):
        return "worked"

    if (
        first_reaction == "up"
        and bars_to_first_down_move is None
        and up_15 >= max(0.8, down_15 * 1.2)
        and up_60 >= max(1.0, down_60 * 1.15)
    ):
        return "continuation_trap"

    if (
        bars_to_first_up_move is not None
        and (bars_to_first_down_move is None or bars_to_first_up_move + 2 < bars_to_first_down_move)
        and up_15 >= max(0.55, down_15 * 1.15)
    ):
        return "late_or_weak"

    if bars_to_first_down_move is not None and bars_to_first_down_move >= 8 and down_60 < max(0.75, up_15):
        return "late_or_weak"

    if _is_early_kind(kind) and first_reaction == "up" and up_15 >= max(1.0, down_15 * 1.1) and up_60 >= max(1.5, down_60 * 1.15):
        return "too_early"

    if first_reaction == "up" and up_15 >= max(0.55, down_15 * 1.15) and up_60 >= max(0.75, down_60 * 1.10):
        return "late_or_weak"

    if first_reaction == "flat" and down_15 < 0.25:
        return "weak"

    if down_60 < 0.35 and up_15 > 0.35:
        return "weak"

    if _is_early_kind(kind) and first_reaction != "down" and down_60 < 0.5:
        return "too_early"

    return "worked"


def _summarize_results(items: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "count": len(items),
        "verdict_counts": {},
    }
    if not items:
        return summary

    verdict_counts: dict[str, int] = {}
    favorable_15: list[float] = []
    adverse_15: list[float] = []
    favorable_60: list[float] = []
    adverse_60: list[float] = []
    minutes_to_first_down: list[float] = []
    minutes_to_first_up: list[float] = []
    horizon_accumulator: dict[str, dict[str, list[float]]] = {
        str(horizon): {
            "favorable_excursion_pct": [],
            "adverse_excursion_pct": [],
            "close_move_pct": [],
        }
        for horizon in BAR_HORIZONS
    }
    tp_hits = 0
    sl_hits = 0

    for item in items:
        verdict = str(item.get("verdict") or "unknown")
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        favorable_15.append(_safe_float(item.get("favorable_excursion_15m_pct"), 0.0))
        adverse_15.append(_safe_float(item.get("adverse_excursion_15m_pct"), 0.0))
        favorable_60.append(_safe_float(item.get("favorable_excursion_60m_pct"), 0.0))
        adverse_60.append(_safe_float(item.get("adverse_excursion_60m_pct"), 0.0))
        first_down = item.get("minutes_to_first_down_035pct")
        first_up = item.get("minutes_to_first_up_035pct")
        if first_down is not None:
            minutes_to_first_down.append(_safe_float(first_down, 0.0))
        if first_up is not None:
            minutes_to_first_up.append(_safe_float(first_up, 0.0))
        if bool(item.get("tp_hit_60m")):
            tp_hits += 1
        if bool(item.get("sl_hit_60m")):
            sl_hits += 1
        raw_horizons = item.get("bar_horizons", {})
        if isinstance(raw_horizons, dict):
            for horizon in BAR_HORIZONS:
                horizon_key = str(horizon)
                horizon_payload = raw_horizons.get(horizon_key, {})
                if not isinstance(horizon_payload, dict):
                    continue
                for metric_key in ("favorable_excursion_pct", "adverse_excursion_pct", "close_move_pct"):
                    if metric_key in horizon_payload:
                        horizon_accumulator[horizon_key][metric_key].append(
                            _safe_float(horizon_payload.get(metric_key), 0.0)
                        )

    def _avg(values: list[float]) -> float:
        if not values:
            return 0.0
        return round(sum(values) / len(values), 3)

    total = max(len(items), 1)
    summary.update(
        {
            "verdict_counts": verdict_counts,
            "tp_hit_rate_60m": round(tp_hits / total, 3),
            "sl_hit_rate_60m": round(sl_hits / total, 3),
            "avg_favorable_excursion_15m_pct": _avg(favorable_15),
            "avg_adverse_excursion_15m_pct": _avg(adverse_15),
            "avg_favorable_excursion_60m_pct": _avg(favorable_60),
            "avg_adverse_excursion_60m_pct": _avg(adverse_60),
            "avg_minutes_to_first_down_035pct": _avg(minutes_to_first_down),
            "avg_minutes_to_first_up_035pct": _avg(minutes_to_first_up),
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


def _analyze_local_observation(
    event: SignalEvent,
    observation: dict[str, Any],
    *,
    timeframe: str,
) -> dict[str, Any] | None:
    raw_horizons = observation.get("horizon_metrics")
    if not isinstance(raw_horizons, dict):
        return None
    metrics_15 = raw_horizons.get("15")
    metrics_60 = raw_horizons.get("60")
    if not isinstance(metrics_15, dict) or not isinstance(metrics_60, dict):
        return None

    down_15 = _safe_float(metrics_15.get("favorable_excursion_pct"), 0.0)
    up_15 = _safe_float(metrics_15.get("adverse_excursion_pct"), 0.0)
    down_60 = _safe_float(metrics_60.get("favorable_excursion_pct"), 0.0)
    up_60 = _safe_float(metrics_60.get("adverse_excursion_pct"), 0.0)
    minutes_to_down = metrics_60.get("minutes_to_first_favorable")
    minutes_to_up = metrics_60.get("minutes_to_first_adverse")
    down_minutes = None if minutes_to_down is None else _safe_float(minutes_to_down)
    up_minutes = None if minutes_to_up is None else _safe_float(minutes_to_up)

    first_reaction = "flat"
    if down_minutes is not None and (up_minutes is None or down_minutes < up_minutes):
        first_reaction = "down"
    elif up_minutes is not None and (down_minutes is None or up_minutes < down_minutes):
        first_reaction = "up"
    elif down_minutes is not None and up_minutes is not None:
        if down_15 >= max(0.35, up_15 + 0.1):
            first_reaction = "down"
        elif up_15 >= max(0.35, down_15 + 0.1):
            first_reaction = "up"

    timeframe_minutes = max(_timeframe_seconds(timeframe) / 60.0, 1.0 / 60.0)

    def _bars_from_minutes(value: float | None) -> int | None:
        if value is None:
            return None
        return max(1, int(round(float(value) / timeframe_minutes)))

    bars_to_first_down = _bars_from_minutes(down_minutes)
    bars_to_first_up = _bars_from_minutes(up_minutes)
    tp_hit_60 = bool(metrics_60.get("tp_hit"))
    sl_hit_60 = bool(metrics_60.get("sl_hit"))
    verdict = _classify_short_signal_outcome(
        kind=event.kind,
        first_reaction=first_reaction,
        up_15=up_15,
        down_15=down_15,
        up_60=up_60,
        down_60=down_60,
        tp_hit_60=tp_hit_60,
        sl_hit_60=sl_hit_60,
        bars_to_first_down_move=bars_to_first_down,
        bars_to_first_up_move=bars_to_first_up,
    )

    bar_horizons: dict[str, dict[str, Any]] = {}
    for horizon in BAR_HORIZONS:
        horizon_minutes = horizon * timeframe_minutes
        if not float(horizon_minutes).is_integer():
            continue
        payload = raw_horizons.get(str(int(horizon_minutes)))
        if isinstance(payload, dict):
            bar_horizons[str(horizon)] = payload

    signal_bar_ts = _safe_float(observation.get("signal_bar_ts"), 0.0)
    metadata = event.raw.get("metadata", {}) if isinstance(event.raw, dict) else {}
    return {
        "profile": event.profile,
        "kind": event.kind,
        "phase": str(event.raw.get("phase") or observation.get("phase") or ""),
        "symbol": event.symbol,
        "signal_ts": _iso_utc(event.ts),
        "delivery_ts": _iso_utc(event.delivery_ts),
        "signal_bar_ts": _iso_utc(signal_bar_ts) if signal_bar_ts > 0 else "",
        "entry_ref": round(_safe_float(observation.get("entry"), event.entry_price or 0.0), 8),
        "tp": event.tp or _extract_float(observation, "take_profit"),
        "sl": event.sl or _extract_float(observation, "stop_loss"),
        "first_reaction_10m": first_reaction,
        "upside_15m_pct": round(up_15, 3),
        "downside_15m_pct": round(down_15, 3),
        "upside_60m_pct": round(up_60, 3),
        "downside_60m_pct": round(down_60, 3),
        "adverse_excursion_15m_pct": round(up_15, 3),
        "favorable_excursion_15m_pct": round(down_15, 3),
        "adverse_excursion_60m_pct": round(up_60, 3),
        "favorable_excursion_60m_pct": round(down_60, 3),
        "bars_to_first_down_035pct": bars_to_first_down,
        "bars_to_first_up_035pct": bars_to_first_up,
        "minutes_to_first_down_035pct": down_minutes,
        "minutes_to_first_up_035pct": up_minutes,
        "tp_hit_60m": tp_hit_60,
        "sl_hit_60m": sl_hit_60,
        "verdict": verdict,
        "bar_horizons": bar_horizons,
        "context": metadata if isinstance(metadata, dict) else {},
        "status": "local_observation",
        "delivered": bool(observation.get("delivered")),
    }


def _analyze_signal(
    client: MarketDataClient,
    event: SignalEvent,
    *,
    timeframe: str = "1",
    local_observation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(local_observation, dict):
        local_result = _analyze_local_observation(
            event,
            local_observation,
            timeframe=timeframe,
        )
        if local_result is not None:
            return local_result

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
    bars_to_first_down_move = _bars_until_move_pct(window_30, entry=entry, direction="down", threshold_pct=0.35)
    bars_to_first_up_move = _bars_until_move_pct(window_30, entry=entry, direction="up", threshold_pct=0.35)
    horizon_metrics = {
        str(horizon): _window_outcome_metrics(at_or_after.iloc[:horizon], entry=entry)
        for horizon in BAR_HORIZONS
    }

    verdict = _classify_short_signal_outcome(
        kind=event.kind,
        first_reaction=first_reaction,
        up_15=up_15,
        down_15=down_15,
        up_60=up_60,
        down_60=down_60,
        tp_hit_60=tp_hit_60,
        sl_hit_60=sl_hit_60,
        bars_to_first_down_move=bars_to_first_down_move,
        bars_to_first_up_move=bars_to_first_up_move,
    )

    return {
        "profile": event.profile,
        "kind": event.kind,
        "phase": str(event.raw.get("phase") or "") if isinstance(event.raw, dict) else "",
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
        "adverse_excursion_15m_pct": round(up_15, 3),
        "favorable_excursion_15m_pct": round(down_15, 3),
        "adverse_excursion_60m_pct": round(up_60, 3),
        "favorable_excursion_60m_pct": round(down_60, 3),
        "bars_to_first_down_035pct": bars_to_first_down_move,
        "bars_to_first_up_035pct": bars_to_first_up_move,
        "minutes_to_first_down_035pct": _minutes_from_bars(bars_to_first_down_move, timeframe=timeframe),
        "minutes_to_first_up_035pct": _minutes_from_bars(bars_to_first_up_move, timeframe=timeframe),
        "tp_hit_60m": tp_hit_60,
        "sl_hit_60m": sl_hit_60,
        "verdict": verdict,
        "bar_horizons": horizon_metrics,
        "context": _signal_context(ind_df, bar_ts, entry),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze recent signal quality from runtime DB/logs against Bybit OHLCV.")
    parser.add_argument("--main-db", default=str(RUNTIME_DIR / "v2_demo_runtime_main.db"))
    parser.add_argument("--early-db", default=str(RUNTIME_DIR / "v2_demo_runtime_early.db"))
    parser.add_argument("--log", default=str(_latest_live_log()))
    parser.add_argument("--signal-events", default=str(RUNTIME_DIR / "signal_events.jsonl"))
    parser.add_argument("--observations", default=str(LOCAL_OBSERVATIONS_PATH))
    parser.add_argument("--main-limit", type=int, default=12)
    parser.add_argument("--early-limit", type=int, default=12)
    parser.add_argument("--dedupe-window-sec", type=int, default=900)
    parser.add_argument("--timeframe", default="1")
    args = parser.parse_args()

    main_rows = _load_db_rows(Path(args.main_db))
    early_rows = _load_db_rows(Path(args.early_db))
    early_log_rows = _parse_log_for_early_alerts(Path(args.log))
    signal_event_rows = _dedupe_recent_alerts(
        _load_signal_event_log(Path(args.signal_events)),
        cooldown_sec=int(args.dedupe_window_sec),
    )
    signal_event_main = [row for row in signal_event_rows if row.profile == "main"]
    signal_event_early = [row for row in signal_event_rows if row.profile == "early"]
    local_observations = _load_completed_signal_observations(Path(args.observations))

    main_entries = signal_event_main[-int(args.main_limit):] or _select_recent_main_entries(main_rows, int(args.main_limit))
    early_entries = signal_event_early[-int(args.early_limit):] or _enrich_early_alerts_with_db_execution(
        early_log_rows[-int(args.early_limit):],
        early_rows,
    )

    client = MarketDataClient(timeout=12, max_retries=2)
    try:
        main_analyzed = [
            _analyze_signal(
                client,
                row,
                timeframe=str(args.timeframe),
                local_observation=local_observations.get(
                    str(row.raw.get("signal_id") or row.order_link_id or "")
                ),
            )
            for row in main_entries
        ]
        early_analyzed = [
            _analyze_signal(
                client,
                row,
                timeframe=str(args.timeframe),
                local_observation=local_observations.get(
                    str(row.raw.get("signal_id") or row.order_link_id or "")
                ),
            )
            for row in early_entries
        ]
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
