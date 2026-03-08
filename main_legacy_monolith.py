#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — стабилизированная полная версия (demo trading)
Функции:
 - загрузка моделей (joblib)
 - построение графиков headless (matplotlib Agg)
 - устойчивые сетевые обращения с автоматическим IP-fallback
 - Telegram уведомления с fallback (sync + async)
 - анализ пар, публикация сигналов, лог-файлы
 - bybit demo trading через BybitClient (если доступен)
 - WebSocket монитор (опционально, если websockets установлен)
"""
from __future__ import annotations

import os

if os.getenv("ALLOW_LEGACY_RUNTIME", "false").strip().lower() not in ("1", "true", "yes"):
    raise RuntimeError("Legacy runtime is quarantined. Use V2 entrypoint app/main.py and trading/* modules.")

import os
import io
import time
import json
import hmac
import hashlib
import socket
import traceback
import argparse
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from collections import deque

# matplotlib headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import asyncio
import numpy as np
import pandas as pd
import joblib
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError, SSLError

# aiohttp client primitives
import aiohttp
from aiohttp import ClientSession, TCPConnector

# optional websockets for WS monitor
try:
    import websockets
except Exception:
    websockets = None

# read .env
from dotenv import load_dotenv
load_dotenv()

# try to import project logger and config if present
try:
    from logger import logger
except Exception:
    # fallback simple logger
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("main")

# optional local modules (may be absent)
try:
    from bybit_client import BybitClient
except Exception:
    BybitClient = None

try:
    from trading_loop import TradingLoop
except Exception:
    TradingLoop = None

try:
    from codex_trainer import train_if_needed
except Exception:
    train_if_needed = None

try:
    from reinforce_trainer import append_trade_to_dataset, maybe_retrain_models
except Exception:
    append_trade_to_dataset = None
    maybe_retrain_models = None

# ================= CONFIG =================
from config import (
    BYBIT_API_KEY,
    BYBIT_API_SECRET,
    BYBIT_DRY_RUN,
    BYBIT_ENV,
    CANDLES,
    CHAT_ID,
    CONCURRENT_TASKS,
    KELTNER_MULTIPLIER,
    LOG_PATH,
    METRICS_PATH,
    MIN_AI_PROB_TO_ORDER,
    MODEL_DIR,
    PRICE_THRESHOLD,
    PROFILE_BINS,
    PROFILE_WINDOW,
    RETRAIN_INTERVAL,
    RISK_ACCOUNT_EQUITY_USDT,
    RISK_CIRCUIT_BREAKER_MINUTES,
    RISK_DAILY_STOP_LOSS_USDT,
    RISK_MAX_CONCURRENT_POSITIONS,
    RISK_MAX_CONSECUTIVE_LOSSES,
    RISK_MAX_QTY,
    RISK_MIN_QTY,
    RISK_PER_TRADE,
    RISK_REWARD,
    RSI_PUMP_THRESHOLD,
    SENTIMENT_API_URL,
    SENTIMENT_EUPHORIA_THRESHOLD,
    SENTIMENT_MAX_AGE_SEC,
    SENTIMENT_REFRESH_SEC,
    TELEGRAM_TOKEN,
    TIMEFRAME,
    TRADE_LOG_PATH,
    VOLUME_THRESHOLD,
)

from engine.data_feed import SentimentConfig, SentimentFeed
from engine.execution import build_signal_id
from engine.metrics import build_metrics_tracker_from_env
from engine.retrain import AsyncRetrainScheduler
from engine.risk import RiskConfig, RiskEngine
from engine.schema import SIGNAL_COLUMNS, TRADE_COLUMNS, append_row_csv, validate_signal_row, validate_trade_row
from engine.state_machine import SignalRecord, SignalState, transition_state

# globals
ai_prob_history = deque(maxlen=200)
active_signals = {}  # symbol -> dict with SignalRecord state
open_signal_ids_by_symbol = {}
open_signal_records_by_id = {}
executor = ThreadPoolExecutor(max_workers=8)
sentiment_cache = {"ts": 0.0, "value": None}
sentiment_feed: SentimentFeed | None = None
os.environ.setdefault("METRICS_PATH", METRICS_PATH)
metrics_tracker = build_metrics_tracker_from_env()
risk_engine = RiskEngine(
    RiskConfig(
        account_equity_usdt=RISK_ACCOUNT_EQUITY_USDT,
        risk_per_trade=RISK_PER_TRADE,
        max_concurrent_positions=RISK_MAX_CONCURRENT_POSITIONS,
        daily_stop_loss_usdt=RISK_DAILY_STOP_LOSS_USDT,
        max_consecutive_losses=RISK_MAX_CONSECUTIVE_LOSSES,
        circuit_breaker_minutes=RISK_CIRCUIT_BREAKER_MINUTES,
        min_qty=RISK_MIN_QTY,
        max_qty=RISK_MAX_QTY,
    )
)

# ================= utilities =================
def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _record_api_error():
    try:
        metrics_tracker.record_api_error()
    except Exception:
        pass

def _maybe_send_degradation_alert():
    try:
        metrics_tracker.flush()
        message = metrics_tracker.should_alert_degradation()
        if message:
            send_telegram(f"<b>Degradation Alert</b>\n{message}")
    except Exception:
        logger.exception("failed to evaluate degradation alert")

def get_decimal_places_for_price(price: float):
    if price is None or price <= 0 or not np.isfinite(price):
        return 6
    exp = int(np.floor(np.log10(abs(price))))
    return 4 if exp >= 0 else min(8, 4 + abs(exp))

def round_price(price, decimals=None):
    if price is None or not np.isfinite(price):
        return price
    if decimals is None:
        decimals = get_decimal_places_for_price(price)
    return float(np.round(price, decimals))

def compute_tp_sl(entry, atr=None, rr=RISK_REWARD, df: pd.DataFrame | None = None, direction: str | None = None):
    """
    Вычисляет TP и SL с учётом ATR и ближайших экстремумов.
    - Для SHORT: SL ставится чуть выше локального high, TP = entry - RR*(entry - SL)
    - Для LONG: SL ставится чуть ниже локального low, TP = entry + RR*(entry - SL)
    """
    try:
        # базовые значения по ATR
        if atr is None or not np.isfinite(atr) or atr <= 0:
            sl_dist = entry * 0.01
        else:
            sl_dist = float(atr) * 1.5

        tp = entry + sl_dist * rr
        sl = entry - sl_dist

        # если есть df и направление — используем локальные экстремумы
        if df is not None and direction:
            window = 10  # можно 8–20
            recent_high = df["high"].iloc[-window:].max()
            recent_low = df["low"].iloc[-window:].min()

            if direction == "LONG":
                sl = min(sl, recent_low * 0.999)  # чуть ниже локального минимума
                tp = entry + (entry - sl) * rr
            else:  # SHORT
                sl = max(sl, recent_high * 1.001)  # чуть выше локального максимума
                tp = entry - (sl - entry) * rr

        dec = get_decimal_places_for_price(entry)
        tp, sl = round_price(tp, dec), round_price(sl, dec)
        return tp, sl, abs(entry - sl)
    except Exception as e:
        logger.warning("compute_tp_sl precise fallback: %s", e)
        return entry * 1.02, entry * 0.98, entry * 0.02

def ensure_tp_sl_order(entry: float, tp: float, sl: float, direction: str):
    dec = get_decimal_places_for_price(entry)
    min_step = 10**(-dec)
    min_step = max(min_step, entry * 0.0001)
    if direction == "LONG":
        if tp <= entry + min_step:
            tp = round_price(entry + max(min_step, entry * 0.0005), dec)
        if sl >= entry - min_step:
            sl = round_price(entry - max(min_step, entry * 0.0005), dec)
    else:
        if tp >= entry - min_step:
            tp = round_price(entry - max(min_step, entry * 0.0005), dec)
        if sl <= entry + min_step:
            sl = round_price(entry + max(min_step, entry * 0.0005), dec)
    return tp, sl

# safe CSV read with simple fixer
def safe_read_csv_or_fix(path: str, expected_cols: int | None = None) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    try:
        # try modern pandas approach
        try:
            df = pd.read_csv(path, on_bad_lines="skip")
            return df
        except TypeError:
            # older pandas
            df = pd.read_csv(path, error_bad_lines=False, warn_bad_lines=True)  # type: ignore
            return df
    except Exception:
        # fallback manual repair
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [ln.rstrip("\n") for ln in f if ln.strip()]
            if not lines:
                return None
            header = lines[0].split(",")
            expected = expected_cols or len(header)
            rows = []
            for ln in lines[1:]:
                parts = ln.split(",")
                if len(parts) < expected:
                    parts += [""] * (expected - len(parts))
                elif len(parts) > expected:
                    parts = parts[:expected]
                rows.append(parts)
            df = pd.DataFrame(rows, columns=header[:expected])
            return df
        except Exception:
            return None

# ================= plotting =================
def plot_candlestick_with_macd(df: pd.DataFrame, symbol: str, direction: str, entry: float, tp: float, sl: float) -> bytes | None:
    try:
        df_plot = df.copy()
        df_plot.columns = [c.lower() for c in df_plot.columns]
        if "time" in df_plot.columns and not isinstance(df_plot.index, pd.DatetimeIndex):
            df_plot["datetime"] = pd.to_datetime(df_plot["time"], unit="ms", errors="coerce")
            df_plot = df_plot.set_index("datetime")
        required = {"open","high","low","close"}
        if not required.issubset(set(df_plot.columns)):
            return None
        cols = ["open","high","low","close"]
        if "volume" in df_plot.columns:
            cols += ["volume"]
        df_plot = df_plot[cols].astype(float).dropna().tail(150)
        if len(df_plot) < 6:
            return None
        df_plot["ema20"] = df_plot["close"].ewm(span=20).mean()
        df_plot["ema50"] = df_plot["close"].ewm(span=50).mean()
        ema12 = df_plot["close"].ewm(span=12).mean()
        ema26 = df_plot["close"].ewm(span=26).mean()
        df_plot["macd"] = ema12 - ema26
        df_plot["signal"] = df_plot["macd"].ewm(span=9).mean()
        df_plot["hist"] = df_plot["macd"] - df_plot["signal"]
        price_range = df_plot["high"].max() - df_plot["low"].min()
        if price_range <= 0 or price_range < (df_plot["close"].mean() * 1e-7):
            df_plot[["open","high","low","close"]] = df_plot[["open","high","low","close"]] * 1e6

        plt.rcParams.update({'font.size': 10})
        fig = plt.figure(figsize=(10,7), facecolor='#0b1220')
        gs = fig.add_gridspec(3,1, height_ratios=[3,0.12,1], hspace=0.08)
        ax_main = fig.add_subplot(gs[0])
        ax_vol  = fig.add_subplot(gs[1], sharex=ax_main)
        ax_macd = fig.add_subplot(gs[2], sharex=ax_main)
        for ax in (ax_main, ax_vol, ax_macd):
            ax.set_facecolor('#0b1220')
            ax.grid(True, linestyle='--', color='#253043', linewidth=0.6, alpha=0.8)

        dates = np.arange(len(df_plot))
        width = 0.6
        for i, (o,h,l,c) in enumerate(zip(df_plot['open'], df_plot['high'], df_plot['low'], df_plot['close'])):
            color = '#26a69a' if c>=o else '#ef5350'
            ax_main.vlines(dates[i], l, h, color='#888888', linewidth=0.8, zorder=1)
            rect = Rectangle((dates[i]-width/2, min(o,c)), width, abs(c-o),
                             facecolor=color, edgecolor='#222222', linewidth=0.3, zorder=2)
            ax_main.add_patch(rect)

        ax_main.plot(dates, df_plot['ema20'], linewidth=1.0)
        ax_main.plot(dates, df_plot['ema50'], linewidth=1.0)
        ax_main.hlines([entry, tp, sl], xmin=dates[0], xmax=dates[-1], linestyles='--', linewidth=1.0)
        ax_main.scatter([dates[-1]], [entry], s=60, zorder=5)
        ax_main.text(dates[-1], entry, ' ENTRY', va='bottom', fontsize=9, fontweight='bold')
        ax_main.set_title(f"{symbol} {direction}", color='white', fontsize=14, pad=10)
        ax_main.set_xlim(dates[0]-1, dates[-1]+1)
        ax_main.set_xticks([])

        if 'volume' in df_plot.columns:
            ax_vol.bar(dates, df_plot['volume'], width=0.6)
            ax_vol.axis('off')

        ax_macd.plot(dates, df_plot['macd'], linewidth=1.0)
        ax_macd.plot(dates, df_plot['signal'], linewidth=1.0)
        ax_macd.bar(dates, df_plot['hist'], width=0.6)
        ax_macd.set_xlim(dates[0]-1, dates[-1]+1)
        ax_main.set_ylabel('Price')
        ax_macd.set_ylabel('MACD')

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception:
        logger.exception("plot_candlestick failed for %s", symbol if 'symbol' in locals() else "")
        return None

# ================= AI models =================
def load_models():
    try:
        model_win = joblib.load(os.path.join(MODEL_DIR, "ai_model_win.pkl"))
        model_horizon = joblib.load(os.path.join(MODEL_DIR, "ai_model_horizon.pkl"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        calibrator = None
        calibrator_path = os.path.join(MODEL_DIR, "ai_calibrator.pkl")
        if os.path.exists(calibrator_path):
            try:
                calibrator = joblib.load(calibrator_path)
                logger.info("AI calibrator loaded")
            except Exception as cal_exc:
                logger.warning("Failed to load AI calibrator: %s", cal_exc)
        logger.info("AI models loaded")
        return model_win, model_horizon, scaler, calibrator
    except Exception as e:
        logger.warning("Failed to load AI models: %s", e)
        return None, None, None, None

model_win, model_horizon, scaler, calibrator = load_models()


def reload_models_in_memory() -> bool:
    global model_win, model_horizon, scaler, calibrator
    model_win, model_horizon, scaler, calibrator = load_models()
    return model_win is not None and model_horizon is not None and scaler is not None


_ai_diagnostics_done = False

# ------------- REPLACE ai_predict with this improved function -------------
def ai_predict(features):
    """
    Predicts (probability, horizon). Uses models if available, otherwise heuristic fallback.
    'features' — list/array-like matching scaler/model feature order:
      [rsi, rsi_5mean, rsi_20mean, price_change, vol_change,
       ema20, ema50, atr, close, ema_diff_norm, atr_norm]
    Returns: (probability: float in [0,1], horizon: float in days/units)
    """
    import numpy as _np
    import pandas as _pd

    # try to make numeric array
    try:
        feats = _np.array([features], dtype=float)
    except Exception as e:
        logger.error("ai_predict: invalid features: %s", e)
        return 0.5, 8.48

    # If ML models available, try using them (safe try/except)
    if model_win is not None and model_horizon is not None and scaler is not None:
        try:
            # Build input for scaler: if scaler expects feature names, provide DataFrame
            if hasattr(scaler, "feature_names_in_"):
                df_features = _pd.DataFrame([feats[0]], columns=scaler.feature_names_in_)
            else:
                df_features = feats
            feats_scaled = scaler.transform(df_features)
            # classification probability
            if hasattr(model_win, "predict_proba"):
                proba = model_win.predict_proba(feats_scaled)[0]
                # choose probability of positive class (assume last column)
                prob = float(proba[-1]) if len(proba) > 1 else float(proba[0])
            else:
                prob = float(model_win.predict(feats_scaled)[0])
            # horizon regressor
            try:
                hpred = model_horizon.predict(feats_scaled)[0]
                horizon = float(hpred) if _np.isfinite(hpred) else None
            except Exception:
                horizon = None
            # clamp
            if prob is None or not _np.isfinite(prob):
                prob = None
            else:
                prob = max(0.0, min(1.0, float(prob)))
            if prob is not None and calibrator is not None and hasattr(calibrator, "transform"):
                try:
                    prob = float(calibrator.transform([prob])[0])
                    prob = max(0.0, min(1.0, prob))
                except Exception as cal_exc:
                    logger.debug("Probability calibration failed: %s", cal_exc)
            if horizon is None or not _np.isfinite(horizon):
                horizon = None
        except Exception as e:
            logger.warning("AI model prediction failed: %s — falling back to heuristic", e)
            prob = None
            horizon = None
    else:
        prob = None
        horizon = None

    # Heuristic fallback (or to complement model output when it is None)
    if prob is None or horizon is None:
        # Extract base indicators from features (safe access)
        try:
            rsi = float(feats[0][0]) if feats.shape[1] > 0 else 50.0
            price_change = float(feats[0][3]) if feats.shape[1] > 3 else 0.0
            vol_change = float(feats[0][4]) if feats.shape[1] > 4 else 1.0
            ema_diff_norm = float(feats[0][9]) if feats.shape[1] > 9 else 0.0
            atr_norm = float(feats[0][10]) if feats.shape[1] > 10 else 0.01
        except Exception:
            rsi = 50.0; price_change = 0.0; vol_change = 1.0; ema_diff_norm = 0.0; atr_norm = 0.01

        # --- probability heuristic ---
        # Normalize RSI to 0..1 (50 -> 0.5 baseline)
        rsi_score = (rsi / 100.0)

        # volume score: compress with log and sigmoid to reduce outliers
        vol_norm = max(0.0, vol_change)
        # treat vol_change relative to 1: vol_change=1 -> neutral, >1 more weight
        vol_score = _np.tanh(min(vol_norm / 5.0, 3.0))  # in (0,1)

        # ema_diff gives directional conviction magnitude
        ema_score = _np.tanh(abs(ema_diff_norm) * 10.0)  # in (0,1)

        # price momentum small positive bias
        momentum = _np.tanh(price_change * 10.0)

        # combine with weights (adjustable)
        prob_heur = 0.12 + 0.50 * rsi_score + 0.18 * vol_score + 0.12 * ema_score + 0.08 * momentum

        # squash to (0.01, 0.99)
        prob_heur = float(max(0.01, min(0.99, prob_heur)))

        if prob is None:
            prob = prob_heur
        else:
            # if model gave prob but it's suspicious, mix with heuristic (robustify)
            prob = float(0.6 * prob + 0.4 * prob_heur)
            prob = max(0.01, min(0.99, prob))

        # --- horizon heuristic ---
        # Base horizon scaled by volatility and ema strength
        # base ~8.5 (former default). Increase horizon when ema_diff and atr_norm larger.
        base = 8.5
        # atr_norm typically small; amplify reasonably
        horizon_heur = base * (1.0 + 3.0 * abs(ema_diff_norm) + 5.0 * min(3.0, atr_norm))
        # add small effect from low prob -> shorter horizon, high prob -> longer horizon
        horizon_heur *= (1.0 + (prob - 0.5) * 0.6)
        horizon_heur = float(max(1.0, min(100.0, round(horizon_heur, 3))))

        if horizon is None:
            horizon = horizon_heur
        else:
            # blend model horizon with heuristic if both exist
            horizon = float(max(1.0, min(100.0, 0.6 * float(horizon) + 0.4 * horizon_heur)))

    # final clamp
    try:
        prob = float(max(0.0, min(1.0, prob)))
        horizon = float(max(1.0, min(100.0, horizon)))
    except Exception:
        prob = 0.5
        horizon = 8.48

    return prob, horizon
# ------------- end ai_predict replacement -------------

# ================= Network helpers & FixedHostResolver =================
_dns_cache = {}  # host -> (ip, ts)

def resolve_host(host: str) -> str:
    """Resolve host to IP using socket.gethostbyname with caching (short TTL)."""
    try:
        now = time.time()
        rec = _dns_cache.get(host)
        if rec and now - rec[1] < 300:  # 5 min cache
            return rec[0]
        ip = socket.gethostbyname(host)
        _dns_cache[host] = (ip, now)
        logger.info("%s -> %s (cached)", host, ip)
        return ip
    except Exception as e:
        logger.warning("DNS error: %s", e)
        return host

class FixedHostResolver(TCPConnector):
    """
    Connector that returns provided IP for a particular hostname.
    Compatible with aiohttp >=3.9/3.10/3.11/3.13.
    """
    def __init__(self, hostname: str | None = None, ip: str | None = None, *args, **kwargs):
        # accept ssl kw for aiohttp connector compatibility
        super().__init__(*args, **kwargs)
        self._hostname = hostname
        self._ip = ip

    # signature includes family default to avoid errors across aiohttp versions
    async def _resolve_host(self, host, port, family=socket.AF_INET, **kwargs):
        # aiohttp may pass family and other kwargs (traces etc.)
        if self._hostname and self._ip and host == self._hostname:
            return [{
                'hostname': host,
                'host': self._ip,
                'port': port,
                'family': family,
                'proto': 0,
                'flags': 0,
            }]
        return await super()._resolve_host(host, port, family=family, **kwargs)

# high-level POST helper with fallback to direct IP (async)
# ================= Improved Telegram and Fallback Networking =================
import ssl
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def send_telegram_sync(text: str, timeout: int = 8):
    """
    Надёжная синхронная отправка текстового сообщения в Telegram.
    Попытки:
      1) обычный requests.post к api.telegram.org
      2) если DNS/TLS/сетевые ошибки — резолвим IP и постим на IP с заголовком Host
      3) если и это не помогает — вторично пробуем на IP с verify=False (крайняя мера)
    """
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logger.debug("Telegram not configured; skipping send_telegram_sync.")
        return

    base_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
    url = f"{base_url}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}

    # 1) Обычная попытка
    try:
        resp = requests.post(url, data=payload, timeout=timeout)
        if resp.status_code == 200:
            logger.debug("send_telegram_sync ok")
            return
        else:
            logger.warning("send_telegram_sync HTTP %s: %s", resp.status_code, resp.text[:300])
    except (RequestException, Timeout, ConnectionError, SSLError) as e:
        logger.warning("Network/DNS/TLS error while send_telegram_sync: %s", e)

    # 2) Попытка через IP (оставляем Host: api.telegram.org)
    try:
        ip = resolve_host("api.telegram.org")
        if ip and ip != "api.telegram.org":
            url_ip = f"https://{ip}/bot{TELEGRAM_TOKEN}/sendMessage"
            headers = {"Host": "api.telegram.org"}
            try:
                resp2 = requests.post(url_ip, data=payload, headers=headers, timeout=timeout)
                if resp2.status_code == 200:
                    logger.info("Fallback POST via IP %s for api.telegram.org succeeded (status=%s)", ip, resp2.status_code)
                    return
                else:
                    logger.warning("Fallback POST via IP %s HTTP %s: %s", ip, resp2.status_code, resp2.text[:300])
            except (RequestException, Timeout, ConnectionError, SSLError) as e2:
                logger.warning("Fallback POST via IP %s for api.telegram.org failed: %s", ip, e2)
                # 3) крайняя мера -> отключаем верификацию сертификата (сертификат не совпадает с IP, поэтому это риск)
                try:
                    resp3 = requests.post(url_ip, data=payload, headers=headers, timeout=timeout, verify=False)
                    if resp3.status_code == 200:
                        logger.warning("Fallback via IP with verify=False succeeded (insecure).")
                        return
                    else:
                        logger.warning("Fallback via IP verify=False HTTP %s: %s", resp3.status_code, resp3.text[:300])
                except Exception as e3:
                    logger.error("Fallback via IP verify=False also failed: %s", e3)
    except Exception as e_all:
        logger.exception("send_telegram_sync: unexpected error in fallback logic: %s", e_all)

    logger.error("send_telegram_sync: all attempts to send Telegram message failed.")

def send_telegram_photo_sync(caption: str, image_bytes: bytes, timeout: int = 20):
    """
    Синхронная отправка изображения в Telegram (requests).
    Аналогичная политика fallback: обычный запрос -> пост на IP с Host -> крайняя мера verify=False.
    """
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logger.debug("Telegram not configured; skipping send_telegram_photo_sync.")
        return

    if not image_bytes:
        logger.warning("send_telegram_photo_sync: image_bytes empty")
        return

    base_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
    url = f"{base_url}/sendPhoto"
    files = {"photo": ("chart.png", io.BytesIO(image_bytes), "image/png")}
    data = {"chat_id": CHAT_ID, "caption": caption, "parse_mode": "HTML"}

    try:
        resp = requests.post(url, data=data, files=files, timeout=timeout)
        if resp.status_code == 200:
            logger.debug("send_telegram_photo_sync ok")
            return
        else:
            logger.warning("send_telegram_photo_sync HTTP %s: %s", resp.status_code, resp.text[:300])
    except (RequestException, Timeout, ConnectionError, SSLError) as e:
        logger.warning("send_telegram_photo_sync network error: %s", e)

    # fallback via IP
    try:
        ip = resolve_host("api.telegram.org")
        if ip and ip != "api.telegram.org":
            url_ip = f"https://{ip}/bot{TELEGRAM_TOKEN}/sendPhoto"
            headers = {"Host": "api.telegram.org"}
            try:
                resp2 = requests.post(url_ip, data=data, files=files, headers=headers, timeout=timeout)
                if resp2.status_code == 200:
                    logger.info("Fallback photo POST via IP %s succeeded", ip)
                    return
                else:
                    logger.warning("Fallback photo POST via IP %s HTTP %s: %s", ip, resp2.status_code, resp2.text[:300])
            except Exception as e2:
                logger.warning("Fallback photo POST via IP failed: %s", e2)
                try:
                    resp3 = requests.post(url_ip, data=data, files=files, headers=headers, timeout=timeout, verify=False)
                    if resp3.status_code == 200:
                        logger.warning("Fallback photo via IP with verify=False succeeded (insecure).")
                        return
                except Exception as e3:
                    logger.error("Fallback photo verify=False also failed: %s", e3)
    except Exception as e_all:
        logger.exception("send_telegram_photo_sync fallback unexpected error: %s", e_all)

    logger.error("send_telegram_photo_sync: all attempts failed.")

async def _post_with_fallback(
    session: ClientSession,
    url: str,
    data: dict | None = None,
    timeout: int = 10,
    host_for_fallback: str | None = None
):
    """
    Усовершенствованный POST с тройным fallback:
      1. aiohttp стандартный POST
      2. aiohttp с IP через FixedHostResolver
      3. requests (sync fallback) при полном отказе DNS или TLS

    Возвращает aiohttp.Response-like объект, либо requests.Response при sync fallback.
    """
    try:
        async with session.post(url, data=data, timeout=timeout) as resp:
            return resp
    except Exception as e1:
        logger.warning("Network/DNS error during async POST: %s", e1)
        if not host_for_fallback:
            raise

        # --- Fallback step 1: пробуем IP через FixedHostResolver ---
        ip = resolve_host(host_for_fallback)
        if not ip or ip == host_for_fallback:
            logger.info("No IP resolved for fallback host %s", host_for_fallback)
        else:
            try:
                ssl_ctx = ssl.create_default_context()
                connector = FixedHostResolver(
                    host_for_fallback,
                    ip,
                    ssl=ssl_ctx,
                    limit=CONCURRENT_TASKS
                )
                async with ClientSession(connector=connector) as sess2:
                    async with sess2.post(url, data=data, timeout=timeout) as resp2:
                        if resp2.status == 200:
                            logger.info("✅ Fallback via IP (%s) for %s succeeded", ip, host_for_fallback)
                            return resp2
                        else:
                            txt = await resp2.text()
                            logger.warning("Fallback via IP %s HTTP %s: %s", ip, resp2.status, txt[:200])
            except Exception as e2:
                logger.warning("Fallback via IP %s failed for %s: %s", ip, host_for_fallback, e2)

        # --- Fallback step 2: requests (sync) ---
        try:
            logger.info("🌐 Final fallback via requests for %s (%s)", host_for_fallback, ip if 'ip' in locals() else None)
            url_replaced = url
            if ip and host_for_fallback in url:
                url_replaced = url.replace(host_for_fallback, ip)
            headers = {"Host": host_for_fallback}

            # ✅ если в fallback используется IP, отключаем проверку SSL
            verify_ssl = not host_for_fallback.replace('.', '').isdigit()

            resp = requests.post(
                url_replaced,
                data=data,
                headers=headers,
                timeout=8,
                verify=verify_ssl
            )

            return resp
        except Exception as e3:
            logger.error("requests fallback failed for %s: %s", host_for_fallback, e3)
            raise e3
# === Telegram helpers with guaranteed fallback (async wrappers + sync fallbacks) ===
async def send_telegram_async(session: ClientSession, text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logger.debug("Telegram not configured; message skipped")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}

    try:
        resp = await _post_with_fallback(session, url, data=payload, timeout=10, host_for_fallback="api.telegram.org")
        # resp may be aiohttp response or requests.Response (sync fallback)
        if hasattr(resp, "status"):
            try:
                txt = await resp.text()
                if resp.status != 200:
                    logger.warning("send_telegram_async HTTP %s: %s", resp.status, txt[:300])
            except Exception:
                pass
        else:
            # requests.Response
            if resp.status_code != 200:
                logger.warning("send_telegram_async (requests fallback) HTTP %s: %s", resp.status_code, resp.text[:300])
    except Exception as e:
        logger.exception("send_telegram_async failed; falling back to sync: %s", e)
        try:
            send_telegram_sync(text)
        except Exception:
            logger.exception("send_telegram_sync also failed")

def send_telegram(text: str):
    """Convenience: try async by creating a short session, else fallback to sync."""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logger.debug("Telegram not configured; skipping send_telegram.")
        return
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # schedule (fire-and-forget) on background task using a fresh session
            async def _fire(text):
                async with ClientSession() as sess:
                    await send_telegram_async(sess, text)
            asyncio.create_task(_fire(text))
            return
    except Exception:
        # if event loop not available or any error — fallback to sync
        pass
    send_telegram_sync(text)

async def send_telegram_photo_async(session: ClientSession, caption: str, image_bytes: bytes):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logger.debug("Telegram not configured; photo skipped")
        return
    if not image_bytes:
        logger.debug("send_telegram_photo_async: empty image")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    form = aiohttp.FormData()
    form.add_field("chat_id", CHAT_ID)
    form.add_field("caption", caption)
    form.add_field("parse_mode", "HTML")
    form.add_field("photo", image_bytes, filename="chart.png", content_type="image/png")
    try:
        async with session.post(url, data=form, timeout=20) as resp:
            txt = await resp.text()
            if resp.status == 200:
                return
            logger.warning("send_telegram_photo_async HTTP %s: %s", resp.status, txt[:300])
    except Exception as e:
        logger.warning("send_telegram_photo_async failed (async attempt): %s", e)
    try:
        send_telegram_photo_sync(caption, image_bytes)
    except Exception:
        logger.exception("send_telegram_photo_sync failed as fallback")

def send_telegram_photo(caption: str, image_bytes: bytes):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logger.debug("Telegram not configured; photo skipped")
        return
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            async def _session_and_send():
                async with ClientSession() as s:
                    await send_telegram_photo_async(s, caption, image_bytes)
            asyncio.create_task(_session_and_send())
            return
    except Exception:
        pass
    send_telegram_photo_sync(caption, image_bytes)


def _send_trade_notification(payload: dict):
    message = payload.get("message", "")
    chart_path = payload.get("chart_path")
    if chart_path:
        try:
            if os.path.exists(chart_path):
                with open(chart_path, "rb") as image_file:
                    send_telegram_photo(message, image_file.read())
                return
        except Exception:
            logger.exception("trade notification photo send failed")
    if message:
        send_telegram(message)


def _order_succeeded(order_result: dict | None) -> bool:
    return isinstance(order_result, dict) and order_result.get("retCode", 0) == 0


def _extract_order_id(order_result: dict | None, symbol: str) -> str:
    result = order_result.get("result", {}) if isinstance(order_result, dict) else {}
    return str(result.get("orderId") or result.get("orderLinkId") or f"{symbol}-{int(time.time() * 1000)}")


def _extract_order_avg_price(order_result: dict | None, fallback: float) -> float:
    result = order_result.get("result", {}) if isinstance(order_result, dict) else {}
    for key in ("avgPrice", "filled_avg_price"):
        value = result.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    return fallback


async def place_managed_order(symbol: str, direction: str, amount: float, entry_price: float, bybit_client=None, trading_loop=None):
    if bybit_client is None:
        return None

    side = "buy" if direction == "LONG" else "sell"
    loop = asyncio.get_running_loop()

    def _place():
        return bybit_client.place_order_market(symbol, side, amount)

    result = await loop.run_in_executor(executor, _place)
    logger.info("Bybit order result for %s: %s", symbol, result)

    if _order_succeeded(result):
        if trading_loop is not None and not getattr(bybit_client, "dry_run", False):
            trading_loop.register_trade(
                _extract_order_id(result, symbol),
                symbol,
                direction,
                _extract_order_avg_price(result, entry_price),
                amount,
            )
        elif getattr(bybit_client, "dry_run", False):
            logger.info("Skipping TradingLoop registration for dry_run order on %s", symbol)
    else:
        logger.warning("Bybit order for %s was not successful: %s", symbol, result)
        _record_api_error()

    return result

# ================= indicators & fetcher =================
def get_sentiment_index(max_age_seconds: int = 300):
    now_ts = time.time()
    cached_value = sentiment_cache.get("value")
    cached_ts = float(sentiment_cache.get("ts") or 0.0)

    if cached_value is not None and (now_ts - cached_ts) <= max_age_seconds:
        return cached_value

    try:
        response = requests.get(SENTIMENT_API_URL, timeout=5)
        response.raise_for_status()
        payload = response.json()
        items = payload.get("data", []) if isinstance(payload, dict) else []
        if items:
            value = float(items[0].get("value"))
            sentiment_cache["value"] = value
            sentiment_cache["ts"] = now_ts
            return value
    except Exception as exc:
        logger.debug("Sentiment index unavailable: %s", exc)
        _record_api_error()

    sentiment_cache["ts"] = now_ts
    return None


async def get_sentiment_index_async(max_age_seconds: int = SENTIMENT_MAX_AGE_SEC):
    global sentiment_feed
    if sentiment_feed is not None:
        try:
            value = await sentiment_feed.get_latest()
            if value is not None:
                sentiment_cache["value"] = value
                sentiment_cache["ts"] = time.time()
                return value
        except Exception:
            logger.debug("sentiment feed read failed", exc_info=True)

    return get_sentiment_index(max_age_seconds=max_age_seconds)

def compute_volume_profile_levels(df: pd.DataFrame, window: int = PROFILE_WINDOW, bins: int = PROFILE_BINS):
    sample = df.tail(max(window, 24)).copy()
    if sample.empty or len(sample) < 10:
        return float("nan"), float("nan"), float("nan")

    typical = ((sample["high"] + sample["low"] + sample["close"]) / 3.0).to_numpy(dtype=float)
    volume = sample["volume"].to_numpy(dtype=float)

    min_price = float(np.nanmin(typical))
    max_price = float(np.nanmax(typical))
    if not np.isfinite(min_price) or not np.isfinite(max_price) or max_price <= min_price:
        return float("nan"), float("nan"), float("nan")

    bins = max(8, int(bins))
    edges = np.linspace(min_price, max_price, bins + 1)
    bucket = np.clip(np.digitize(typical, edges) - 1, 0, bins - 1)

    vol_by_bucket = np.zeros(bins, dtype=float)
    for bucket_idx, vol in zip(bucket, volume):
        if np.isfinite(vol) and vol > 0:
            vol_by_bucket[bucket_idx] += vol

    if float(vol_by_bucket.sum()) <= 0:
        return float("nan"), float("nan"), float("nan")

    centers = (edges[:-1] + edges[1:]) / 2.0
    poc_idx = int(np.argmax(vol_by_bucket))
    poc = float(centers[poc_idx])

    order = np.argsort(vol_by_bucket)[::-1]
    cumulative = 0.0
    target = float(vol_by_bucket.sum()) * 0.7
    selected = []
    for idx in order:
        cumulative += float(vol_by_bucket[idx])
        selected.append(int(idx))
        if cumulative >= target:
            break

    selected_centers = centers[selected]
    vah = float(np.max(selected_centers))
    val = float(np.min(selected_centers))
    return poc, vah, val


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, 1e-5)
    df["rsi"] = 100 - (100 / (1 + gain / loss))

    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std(ddof=0)
    df["bb_upper"] = bb_mid + 2.0 * bb_std
    df["bb_lower"] = bb_mid - 2.0 * bb_std

    df["kc_mid"] = df["ema20"]
    df["kc_upper"] = df["kc_mid"] + KELTNER_MULTIPLIER * df["atr"]
    df["kc_lower"] = df["kc_mid"] - KELTNER_MULTIPLIER * df["atr"]

    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_vol = df["volume"].cumsum().replace(0, np.nan)
    df["vwap"] = (typical_price * df["volume"]).cumsum() / cum_vol

    obv_step = np.sign(df["close"].diff()).fillna(0.0) * df["volume"]
    df["obv"] = obv_step.cumsum()

    signed_volume = np.where(df["close"] >= df["open"], df["volume"], -df["volume"])
    df["cvd"] = pd.Series(signed_volume, index=df.index).cumsum()
    return df

async def fetch_candles(session: ClientSession, symbol: str, max_attempts: int = 3) -> pd.DataFrame | None:
    """
    Robust fetcher for Bybit kline endpoint with DNS fallback.
    symbol example: 'BTC/USDT'
    """
    base = symbol.split("/")[0]
    api_host = "api.bybit.com"
    path = f"/v5/market/kline?category=linear&symbol={base}USDT&interval={TIMEFRAME}&limit={CANDLES}"
    url = f"https://{api_host}{path}"

    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            async with session.get(url, timeout=10) as resp:
                if resp.status != 200:
                    txt = await resp.text()
                    logger.warning("fetch_candles HTTP %s for %s: %s", resp.status, symbol, txt[:200])
                    _record_api_error()
                    return None
                data = await resp.json()
                if "result" not in data or "list" not in data["result"]:
                    _record_api_error()
                    return None
                candles = data["result"]["list"]
                df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "volume", "turnover"])
                df = df[["time", "open", "high", "low", "close", "volume"]].astype(float)
                df = df.sort_values("time", ascending=True).reset_index(drop=True)
                df["datetime"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
                df = df.set_index("datetime")
                df = df.loc[~df.index.isna()]
                return df
        except aiohttp.client_exceptions.ClientConnectorDNSError as e:
            last_exc = e
            logger.warning("DNS error on attempt %d for %s: %s", attempt, symbol, e)
            _record_api_error()
            ip = resolve_host("api.bybit.com")
            if ip and ip != "api.bybit.com":
                try:
                    tmp_conn = FixedHostResolver("api.bybit.com", ip, limit=CONCURRENT_TASKS)
                    async with ClientSession(connector=tmp_conn) as tmp_s:
                        url_direct = f"https://api.bybit.com{path}"
                        async with tmp_s.get(url_direct, timeout=10) as resp2:
                            if resp2.status != 200:
                                txt2 = await resp2.text()
                                logger.warning("fetch_candles (direct) HTTP %s for %s: %s", resp2.status, symbol, txt2[:200])
                                _record_api_error()
                                return None
                            data2 = await resp2.json()
                            if "result" not in data2 or "list" not in data2["result"]:
                                _record_api_error()
                                return None
                            candles2 = data2["result"]["list"]
                            df = pd.DataFrame(candles2, columns=["time", "open", "high", "low", "close", "volume", "turnover"])
                            df = df[["time", "open", "high", "low", "close", "volume"]].astype(float)
                            df = df.sort_values("time", ascending=True).reset_index(drop=True)
                            df["datetime"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
                            df = df.set_index("datetime")
                            df = df.loc[~df.index.isna()]
                            logger.info("Fallback via IP successful for %s", symbol)
                            return df
                except Exception as inner:
                    last_exc = inner
                    logger.warning("fetch_candles direct IP attempt failed for %s: %s", symbol, inner)
                    _record_api_error()
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as net_e:
            last_exc = net_e
            logger.warning("Network error on attempt %d for %s: %s", attempt, symbol, net_e)
            _record_api_error()
        await asyncio.sleep(1.5 ** attempt)
    logger.error("fetch_candles failed for %s after %d attempts: %s", symbol, max_attempts, last_exc)
    _record_api_error()
    return None

# ================= core analyze_pair =================
async def analyze_pair(symbol: str, semaphore: asyncio.Semaphore, last_alerts: dict, session: ClientSession,
                       bybit_client=None, trading_loop=None):
    async with semaphore:
        try:
            df = await fetch_candles(session, symbol)
            if df is None or len(df) < 40:
                return

            df = compute_indicators(df)
            last, prev = df.iloc[-1], df.iloc[-2]

            avg_vol = df["volume"].iloc[:-1].mean() if len(df) > 1 else 0.0
            vol_change = float(last["volume"] / avg_vol) if avg_vol > 0 else 0.0

            now_ts = time.time()
            if symbol in last_alerts and (now_ts - last_alerts[symbol] < 60):
                return

            pump_detected = (
                float(last.get("rsi", 0.0)) >= RSI_PUMP_THRESHOLD
                and vol_change >= VOLUME_THRESHOLD
                and float(last.get("close", 0.0)) > float(last.get("bb_upper", float("inf")))
                and float(last.get("close", 0.0)) > float(last.get("kc_upper", float("inf")))
            )
            if not pump_detected:
                return

            lookback = min(6, len(df) - 1)
            if lookback < 2:
                return

            price_up = float(last["close"]) > float(df["close"].iloc[-1 - lookback])
            obv_down = float(last.get("obv", 0.0)) < float(df["obv"].iloc[-1 - lookback])
            cvd_down = float(last.get("cvd", 0.0)) < float(df["cvd"].iloc[-1 - lookback])
            weakness_confirmed = price_up and (obv_down or cvd_down)
            if not weakness_confirmed:
                return

            poc, vah, val = compute_volume_profile_levels(df)
            if not (np.isfinite(poc) and np.isfinite(vah)):
                return

            pullback_under_vah = float(prev["close"]) >= vah and float(last["close"]) < vah
            if not pullback_under_vah:
                return

            sentiment = await get_sentiment_index_async()
            sentiment_ok = sentiment is not None and sentiment >= SENTIMENT_EUPHORIA_THRESHOLD
            vwap_ok = float(last.get("close", 0.0)) > float(last.get("vwap", 0.0))
            if not (sentiment_ok and vwap_ok):
                return

            direction = "SHORT"
            strategy_type = "pump_short_profile"
            signal_id = build_signal_id(symbol)
            last_alerts[symbol] = now_ts

            entry = float(last["close"])
            atr_val = float(last.get("atr", np.nan))
            if not np.isfinite(atr_val) or atr_val <= 0:
                atr_val = entry * 0.01

            sl = max(vah, entry + atr_val)
            tp = poc
            if tp >= entry:
                tp = entry - atr_val * RISK_REWARD
            if sl <= entry:
                sl = entry + atr_val
            tp, sl = ensure_tp_sl_order(entry, tp, sl, direction)

            price_change = float((last["close"] - prev["close"]) / prev["close"]) if prev["close"] else 0.0
            features = [
                float(last.get("rsi", 0.0)),
                float(df["rsi"].iloc[-5:].mean()) if len(df) >= 5 else float(last.get("rsi", 0.0)),
                float(df["rsi"].iloc[-20:].mean()) if len(df) >= 20 else float(last.get("rsi", 0.0)),
                price_change,
                vol_change,
                float(last.get("ema20", 0.0)),
                float(last.get("ema50", 0.0)),
                float(last.get("atr", 0.0)),
                float(last.get("close", 0.0)),
                float((last["ema20"] - last["ema50"]) / max(float(last.get("close", 1.0)), 1e-9)),
                float(last["atr"] / max(float(last.get("close", 1.0)), 1e-9)),
            ]

            ai_prob, ai_horizon = ai_predict(features)
            if ai_prob is not None and np.isfinite(ai_prob):
                ai_prob_history.append(ai_prob)

            base = symbol.split("/")[0]
            sentiment_text = f"{sentiment:.0f}" if sentiment is not None else "n/a"
            msg = (
                f"<b>{direction}: {symbol}</b>\n"
                f"Entry: <b>{entry:.{get_decimal_places_for_price(entry)}f}</b>\n"
                f"TP: <b>{tp:.{get_decimal_places_for_price(tp)}f}</b> | "
                f"SL: <b>{sl:.{get_decimal_places_for_price(sl)}f}</b>\n"
                f"RSI: {last['rsi']:.1f} | Vol: {vol_change:.2f}x\n"
                f"OBV div: {obv_down} | CVD div: {cvd_down}\n"
                f"VAH: {vah:.6f} | POC: {poc:.6f} | Sentiment: {sentiment_text}\n"
                f"<a href='https://www.bybit.com/trade/usdt/{base}'>Chart</a>"
            )
            if ai_prob is not None:
                msg += f"\n\n<b>AI Win Probability:</b> {ai_prob * 100:.1f}%"
            if ai_horizon is not None:
                msg += f"\n<b>Expected Horizon:</b> {round(ai_horizon, 2)}"

            chart_bytes = plot_candlestick_with_macd(df, symbol, direction, entry, tp, sl)
            try:
                if chart_bytes:
                    send_telegram_photo(msg, chart_bytes)
                else:
                    send_telegram(msg)
            except Exception:
                logger.exception("Telegram send failed for %s", symbol)

            signal_row = validate_signal_row(
                {
                    "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": symbol,
                    "direction": direction,
                    "entry": entry,
                    "tp": tp,
                    "sl": sl,
                    "rsi": last.get("rsi", float("nan")),
                    "vol_change": vol_change,
                    "ai_prob": ai_prob,
                    "ai_horizon": ai_horizon,
                    "strategy": strategy_type,
                    "vwap": float(last.get("vwap", float("nan"))),
                    "poc": float(poc),
                    "vah": float(vah),
                    "val": float(val) if np.isfinite(val) else float("nan"),
                    "sentiment": sentiment if sentiment is not None else float("nan"),
                    "obv": float(last.get("obv", float("nan"))),
                    "cvd": float(last.get("cvd", float("nan"))),
                    "state": SignalState.DETECTED.value,
                }
            )
            append_row_csv(LOG_PATH, signal_row, SIGNAL_COLUMNS)
            metrics_tracker.record_signal_detected()

            record = SignalRecord(
                signal_id=signal_id,
                symbol=symbol,
                direction=direction,
                entry=entry,
                tp=tp,
                sl=sl,
                strategy=strategy_type,
                ai_prob=ai_prob if ai_prob is None or np.isfinite(ai_prob) else None,
            )

            active_signals[symbol] = {
                "signal_id": signal_id,
                "record": record,
                "direction": direction,
                "entry": entry,
                "tp": tp,
                "sl": sl,
                "timestamp": time.time(),
                "confirmed": False,
                "ai_prob": ai_prob,
                "strategy": strategy_type,
                "state": record.state.value,
            }

        except Exception:
            logger.exception("analyze_pair error for %s", symbol)
            return

# ================= monitor_entries (background) =================
async def monitor_entries(session: ClientSession, bybit_client=None, trading_loop=None):
    logger.info("monitor_entries task started")
    while True:
        try:
            if not active_signals:
                await asyncio.sleep(10)
                continue

            for symbol, sig in list(active_signals.items()):
                try:
                    record = sig.get("record")
                    if not isinstance(record, SignalRecord):
                        active_signals.pop(symbol, None)
                        continue

                    if time.time() - float(sig.get("timestamp", time.time())) > 1800:
                        transition_state(record, SignalState.EXPIRED)
                        sig["state"] = record.state.value
                        active_signals.pop(symbol, None)
                        continue

                    df = await fetch_candles(session, symbol)
                    if df is None or len(df) < 2:
                        continue

                    last_price = float(df["close"].iloc[-1])
                    direction = sig["direction"]
                    entry = float(sig["entry"])
                    confirmed = False
                    if direction == "LONG" and last_price >= entry:
                        confirmed = True
                    elif direction == "SHORT" and last_price <= entry:
                        confirmed = True

                    if not confirmed or sig.get("confirmed", False):
                        continue

                    sig["confirmed"] = True
                    transition_state(record, SignalState.CONFIRMED)
                    sig["state"] = record.state.value

                    ai_prob = sig.get("ai_prob", None)
                    prob_text = f"{ai_prob * 100:.1f}%" if ai_prob is not None else "n/a"
                    send_telegram(
                        f"<b>Entry confirmed: {symbol}</b>\n"
                        f"Direction: <b>{direction}</b>\n"
                        f"Reached price: <b>{last_price:.6f}</b>\n"
                        f"AI probability: <b>{prob_text}</b>\n"
                        f"Strategy: {sig.get('strategy', '')}"
                    )

                    if ai_prob is None or ai_prob < MIN_AI_PROB_TO_ORDER:
                        transition_state(record, SignalState.REJECTED)
                        sig["state"] = record.state.value
                        logger.info("Skipping order for %s due to AI probability %.3f", symbol, float(ai_prob or 0.0))
                        active_signals.pop(symbol, None)
                        continue

                    can_open, reason = risk_engine.can_open_trade()
                    if not can_open:
                        transition_state(record, SignalState.REJECTED)
                        sig["state"] = record.state.value
                        send_telegram(f"<b>Order rejected by risk engine</b>\n{symbol}: {reason}")
                        active_signals.pop(symbol, None)
                        continue

                    qty = risk_engine.recommend_qty(entry=entry, sl=float(sig["sl"]))
                    if qty <= 0:
                        transition_state(record, SignalState.REJECTED)
                        sig["state"] = record.state.value
                        logger.warning("Risk engine returned non-positive qty for %s", symbol)
                        active_signals.pop(symbol, None)
                        continue

                    if bybit_client is None:
                        logger.info("No bybit_client configured, skipping order for %s", symbol)
                        active_signals.pop(symbol, None)
                        continue

                    try:
                        result = await place_managed_order(
                            symbol,
                            direction,
                            amount=qty,
                            entry_price=last_price,
                            bybit_client=bybit_client,
                            trading_loop=trading_loop,
                        )
                    except Exception:
                        logger.exception("Failed to place order on confirmation")
                        _record_api_error()
                        transition_state(record, SignalState.REJECTED)
                        sig["state"] = record.state.value
                        active_signals.pop(symbol, None)
                        continue

                    if _order_succeeded(result):
                        transition_state(record, SignalState.ORDERED)
                        sig["state"] = record.state.value
                        metrics_tracker.record_signal_ordered()
                        if not getattr(bybit_client, "dry_run", False):
                            risk_engine.on_trade_open(
                                signal_id=sig["signal_id"],
                                symbol=symbol,
                                qty=qty,
                                entry=last_price,
                                sl=float(sig["sl"]),
                            )
                            open_signal_ids_by_symbol[symbol] = sig["signal_id"]
                            open_signal_records_by_id[sig["signal_id"]] = record
                    else:
                        _record_api_error()
                        transition_state(record, SignalState.REJECTED)
                        sig["state"] = record.state.value

                    active_signals.pop(symbol, None)
                    _maybe_send_degradation_alert()
                except Exception:
                    logger.exception("monitor_entries loop error for %s", symbol)
                    continue
            await asyncio.sleep(10)
        except Exception:
            logger.exception("monitor_entries outer loop")
            await asyncio.sleep(5)

# ================= show_positions_loop =================
async def show_positions_loop(bybit_client):
    logger.info("show_positions_loop started")
    while True:
        try:
            if bybit_client is None:
                await asyncio.sleep(5)
                continue
            loop = asyncio.get_running_loop()
            def get_pos():
                try:
                    return bybit_client.get_open_positions()
                except Exception as e:
                    return {"error": str(e)}
            res = await loop.run_in_executor(executor, get_pos)
            if not res:
                logger.info("No open positions")
            elif isinstance(res, dict) and res.get("error"):
                logger.warning("get_open_positions error: %s", res.get("error"))
            else:
                logger.info("Active positions:")
                try:
                    for p in res:
                        symbol = p.get("symbol")
                        side = p.get("side")
                        entry = float(p.get("entry_price", 0))
                        pnl = float(p.get("unrealised_pnl", 0))
                        logger.info("  %s | %s | entry=%s | PnL=%s", symbol, side, entry, pnl)
                except Exception:
                    logger.exception("show_positions_loop unexpected format")
            await asyncio.sleep(15)
        except Exception:
            logger.exception("show_positions_loop error")
            await asyncio.sleep(5)

# ================= WebSocket monitor class (optional) =================
class BybitWSMonitor:
    def __init__(self, api_key, api_secret, send_func=None, sandbox: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret.encode() if isinstance(api_secret, str) else api_secret
        self.sandbox = sandbox
        self.send_func = send_func
        self.ws = None
        self.connected = False
        if self.sandbox:
            self.endpoint = "wss://stream-testnet.bybit.com/v5/private"
        else:
            self.endpoint = "wss://stream.bybit.com/v5/private"

    async def connect(self):
        if websockets is None:
            logger.warning("websockets library not available — skipping WS monitor.")
            return
        try:
            logger.info("Connecting to Bybit WS: %s", self.endpoint)
            self.ws = await websockets.connect(self.endpoint)
            await self.authenticate()
            self.connected = True
            asyncio.create_task(self.listen())
            logger.info("Bybit WS connected")
        except Exception:
            logger.exception("Bybit WS connect failed")

    async def authenticate(self):
        ts = int(time.time() * 1000)
        payload = f"GET/realtime{ts}"
        sign = hmac.new(self.api_secret, payload.encode(), hashlib.sha256).hexdigest()
        auth_msg = {"op": "auth", "args": [self.api_key, ts, sign]}
        await self.ws.send(json.dumps(auth_msg))
        await asyncio.sleep(0.3)
        await self.ws.send(json.dumps({"op": "subscribe", "args": ["position"]}))

    async def listen(self):
        try:
            async for msg in self.ws:
                try:
                    data = json.loads(msg)
                    if data.get("topic") == "position":
                        self.handle_position_update(data)
                except Exception:
                    logger.exception("WS message handling error")
        except Exception:
            logger.exception("Bybit WS listen stopped")
            self.connected = False

    def handle_position_update(self, data):
        try:
            rows = data.get("data", [])
            for pos in rows:
                symbol = pos.get("symbol")
                side = pos.get("side")
                size = float(pos.get("size", 0))
                entry = float(pos.get("avgPrice", 0))
                pnl = float(pos.get("unrealisedPnl", 0))
                ts = datetime.utcnow().strftime("%H:%M:%S")
                if size > 0:
                    msg = f"🟢 [{ts}] {symbol} {side} | size={size} | entry={entry:.6f} | PnL={pnl:.4f}"
                    logger.info(msg)
                    if self.send_func:
                        try: self.send_func(msg)
                        except Exception: pass
                else:
                    msg = f"⚪ [{ts}] {symbol} position closed"
                    logger.info(msg)
                    if self.send_func:
                        try: self.send_func(msg)
                        except Exception: pass
        except Exception:
            logger.exception("handle_position_update failed")

# ================= fetch pairs (instruments list) =================
async def fetch_pairs(session: ClientSession, url: str, attempts: int = 3) -> list[str]:
    logger.info("fetch pairs: %s", url)
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            async with session.get(url, timeout=10) as resp:
                if resp.status != 200:
                    txt = await resp.text()
                    logger.warning("fetch_pairs HTTP %s: %s", resp.status, txt[:300])
                    _record_api_error()
                    return []
                data = await resp.json()
                instruments = data.get("result", {}).get("list", [])
                symbols = []
                for item in instruments:
                    if (
                        item.get("contractType") in ("LinearPerpetual", "LinearPerpetualContract", "LinearPerpetual")
                        and item.get("quoteCoin") == "USDT"
                        and item.get("status") == "Trading"
                    ):
                        base = item.get("baseCoin")
                        if base:
                            symbols.append(f"{base}/USDT")
                logger.info("pairs found: %s", len(symbols))
                return sorted(set(symbols))
        except aiohttp.client_exceptions.ClientConnectorDNSError as e:
            last_exc = e
            logger.warning("fetch_pairs attempt %d failed: %s", attempt, e)
            _record_api_error()
            ip = resolve_host("api.bybit.com")
            if ip and ip != "api.bybit.com":
                try:
                    tmp_conn = FixedHostResolver("api.bybit.com", ip, limit=CONCURRENT_TASKS)
                    async with ClientSession(connector=tmp_conn) as tmp_s:
                        async with tmp_s.get(url, timeout=10) as resp2:
                            if resp2.status != 200:
                                txt = await resp2.text()
                                logger.warning("fetch_pairs direct HTTP %s: %s", resp2.status, txt[:300])
                                _record_api_error()
                                return []
                            data2 = await resp2.json()
                            instruments = data2.get("result", {}).get("list", [])
                            symbols = []
                            for item in instruments:
                                if (
                                    item.get("contractType") in ("LinearPerpetual", "LinearPerpetualContract")
                                    and item.get("quoteCoin") == "USDT"
                                    and item.get("status") == "Trading"
                                ):
                                    base = item.get("baseCoin")
                                    if base:
                                        symbols.append(f"{base}/USDT")
                            logger.info("fetch_pairs direct success: %s", len(symbols))
                            return sorted(set(symbols))
                except Exception as inner:
                    last_exc = inner
                    logger.warning("fetch_pairs direct IP attempt failed: %s", inner)
                    _record_api_error()
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as net_e:
            last_exc = net_e
            logger.warning("fetch_pairs attempt %d failed: %s", attempt, net_e)
            _record_api_error()
        await asyncio.sleep(1.5 ** attempt)
    logger.error("fetch_pairs failed after retries: %s", last_exc)
    _record_api_error()
    return []

# ================= main loop =================
async def main(retrain_interval: int = RETRAIN_INTERVAL, enable_demo_trading: bool = True):
    import asyncio
    global sentiment_feed

    logger.info("Bot starting with AI + demo trading option.")
    try:
        send_telegram("<b>Bot with AI started (demo mode)</b> - scanning market...")
    except Exception:
        pass

    bybit_client = None
    trading_loop = None
    if enable_demo_trading and BybitClient is not None:
        try:
            if not BYBIT_DRY_RUN and (not BYBIT_API_KEY or not BYBIT_API_SECRET):
                logger.warning("Bybit live mode requested without API credentials; trading client disabled.")
            else:
                bybit_client = BybitClient(
                    api_key=BYBIT_API_KEY,
                    api_secret=BYBIT_API_SECRET,
                    sandbox=(BYBIT_ENV != "mainnet"),
                    dry_run=BYBIT_DRY_RUN,
                )
            if bybit_client is not None and TradingLoop is not None and not BYBIT_DRY_RUN:
                trading_loop = TradingLoop(bybit_client, notifier=_send_trade_notification)
                try:
                    asyncio.create_task(trading_loop.monitor_loop(on_trade_closed_callback=_on_trade_closed))
                except Exception:
                    logger.debug("TradingLoop monitor not started or not implemented")
            elif bybit_client is not None and TradingLoop is not None:
                logger.info("TradingLoop disabled in dry_run mode because Bybit returns no live positions.")
            if bybit_client is not None:
                logger.info("Bybit demo trading enabled (dry_run=%s).", BYBIT_DRY_RUN)
        except Exception:
            logger.exception("Bybit client init failed")
            bybit_client = None
            trading_loop = None

    resolved_ip = resolve_host("api.bybit.com")
    connector = FixedHostResolver("api.bybit.com", resolved_ip, limit=CONCURRENT_TASKS)

    last_alerts = {}
    sem = asyncio.Semaphore(CONCURRENT_TASKS)
    url_pairs = "https://api.bybit.com/v5/market/instruments-info?category=linear&limit=1000&status=Trading"

    retrain_scheduler = None
    if train_if_needed:
        retrain_scheduler = AsyncRetrainScheduler(
            train_callable=train_if_needed,
            reload_models_callable=reload_models_in_memory,
            retrain_interval=retrain_interval,
            model_dir=MODEL_DIR,
        )

    async with ClientSession(connector=connector) as session:
        sentiment_stop_event = asyncio.Event()
        sentiment_feed = SentimentFeed(
            SentimentConfig(
                url=SENTIMENT_API_URL,
                refresh_sec=SENTIMENT_REFRESH_SEC,
                max_age_sec=SENTIMENT_MAX_AGE_SEC,
            )
        )
        sentiment_task = asyncio.create_task(sentiment_feed.run(session, sentiment_stop_event))

        task_monitor = asyncio.create_task(
            monitor_entries(session, bybit_client=bybit_client, trading_loop=trading_loop)
        )
        task_positions = None
        if bybit_client is not None and not BYBIT_DRY_RUN:
            task_positions = asyncio.create_task(show_positions_loop(bybit_client))

        if BYBIT_API_KEY and BYBIT_API_SECRET and websockets is not None and not BYBIT_DRY_RUN:
            ws = BybitWSMonitor(BYBIT_API_KEY, BYBIT_API_SECRET, send_func=send_telegram, sandbox=(BYBIT_ENV != "mainnet"))
            asyncio.create_task(ws.connect())

        try:
            while True:
                if retrain_scheduler is not None:
                    await retrain_scheduler.tick()

                symbols = await fetch_pairs(session, url_pairs)
                now_str = datetime.now().strftime("%H:%M:%S")
                logger.info("%s | Working on %d pairs...", now_str, len(symbols))
                if not symbols:
                    _maybe_send_degradation_alert()
                    await asyncio.sleep(5)
                    continue

                tasks = [
                    analyze_pair(s, sem, last_alerts, session, bybit_client=bybit_client, trading_loop=trading_loop)
                    for s in symbols
                ]
                await asyncio.gather(*tasks, return_exceptions=True)

                if len(ai_prob_history) >= 10:
                    avg_prob = np.mean(ai_prob_history)
                    med_prob = np.median(ai_prob_history)
                    min_prob = np.min(ai_prob_history)
                    max_prob = np.max(ai_prob_history)
                    logger.info(
                        "AI stats (last %d): avg=%.1f%% med=%.1f%% min=%.1f%% max=%.1f%%",
                        len(ai_prob_history),
                        avg_prob * 100,
                        med_prob * 100,
                        min_prob * 100,
                        max_prob * 100,
                    )

                _maybe_send_degradation_alert()
                await asyncio.sleep(2)
        except asyncio.CancelledError:
            logger.info("main loop cancelled")
        except Exception:
            logger.exception("main loop crashed")
        finally:
            if retrain_scheduler is not None:
                retrain_scheduler.shutdown()

            sentiment_stop_event.set()
            try:
                await asyncio.wait_for(sentiment_task, timeout=2)
            except Exception:
                sentiment_task.cancel()
            sentiment_feed = None

            try:
                task_monitor.cancel()
            except Exception:
                pass
            if task_positions:
                try:
                    task_positions.cancel()
                except Exception:
                    pass
            await asyncio.sleep(0.5)

# ---- callback for closed trades ----------------------------------------------------------------
def _on_trade_closed(trade_res: dict):
    try:
        logger.info("Trade closed: %s", trade_res)

        trade_row = validate_trade_row(
            {
                "time": trade_res.get("closed_at") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": trade_res.get("symbol"),
                "profit": trade_res.get("profit"),
                "duration": trade_res.get("duration"),
                "side": trade_res.get("side"),
            }
        )
        append_row_csv(TRADE_LOG_PATH, trade_row, TRADE_COLUMNS)

        if append_trade_to_dataset and os.path.normpath(TRADE_LOG_PATH) != os.path.normpath("logs/trades_log.csv"):
            try:
                append_trade_to_dataset(trade_res)
            except Exception:
                logger.exception("append_trade_to_dataset failed")

        symbol = trade_row.get("symbol", "?")
        pnl = float(trade_row.get("profit", 0.0))
        signal_id = open_signal_ids_by_symbol.pop(symbol, None)
        if signal_id:
            risk_engine.close_position(signal_id)
            record = open_signal_records_by_id.pop(signal_id, None)
            if isinstance(record, SignalRecord):
                transition_state(record, SignalState.CLOSED)
        risk_engine.on_trade_closed(pnl)
        metrics_tracker.record_trade_closed(pnl)
        _maybe_send_degradation_alert()

        logger.info("Risk snapshot after close: %s", risk_engine.snapshot())
        logger.info("Trade closure callback processed for %s", symbol)
        if maybe_retrain_models and train_if_needed:
            try:
                maybe_retrain_models(lambda force=True: train_if_needed(force=force), min_trades=20)
            except Exception:
                logger.exception("maybe_retrain_models failed")
    except Exception:
        logger.exception("_on_trade_closed failed")

# ---------------- entrypoint ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrain-interval', type=int, default=RETRAIN_INTERVAL, help='retrain interval seconds')
    parser.add_argument('--no-demo', action='store_true', help='disable demo trading')
    args = parser.parse_args()
    try:
        # main runs demo trading by default (dry_run=True); pass --no-demo to disable demo features
        asyncio.run(main(retrain_interval=args.retrain_interval, enable_demo_trading=not args.no_demo))
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception:
        logger.exception("Unhandled exception in __main__")

