#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ai.retrain_online import OnlineRetrainConfig, OnlineRetrainer
from ai.utils import load_model_bundle, predict_with_bundle
from alerts.chart_generator import build_signal_chart
from alerts.discord_client import DiscordClient
from alerts.telegram_client import TelegramClient
from backtesting.backtest import BacktestConfig, load_ohlcv_csv, run_backtest
from bybit_client import BybitClient
from core.feature_engineering import REQUIRED_MODEL_FEATURES, build_feature_row
from core.indicators import compute_indicators
from core.market_data import MarketDataClient
from core.market_regime import MarketRegime, detect_market_regime
from core.risk_engine import RiskConfig, RiskEngine
from core.settings import load_settings
from core.signal_generator import SignalConfig, SignalContext, SignalGenerator
from core.volume_profile import compute_volume_profile

try:
    from logger import logger
except Exception:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("crypto_bot")


SIGNALS_LOG = Path("logs/signals_log.csv")
TRADES_LOG = Path("logs/trades_log.csv")


def ensure_log_path(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def append_csv(path: Path, row: dict):
    ensure_log_path(path)
    df = pd.DataFrame([row])
    if not path.exists():
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, mode="a", index=False, header=False)


def _to_markdown_message(message_html: str) -> str:
    return message_html.replace("<b>", "**").replace("</b>", "**")


def format_signal_message(signal, regime: MarketRegime, ml_prob: float, horizon: float) -> str:
    return (
        f"<b>{signal.side} {signal.symbol}</b>\n"
        f"Entry: <b>{signal.entry:.6f}</b>\n"
        f"TP: <b>{signal.tp:.6f}</b> | SL: <b>{signal.sl:.6f}</b>\n"
        f"Confidence: <b>{signal.confidence:.2f}</b>\n"
        f"ML Probability: <b>{ml_prob * 100:.1f}%</b>\n"
        f"Horizon: <b>{horizon:.1f}</b> bars\n"
        f"Regime: <b>{regime.value}</b>"
    )


def format_close_message(trade: dict) -> str:
    return (
        f"<b>Trade closed {trade['symbol']}</b>\n"
        f"Side: <b>{trade['side']}</b>\n"
        f"Exit reason: <b>{trade['closed_reason']}</b>\n"
        f"PnL: <b>{trade['pnl']:.4f} USDT</b>\n"
        f"Duration: <b>{trade['duration_sec']:.1f}s</b>"
    )


def send_alerts(telegram: TelegramClient, discord: DiscordClient, message_html: str, chart: bytes | None = None):
    telegram.send_text(message_html)
    if chart:
        telegram.send_photo(caption=message_html, image_bytes=chart)

    msg_md = _to_markdown_message(message_html)
    if chart:
        ok = discord.send_image(msg_md, chart)
        if not ok:
            discord.send_text(msg_md)
    else:
        discord.send_text(msg_md)


def _load_models_for_regimes(model_dir: str, use_regime_models: bool):
    bundles = {"default": load_model_bundle(model_dir=model_dir, regime=None)}
    if not use_regime_models:
        return bundles

    for regime in ("TREND", "RANGE", "PUMP", "PANIC"):
        bundles[regime] = load_model_bundle(model_dir=model_dir, regime=regime)
    return bundles


def _pick_bundle(bundles: dict, regime: MarketRegime):
    bundle = bundles.get(regime.value)
    if bundle and bundle.classifier is not None:
        return bundle
    return bundles["default"]


def _current_open_exposure(risk_engine: RiskEngine) -> float:
    exposure = 0.0
    for pos in risk_engine.open_positions.values():
        exposure += float(pos.get("entry", 0.0)) * float(pos.get("qty", 0.0))
    return exposure


def _timeframe_to_minutes(tf: str) -> int:
    t = str(tf).strip().lower()
    if t.isdigit():
        return max(1, int(t))
    if t == "d":
        return 24 * 60
    if t == "w":
        return 7 * 24 * 60
    return 1


def append_online_training_row(dataset_path: Path, row: dict):
    ensure_log_path(dataset_path)
    df = pd.DataFrame([row])
    if not dataset_path.exists():
        df.to_csv(dataset_path, index=False)
    else:
        df.to_csv(dataset_path, mode="a", index=False, header=False)


def run_backtest_mode(data_path: str, out_path: str):
    try:
        df = load_ohlcv_csv(data_path)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    trades_df, stats = run_backtest(
        df,
        cfg=BacktestConfig(
            initial_equity=float(os.getenv("BT_EQUITY", "1000")),
            risk_per_trade=float(os.getenv("BT_RISK", "0.01")),
            max_hold_bars=int(os.getenv("BT_MAX_HOLD", "120")),
            fee_bps_per_side=float(os.getenv("BT_FEE_BPS", "5")),
            slippage_bps_per_side=float(os.getenv("BT_SLIPPAGE_BPS", "2")),
        ),
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(out, index=False)

    logger.info("Backtest complete -> %s", out)
    for k, v in stats.items():
        logger.info("%s = %s", k, v)


def run_live_or_paper(mode: str):
    settings = load_settings()

    bybit_env = os.getenv("BYBIT_ENV", "mainnet").lower()
    bybit_key = os.getenv("BYBIT_API_KEY", "")
    bybit_secret = os.getenv("BYBIT_API_SECRET", "")

    dry_run = settings.bot.dry_run if mode == "paper" else False
    if os.getenv("BOT_DRY_RUN"):
        dry_run = os.getenv("BOT_DRY_RUN", "true").lower() in ("1", "true", "yes")

    telegram = TelegramClient(
        token=os.getenv("TELEGRAM_TOKEN", ""),
        chat_id=os.getenv("TELEGRAM_CHAT_ID", os.getenv("CHAT_ID", "")),
    )
    discord = DiscordClient(webhook_url=os.getenv("DISCORD_WEBHOOK_URL", ""))

    market = MarketDataClient(
        base_url=settings.market_data.bybit_base_url,
        sentiment_url=settings.market_data.sentiment_api_url,
        timeout=int(settings.market_data.request_timeout_sec),
        max_retries=int(settings.market_data.max_retries),
    )

    bybit = None
    if mode == "live" or (mode == "paper" and not dry_run):
        bybit = BybitClient(
            api_key=bybit_key,
            api_secret=bybit_secret,
            sandbox=(bybit_env != "mainnet"),
            dry_run=dry_run,
        )

    from core.execution import ExecutionEngine

    execution = ExecutionEngine(bybit_client=bybit, dry_run=dry_run)

    signal_cfg = SignalConfig(
        rsi_high=settings.strategy.rsi_high,
        rsi_low=settings.strategy.rsi_low,
        volume_spike_threshold=settings.strategy.volume_spike_threshold,
        weakness_lookback=settings.strategy.weakness_lookback,
        sentiment_bullish_threshold=settings.strategy.sentiment_bullish_threshold,
        sentiment_bearish_threshold=settings.strategy.sentiment_bearish_threshold,
        risk_reward=settings.strategy.risk_reward,
        atr_sl_mult=settings.strategy.atr_sl_mult,
        entry_tolerance_pct=settings.strategy.entry_tolerance_pct,
        vwap_tolerance_pct=settings.strategy.vwap_tolerance_pct,
        funding_tolerance=settings.strategy.funding_tolerance,
        long_short_ratio_tolerance=settings.strategy.long_short_ratio_tolerance,
    )
    signal_gen = SignalGenerator(signal_cfg)

    risk_cfg = RiskConfig(
        account_equity_usdt=settings.risk.account_equity_usdt,
        max_risk_per_trade=settings.risk.max_risk_per_trade,
        max_open_positions=settings.risk.max_open_positions,
        max_total_exposure_pct=settings.risk.max_total_exposure_pct,
        daily_loss_limit_pct=settings.risk.daily_loss_limit_pct,
        max_consecutive_losses=settings.risk.max_consecutive_losses,
        cooldown_minutes=settings.risk.cooldown_minutes,
        min_qty=settings.risk.min_qty,
        max_qty=settings.risk.max_qty,
        slippage_bps=settings.risk.slippage_bps,
    )
    risk_engine = RiskEngine(risk_cfg)

    bundles = _load_models_for_regimes(settings.ml.model_dir, settings.ml.use_regime_models)

    online_dataset = Path(settings.ml.online_dataset_path)
    online_retrainer = OnlineRetrainer(
        OnlineRetrainConfig(
            dataset_path=str(online_dataset),
            model_dir=settings.ml.model_dir,
            retrain_interval_sec=settings.ml.online_retrain_interval_sec,
            min_new_rows=settings.ml.online_retrain_min_rows,
        )
    )

    open_signal_features: dict[str, dict] = {}

    symbol_override = os.getenv("BOT_SYMBOLS", "").strip()
    if symbol_override:
        symbols = [s.strip() for s in symbol_override.split(",") if s.strip()]
    else:
        symbols = market.fetch_symbols(quote=None, categories=("linear", "inverse"))
        if int(settings.bot.symbols_limit) > 0:
            symbols = symbols[: int(settings.bot.symbols_limit)]

    if not symbols:
        logger.warning("No symbols found. Check connectivity or BOT_SYMBOLS override.")
        return

    fast_scan = bool(settings.bot.fast_scan_in_dry_run and mode == "paper" and dry_run)
    progress_every = max(1, int(settings.bot.progress_log_every))
    diagnose_filters = os.getenv("BOT_DIAG_FILTERS", "true").lower() in ("1", "true", "yes")
    ml_threshold = float(settings.ml.min_probability)
    if fast_scan:
        ml_threshold = min(ml_threshold, 0.25)

    logger.info("Starting bot mode=%s dry_run=%s symbols=%d limit=%d universe=all_futures", mode, dry_run, len(symbols), int(settings.bot.symbols_limit))
    logger.info(
        "Scan profile fast_scan=%s progress_every=%d md_timeout=%ss md_retries=%d ml_threshold=%.3f",
        fast_scan,
        progress_every,
        int(settings.market_data.request_timeout_sec),
        int(settings.market_data.max_retries),
        ml_threshold,
    )
    telegram.send_text(f"<b>Bot started</b> mode={mode} dry_run={dry_run} symbols={len(symbols)}")

    tf_min = _timeframe_to_minutes(settings.bot.timeframe)

    try:
        while True:
            cycle_started = time.monotonic()
            cycle_signals = 0
            cycle_errors = 0
            cycle_filter_stats: Counter[str] = Counter()

            global_sentiment = market.fetch_sentiment_index()
            if global_sentiment is None:
                global_sentiment = 50.0
            logger.info("cycle start symbols=%d sentiment=%.1f", len(symbols), float(global_sentiment))

            for idx, symbol in enumerate(symbols, start=1):
                symbol_started = time.monotonic()
                try:
                    snap = market.fetch_snapshot(
                        symbol=symbol,
                        interval=settings.bot.timeframe,
                        limit=settings.bot.candles_limit,
                        include_orderbook=not fast_scan,
                        include_funding_rate=not fast_scan,
                        include_open_interest=not fast_scan,
                        include_long_short_ratio=not fast_scan,
                        include_liquidations=not fast_scan,
                        include_sentiment=False,
                        sentiment_index=float(global_sentiment),
                    )
                    if snap.ohlcv.empty or len(snap.ohlcv) < 80:
                        continue

                    df = compute_indicators(snap.ohlcv)
                    vp = compute_volume_profile(
                        df,
                        window=settings.strategy.volume_profile_window,
                        bins=settings.strategy.volume_profile_bins,
                    )
                    regime = detect_market_regime(df)

                    # Update paper positions on every new bar, even without new signals.
                    closes = execution.update_paper_positions(symbol=symbol, last_price=float(df.iloc[-1]["close"]))
                    for trade in closes:
                        risk_engine.close_position(trade["signal_id"], pnl_usdt=trade["pnl"])
                        append_csv(
                            TRADES_LOG,
                            {
                                "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                                **trade,
                            },
                        )
                        close_msg = format_close_message(trade)
                        send_alerts(telegram, discord, close_msg)

                        # Online supervised row: features at entry + realized outcome.
                        meta = open_signal_features.pop(trade["signal_id"], None)
                        if meta:
                            target_horizon = max(1.0, float(trade.get("duration_sec", 0.0)) / max(1.0, tf_min * 60.0))
                            ds_row = {
                                "timestamp": meta.get("timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")),
                                "market_regime": meta.get("market_regime", regime.value),
                                "target_win": 1 if float(trade.get("pnl", 0.0)) > 0 else 0,
                                "target_horizon": target_horizon,
                                "future_return": float(trade.get("pnl", 0.0)) / max(settings.risk.account_equity_usdt, 1e-9),
                            }
                            for name in REQUIRED_MODEL_FEATURES:
                                ds_row[name] = float(meta.get(name, 0.0))
                            append_online_training_row(online_dataset, ds_row)

                    context = SignalContext(
                        symbol=symbol,
                        df=df,
                        volume_profile=vp,
                        regime=regime,
                        sentiment_index=float(snap.sentiment_index) if snap.sentiment_index is not None else float(global_sentiment),
                        funding_rate=float(snap.funding_rate) if snap.funding_rate is not None else None,
                        long_short_ratio=float(snap.long_short_ratio) if snap.long_short_ratio is not None else None,
                    )
                    signal = signal_gen.generate(context)
                    if signal is None:
                        if diagnose_filters:
                            side, _l1 = signal_gen._layer1_pump_or_panic(df)
                            if side is None:
                                cycle_filter_stats["fail_l1"] += 1
                            else:
                                l2_ok, _l2 = signal_gen._layer2_weakness_confirmation(df, side)
                                if not l2_ok:
                                    cycle_filter_stats["fail_l2"] += 1
                                else:
                                    l3_ok, _l3 = signal_gen._layer3_entry_level(df, side, vp)
                                    if not l3_ok:
                                        cycle_filter_stats["fail_l3"] += 1
                                    else:
                                        l4_ok, _l4 = signal_gen._layer4_fake_filter(
                                            df=df,
                                            side=side,
                                            sentiment_index=context.sentiment_index,
                                            funding_rate=context.funding_rate,
                                            long_short_ratio=context.long_short_ratio,
                                        )
                                        if not l4_ok:
                                            cycle_filter_stats["fail_l4"] += 1
                                        else:
                                            cycle_filter_stats["fail_unknown"] += 1
                        continue

                    feature_row = build_feature_row(
                        symbol=symbol,
                        df=df,
                        volume_profile=vp,
                        regime=regime,
                        extras={
                            "funding_rate": snap.funding_rate,
                            "open_interest": snap.open_interest,
                            "long_short_ratio": snap.long_short_ratio,
                            "sentiment_index": snap.sentiment_index,
                            "liquidation_cluster_high": snap.liquidation_cluster_high,
                            "liquidation_cluster_low": snap.liquidation_cluster_low,
                        },
                    )
                    if feature_row is None:
                        continue

                    bundle = _pick_bundle(bundles, regime)
                    prob, horizon = predict_with_bundle(bundle, feature_row.values)
                    if prob < ml_threshold:
                        cycle_filter_stats["ml_reject"] += 1
                        continue

                    open_exposure = _current_open_exposure(risk_engine)
                    sizing = risk_engine.evaluate_order(
                        signal_id=signal.signal_id,
                        side=signal.side,
                        entry=signal.entry,
                        sl=signal.sl,
                        open_exposure_usdt=open_exposure,
                    )
                    if not sizing.approved:
                        logger.info("Risk rejected %s: %s", signal.symbol, sizing.reason)
                        continue

                    exec_res = execution.execute(signal, qty=sizing.qty, fill_price=sizing.expected_fill)
                    if not exec_res.success:
                        risk_engine.open_positions.pop(signal.signal_id, None)
                        logger.warning("Execution failed for %s: %s", symbol, exec_res.error)
                        continue

                    open_signal_features[signal.signal_id] = {
                        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                        "market_regime": regime.value,
                        **feature_row.values,
                    }

                    chart = None
                    if settings.alerts.send_chart:
                        chart = build_signal_chart(
                            symbol=symbol,
                            df=df,
                            side=signal.side,
                            entry=signal.entry,
                            tp=signal.tp,
                            sl=signal.sl,
                            volume_profile=vp,
                        )

                    msg = format_signal_message(signal=signal, regime=regime, ml_prob=prob, horizon=horizon)
                    send_alerts(telegram, discord, msg, chart=chart)

                    append_csv(
                        SIGNALS_LOG,
                        {
                            "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                            "signal_id": signal.signal_id,
                            "symbol": signal.symbol,
                            "side": signal.side,
                            "entry": signal.entry,
                            "tp": signal.tp,
                            "sl": signal.sl,
                            "partial_tps": "|".join(f"{x:.8f}" for x in signal.partial_tps),
                            "confidence": signal.confidence,
                            "regime": regime.value,
                            "ml_prob": prob,
                            "ml_horizon": horizon,
                            **feature_row.values,
                        },
                    )
                    cycle_signals += 1

                except Exception as symbol_exc:
                    cycle_errors += 1
                    logger.exception("Error processing %s: %s", symbol, symbol_exc)
                finally:
                    if idx % progress_every == 0 or idx == len(symbols):
                        elapsed = time.monotonic() - cycle_started
                        symbol_elapsed = time.monotonic() - symbol_started
                        logger.info(
                            "scan progress %d/%d elapsed=%.1fs symbol=%.2fs signals=%d errors=%d",
                            idx,
                            len(symbols),
                            elapsed,
                            symbol_elapsed,
                            cycle_signals,
                            cycle_errors,
                        )

            if settings.ml.online_retrain_enabled:
                try:
                    if online_retrainer.maybe_retrain(model_type="auto"):
                        bundles = _load_models_for_regimes(settings.ml.model_dir, settings.ml.use_regime_models)
                        send_alerts(telegram, discord, "<b>ML models reloaded after online retrain</b>")
                except Exception:
                    logger.exception("online retrain failed")

            cycle_elapsed = time.monotonic() - cycle_started
            logger.info(
                "loop complete in %.1fs | signals=%d errors=%d | risk=%s",
                cycle_elapsed,
                cycle_signals,
                cycle_errors,
                risk_engine.snapshot(),
            )
            if diagnose_filters:
                logger.info("filter stats: %s", dict(cycle_filter_stats))
            time.sleep(max(2, int(settings.bot.scan_interval_sec)))

    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        market.close()
        if bybit is not None:
            try:
                bybit.close()
            except Exception:
                pass


def parse_args():
    parser = argparse.ArgumentParser(description="Professional layered crypto AI bot")
    parser.add_argument("--mode", default="paper", choices=["paper", "live", "backtest"])
    parser.add_argument("--backtest-data", default="", help="OHLCV CSV path for backtest mode")
    parser.add_argument("--backtest-out", default="logs/backtest_trades.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "backtest":
        if not args.backtest_data:
            raise SystemExit("--backtest-data is required in backtest mode")
        run_backtest_mode(args.backtest_data, args.backtest_out)
    else:
        run_live_or_paper(args.mode)





