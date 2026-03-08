from __future__ import annotations

import argparse
import time
from typing import TYPE_CHECKING

from app.bootstrap import ConfigError, RuntimeConfig, load_runtime_config
from trading.alerts.discord import DiscordAlerter
from trading.alerts.telegram import TelegramAlerter
from trading.execution.engine import ExecutionEngine
from trading.exchange.bybit_adapter import BybitAdapter
from trading.metrics.counters import MetricsCounter
from trading.metrics.logging import setup_logging
from trading.risk.engine import RiskEngine
from trading.signals.base import HoldStrategy
from trading.signals.layered_strategy import LayeredPumpStrategy
from trading.signals.signal_types import IntentAction
from trading.signals.strategy_interface import StrategyContext
from trading.state.machine import StateMachine
from trading.state.models import TradeState
from trading.state.persistence import RuntimeStore

if TYPE_CHECKING:
    from trading.features.pipeline import FeaturePipeline
    from trading.market_data.feed import MarketDataFeed
    from trading.market_data.reconciliation import ExchangeReconciler
    from trading.market_data.ws_reconciliation import ExchangeSyncService



def _build_strategy(name: str):
    if name == "hold":
        return HoldStrategy()
    return LayeredPumpStrategy()



def _build_alerters(cfg: RuntimeConfig):
    out = []
    if cfg.alerts.telegram_token and cfg.alerts.telegram_chat_id:
        out.append(TelegramAlerter(token=cfg.alerts.telegram_token, chat_id=cfg.alerts.telegram_chat_id))
    if cfg.alerts.discord_webhook_url:
        out.append(DiscordAlerter(webhook_url=cfg.alerts.discord_webhook_url))
    return out



def _send_alerts(alerters, text: str):
    for alerter in alerters:
        try:
            alerter.send(text)
        except Exception:
            continue



def _startup_reconcile(
    *,
    symbols: list[str],
    sync: ExchangeSyncService,
    state_machine: StateMachine,
    execution: ExecutionEngine,
):
    summary: dict[str, str] = {}
    for symbol in symbols:
        snapshot = sync.snapshot(symbol)
        rec_state = state_machine.reconcile(symbol, snapshot.positions, snapshot.open_orders)
        execution.recover_from_restart(symbol, snapshot)

        # Recovery can change exchange state (cancel/close/attach stop), refresh from exchange truth.
        snapshot = sync.reconciler.snapshot(symbol)
        rec_state = state_machine.reconcile(symbol, snapshot.positions, snapshot.open_orders)
        issues = execution.detect_external_intervention(symbol, snapshot)
        if issues:
            summary[symbol] = f"{state_machine.get(symbol).state.value}|issues={','.join(issues)}"
        else:
            summary[symbol] = rec_state.state.value
    return summary



def run_cycle(
    *,
    symbols: list[str],
    adapter: BybitAdapter,
    feed: MarketDataFeed,
    sync: ExchangeSyncService,
    pipeline: FeaturePipeline,
    strategy,
    risk: RiskEngine,
    execution: ExecutionEngine,
    logger,
    counters: MetricsCounter,
    timeframe: str,
    candles_limit: int,
    alerters,
    state_alert_cache: dict[str, str],
):
    sync.pull_adapter_events(adapter)

    for symbol in symbols:
        try:
            snapshot = sync.snapshot(symbol)
            rec_state = execution.state_machine.reconcile(symbol, snapshot.positions, snapshot.open_orders)
            execution.recover_from_restart(symbol, snapshot)

            # Recovery is allowed to mutate exchange state, so refresh before new decision.
            snapshot = sync.reconciler.snapshot(symbol)
            rec_state = execution.state_machine.reconcile(symbol, snapshot.positions, snapshot.open_orders)

            intervention = execution.detect_external_intervention(symbol, snapshot)
            if intervention:
                counters.inc("interventions")
                issues = ",".join(intervention)
                logger.error(
                    "symbol=%s state=%s intervention=%s",
                    symbol,
                    execution.state_machine.get(symbol).state.value,
                    issues,
                    extra={"event": "intervention"},
                )
                _send_alerts(
                    alerters,
                    f"[CRITICAL] intervention symbol={symbol} issues={issues} state={execution.state_machine.get(symbol).state.value}",
                )

            current_state = execution.state_machine.get(symbol).state
            if current_state in (TradeState.HALTED, TradeState.RECOVERING, TradeState.ERROR):
                counters.inc("state_blocked")
                state_key = current_state.value
                if state_alert_cache.get(symbol) != state_key:
                    reason = execution.state_machine.get(symbol).reason
                    logger.error(
                        "symbol=%s blocked_state=%s reason=%s",
                        symbol,
                        state_key,
                        reason,
                        extra={"event": "state_blocked"},
                    )
                    _send_alerts(
                        alerters,
                        f"[CRITICAL] state_blocked symbol={symbol} state={state_key} reason={reason}",
                    )
                    state_alert_cache[symbol] = state_key
                continue

            state_alert_cache.pop(symbol, None)
            rec_state = execution.state_machine.get(symbol)

            frame = feed.fetch_frame(symbol=symbol, timeframe=timeframe, candles=candles_limit)
            if frame.ohlcv.empty:
                counters.inc("empty_ohlcv")
                continue

            as_of = frame.ohlcv.index[-1]
            extras = {
                "sentiment_index": 50.0,
                "sentiment_source": "fallback_neutral_50",
                "funding_rate": None,
                "long_short_ratio": None,
            }
            features = pipeline.build(symbol=symbol, ohlcv=frame.ohlcv, as_of=as_of, extras=extras)

            mark_price = frame.mark_price if frame.mark_price > 0 else float(features.enriched.iloc[-1]["close"])
            intent = strategy.generate(
                StrategyContext(
                    symbol=symbol,
                    market_ohlcv=features.enriched,
                    mark_price=mark_price,
                    exchange=snapshot,
                    synced_state=rec_state.state,
                    sentiment_index=extras.get("sentiment_index"),
                    sentiment_source=extras.get("sentiment_source"),
                    funding_rate=extras.get("funding_rate"),
                    long_short_ratio=extras.get("long_short_ratio"),
                )
            )

            try:
                rules = adapter.get_instrument_rules(symbol)
            except Exception as exc:
                counters.inc("metadata_errors")
                logger.error(
                    "symbol=%s metadata_error=%s", symbol, exc, extra={"event": "metadata_error"}
                )
                continue

            decision = risk.evaluate(
                intent=intent,
                account=snapshot.account,
                existing_positions=snapshot.positions,
                mark_price=mark_price,
                rules=rules,
            )

            outcome = execution.execute(intent=intent, risk=decision, snapshot=snapshot, mark_price=mark_price)

            if intent.action in (IntentAction.EXIT_LONG, IntentAction.EXIT_SHORT) and outcome.filled_qty > 0:
                risk.record_trade_result(outcome.realized_pnl, stopped_out=outcome.stopped_out)

            counters.inc("signals_total")
            counters.inc(f"intent_{intent.action.value.lower()}")
            counters.inc(f"exec_{outcome.status.lower()}")
            layer_trace = intent.metadata.get("layer_trace", {}) if isinstance(intent.metadata, dict) else {}
            layer_failed = intent.metadata.get("layer_failed", "") if isinstance(intent.metadata, dict) else ""
            layer4 = {}
            if isinstance(layer_trace, dict):
                layer4 = (
                    layer_trace.get("layers", {})
                    .get("layer4_fake_filter", {})
                    .get("details", {})
                )
            sentiment_mode = layer4.get("sentiment_source", "n/a") if isinstance(layer4, dict) else "n/a"
            sentiment_degraded = False
            if isinstance(layer4, dict):
                sentiment_degraded = bool(float(layer4.get("degraded_mode", 0.0) or 0.0))

            logger.info(
                "symbol=%s state=%s intent=%s risk=%s exec=%s reason=%s layer_failed=%s sentiment_mode=%s sentiment_degraded=%s",
                symbol,
                rec_state.state.value,
                intent.action.value,
                decision.reason,
                outcome.status,
                outcome.reason,
                layer_failed,
                sentiment_mode,
                sentiment_degraded,
                extra={"event": "decision"},
            )

            if intent.action in (IntentAction.LONG_ENTRY, IntentAction.SHORT_ENTRY) and outcome.accepted:
                _send_alerts(
                    alerters,
                    f"{intent.action.value} {symbol} qty={outcome.filled_qty:.6f} reason={intent.reason}",
                )

        except Exception as exc:
            counters.inc("cycle_errors")
            logger.exception("cycle_error symbol=%s err=%s", symbol, exc, extra={"event": "cycle_error"})



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bybit Futures Bot V2")
    parser.add_argument("--strategy", choices=["layered", "hold"], default="layered")
    parser.add_argument("--loop", action="store_true", help="Run continuous loop")
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    logger = setup_logging("INFO")

    try:
        cfg = load_runtime_config()
    except ConfigError as exc:
        logger.error("startup_config_error=%s", exc, extra={"event": "config_error"})
        return 2

    try:
        from trading.features.pipeline import FeaturePipeline
        from trading.market_data.feed import MarketDataFeed
        from trading.market_data.reconciliation import ExchangeReconciler
        from trading.market_data.ws_reconciliation import ExchangeSyncService
    except Exception as exc:
        logger.error("startup_dependency_error=%s", exc, extra={"event": "dependency_error"})
        return 2

    adapter = BybitAdapter(cfg.adapter)
    adapter.set_ws_symbols(cfg.symbols)

    feed = MarketDataFeed(base_url="https://api.bybit.com")
    reconciler = ExchangeReconciler(adapter)
    sync = ExchangeSyncService(
        reconciler,
        poll_interval_sec=cfg.ws_poll_interval_sec,
        max_event_staleness_sec=cfg.ws_event_staleness_sec,
    )
    runtime_store = RuntimeStore(cfg.runtime_db_path)

    state_machine = StateMachine(persistence=runtime_store)
    risk = RiskEngine(cfg.risk_limits, persistence=runtime_store)
    execution = ExecutionEngine(
        adapter=adapter,
        state_machine=state_machine,
        hedge_mode=cfg.adapter.hedge_mode,
        stop_loss_required=cfg.risk_limits.require_stop_loss,
        require_reconciliation=cfg.flags.reconciliation_required,
        stop_attach_grace_sec=cfg.stop_attach_grace_sec,
        stale_open_order_sec=cfg.stale_open_order_sec,
        max_exchange_retries=cfg.max_exchange_retries,
        persistence=runtime_store,
    )
    pipeline = FeaturePipeline()
    strategy = _build_strategy(args.strategy)
    counters = MetricsCounter()
    alerters = _build_alerters(cfg)

    startup_state = _startup_reconcile(
        symbols=cfg.symbols,
        sync=sync,
        state_machine=state_machine,
        execution=execution,
    )
    maintenance = runtime_store.maintenance()
    inflight = runtime_store.load_open_inflight_intents()
    logger.info(
        "startup_safety mode=%s testnet=%s dry_run=%s symbols=%d db=%s inflight=%d states=%s maintenance=%s live_cap=%s",
        cfg.mode,
        cfg.adapter.testnet,
        cfg.adapter.dry_run,
        len(cfg.symbols),
        cfg.runtime_db_path,
        len(inflight),
        startup_state,
        maintenance,
        cfg.live_startup_max_notional_usdt,
        extra={"event": "startup_safety"},
    )

    if cfg.mode == "live":
        _send_alerts(
            alerters,
            f"[STARTUP] LIVE mode enabled with cap={cfg.live_startup_max_notional_usdt:.2f} USDT symbols={','.join(cfg.symbols)}",
        )

    last_maintenance_ts = time.time()
    state_alert_cache: dict[str, str] = {}
    try:
        while True:
            run_cycle(
                symbols=cfg.symbols,
                adapter=adapter,
                feed=feed,
                sync=sync,
                pipeline=pipeline,
                strategy=strategy,
                risk=risk,
                execution=execution,
                logger=logger,
                counters=counters,
                timeframe=cfg.timeframe,
                candles_limit=cfg.candles_limit,
                alerters=alerters,
                state_alert_cache=state_alert_cache,
            )

            now = time.time()
            if now - last_maintenance_ts >= cfg.maintenance_interval_sec:
                runtime_store.maintenance()
                last_maintenance_ts = now

            health = sync.health()
            logger.info(
                "metrics=%s risk=%s sync=%s metadata=%s",
                counters.snapshot(),
                risk.health_snapshot(),
                {
                    "ws_connected": health.ws_connected,
                    "ws_stale": health.ws_stale,
                    "fallback_polling": health.fallback_polling,
                    "snapshot_required": health.snapshot_required,
                },
                adapter.metadata_health(),
                extra={"event": "health"},
            )

            if not args.loop:
                break
            time.sleep(max(1, cfg.scan_interval_sec))
    finally:
        feed.close()
        adapter.close()
        runtime_store.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())




