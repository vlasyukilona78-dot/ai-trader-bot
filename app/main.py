from __future__ import annotations

import argparse
import time
from collections.abc import Mapping
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
from trading.signals.runtime_source_adapter import build_runtime_signal_inputs
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



def _collect_runtime_payload(frame) -> dict[str, object]:
    payload: dict[str, object] = {}
    frame_payload = getattr(frame, "runtime_payload", None)
    if isinstance(frame_payload, Mapping):
        payload.update(frame_payload)

    ohlcv_attrs = getattr(frame.ohlcv, "attrs", None)
    if isinstance(ohlcv_attrs, Mapping):
        for key in ("runtime_payload", "signal_sources", "source_payload"):
            value = ohlcv_attrs.get(key)
            if isinstance(value, Mapping):
                payload.update(value)

    return payload


def _strategy_audit_log_payload(strategy) -> dict[str, object]:
    full_snapshot = {}
    compact_snapshot = {}
    layer4_snapshot = {}
    source_quality_snapshot = {}

    if hasattr(strategy, "audit_observation_snapshot"):
        try:
            observation = strategy.audit_observation_snapshot()
        except Exception:
            observation = {}
        if isinstance(observation, Mapping):
            if isinstance(observation.get("strategy_audit_compact"), Mapping):
                compact_snapshot = dict(observation.get("strategy_audit_compact", {}))
            if isinstance(observation.get("strategy_audit_layer4"), Mapping):
                layer4_snapshot = dict(observation.get("strategy_audit_layer4", {}))
            if isinstance(observation.get("strategy_audit_source_quality"), Mapping):
                source_quality_snapshot = dict(observation.get("strategy_audit_source_quality", {}))

    if hasattr(strategy, "audit_snapshot"):
        try:
            snapshot_candidate = strategy.audit_snapshot()
        except Exception:
            snapshot_candidate = {}
        if isinstance(snapshot_candidate, Mapping):
            full_snapshot = dict(snapshot_candidate)

    if not compact_snapshot:
        if hasattr(strategy, "audit_compact_snapshot"):
            try:
                compact_candidate = strategy.audit_compact_snapshot()
            except Exception:
                compact_candidate = {}
            if isinstance(compact_candidate, Mapping):
                compact_snapshot = dict(compact_candidate)
        if not compact_snapshot:
            compact_snapshot = dict(full_snapshot)

    if not layer4_snapshot:
        layer4_snapshot = {
            "layer4_fail_count": int(full_snapshot.get("layer4_fail_count", 0)),
            "layer4_sentiment_blocker_count": int(full_snapshot.get("layer4_sentiment_blocker_count", 0)),
            "layer4_funding_blocker_count": int(full_snapshot.get("layer4_funding_blocker_count", 0)),
            "layer4_lsr_blocker_count": int(full_snapshot.get("layer4_lsr_blocker_count", 0)),
            "layer4_oi_blocker_count": int(full_snapshot.get("layer4_oi_blocker_count", 0)),
            "layer4_price_blocker_count": int(full_snapshot.get("layer4_price_blocker_count", 0)),
            "layer4_degraded_mode_count": int(full_snapshot.get("layer4_degraded_mode_count", 0)),
            "layer4_soft_pass_candidate_count": int(full_snapshot.get("layer4_soft_pass_candidate_count", 0)),
        }

    if not source_quality_snapshot and isinstance(full_snapshot.get("source_quality_summary"), Mapping):
        source_quality_snapshot = dict(full_snapshot.get("source_quality_summary", {}))

    return {
        "strategy_audit_compact": compact_snapshot,
        "strategy_audit_layer4": layer4_snapshot,
        "strategy_audit_source_quality": source_quality_snapshot,
        "strategy_audit": full_snapshot,
    }


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
            runtime_payload = _collect_runtime_payload(frame)
            runtime_inputs = build_runtime_signal_inputs(frame.ohlcv, runtime_payload=runtime_payload)
            extras = {
                "sentiment_index": runtime_inputs.get("sentiment_index"),
                "funding_rate": runtime_inputs.get("funding_rate"),
                "long_short_ratio": runtime_inputs.get("long_short_ratio"),
                "open_interest": runtime_inputs.get("open_interest"),
                "news_veto": runtime_inputs.get("news_veto"),
                "news_source": runtime_inputs.get("news_source"),
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
                    sentiment_index=runtime_inputs.get("sentiment_index"),
                    sentiment_value=runtime_inputs.get("sentiment_value"),
                    sentiment_source=runtime_inputs.get("sentiment_source"),
                    sentiment_degraded=runtime_inputs.get("sentiment_degraded"),
                    funding_rate=runtime_inputs.get("funding_rate"),
                    funding_source=runtime_inputs.get("funding_source"),
                    funding_degraded=runtime_inputs.get("funding_degraded"),
                    long_short_ratio=runtime_inputs.get("long_short_ratio"),
                    long_short_ratio_source=runtime_inputs.get("long_short_ratio_source"),
                    long_short_ratio_degraded=runtime_inputs.get("long_short_ratio_degraded"),
                    open_interest=runtime_inputs.get("open_interest"),
                    open_interest_ratio=runtime_inputs.get("open_interest_ratio"),
                    oi_signal=runtime_inputs.get("oi_signal"),
                    oi_source=runtime_inputs.get("oi_source"),
                    oi_degraded=runtime_inputs.get("oi_degraded"),
                    open_interest_source=runtime_inputs.get("open_interest_source"),
                    news_veto=runtime_inputs.get("news_veto"),
                    news_source=runtime_inputs.get("news_source"),
                    news_degraded=runtime_inputs.get("news_degraded"),
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
            regime_diag = {}
            if isinstance(layer_trace, dict):
                layer4 = (
                    layer_trace.get("layers", {})
                    .get("layer4_fake_filter", {})
                    .get("details", {})
                )
                regime_diag = (
                    layer_trace.get("layers", {})
                    .get("regime_filter", {})
                    .get("details", {})
                )
            sentiment_mode = layer4.get("sentiment_source", "n/a") if isinstance(layer4, dict) else "n/a"
            sentiment_degraded = False
            sentiment_quality = "n/a"
            regime_news_quality = "n/a"
            if isinstance(layer4, dict):
                sentiment_degraded = bool(float(layer4.get("degraded_mode", 0.0) or 0.0))
                sentiment_quality = (
                    layer4.get("source_flags", {}).get("sentiment_quality")
                    or layer4.get("source_quality", {}).get("sentiment")
                    or "n/a"
                )
            if isinstance(regime_diag, dict):
                regime_news_quality = regime_diag.get("source_flags", {}).get("news_quality") or "n/a"

            logger.info(
                "symbol=%s state=%s intent=%s risk=%s exec=%s reason=%s layer_failed=%s sentiment_mode=%s sentiment_quality=%s regime_news_quality=%s sentiment_degraded=%s",
                symbol,
                rec_state.state.value,
                intent.action.value,
                decision.reason,
                outcome.status,
                outcome.reason,
                layer_failed,
                sentiment_mode,
                sentiment_quality,
                regime_news_quality,
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
            strategy_audit_payload = _strategy_audit_log_payload(strategy)
            logger.info(
                "metrics=%s risk=%s sync=%s metadata=%s strategy_audit_compact=%s strategy_audit_layer4=%s strategy_audit_source_quality=%s strategy_audit=%s",
                counters.snapshot(),
                risk.health_snapshot(),
                {
                    "ws_connected": health.ws_connected,
                    "ws_stale": health.ws_stale,
                    "fallback_polling": health.fallback_polling,
                    "snapshot_required": health.snapshot_required,
                },
                adapter.metadata_health(),
                strategy_audit_payload.get("strategy_audit_compact", {}),
                strategy_audit_payload.get("strategy_audit_layer4", {}),
                strategy_audit_payload.get("strategy_audit_source_quality", {}),
                strategy_audit_payload.get("strategy_audit", {}),
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









