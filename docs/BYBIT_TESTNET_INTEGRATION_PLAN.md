## Automated Harness

Run the V2 testnet harness for an operational pass/fail report:

```powershell
python -m app.testnet_validation --execute-orders --run-full-suite --symbol BTCUSDT --max-notional-usdt 20 --soak-seconds 180 --chaos-cycles 8 --report-out logs/testnet_validation_report.json
```
# Bybit Testnet Integration Plan (V2)

## Preconditions
- `BOT_RUNTIME_MODE=testnet`
- `LIVE_TRADING_ENABLED=false`
- `BYBIT_API_KEY` and `BYBIT_API_SECRET` set for testnet account
- `WS_ENABLED=true` and `WS_PRIVATE_ENABLED=true`
- isolated symbol list (`BOT_SYMBOLS=BTCUSDT`) for first run
- reduced limits for test rollout (`RISK_MAX_TOTAL_NOTIONAL_PCT<=0.10`)

## Scenarios
1. Startup reconciliation
- start process with no local DB, confirm startup safety report and schema version.
- restart process with open position and open orders to verify restart recovery paths.

2. Real order lifecycle
- place entry order, verify exchange fill reflected through WS and polling fallback.
- confirm exactly one execution attempt for a single intent id.

3. Partial fill lifecycle
- force partial fill by using larger quantity / low liquidity hours.
- verify stop is attached for filled quantity only.
- verify remaining entry order is reconciled and stale cleanup works.

4. Stop-attach failure recovery
- inject temporary failure (block private WS or simulate API error).
- verify engine enters deterministic recovery: attach retry -> protective close -> HALT if still unsafe.

5. Cancel/replace and stale order cleanup
- create open entry order, restart bot, verify stale threshold cancellation path.
- verify no duplicated open orders after reconnect.

6. Websocket reconnect chaos
- disconnect network or kill websocket connection.
- verify `SNAPSHOT_REQUIRED` events trigger polling fallback and state re-sync.

7. Rate limit and retry behavior
- reduce recv window / burst requests to induce transient rate limits.
- verify bounded retries and no duplicate fills.

8. Manual exchange-side intervention
- manually close/reverse position in Bybit UI.
- verify intervention detection transitions to `RECOVERING` or `HALTED` and emits high-severity alert.

## Evidence to collect
- structured logs: `startup_safety`, `decision`, `intervention`, `health`
- DB tables: `state_transitions`, `inflight_intents`, `order_decisions`
- exchange screenshots: positions/orders before and after restart

## Exit criteria for capped-live readiness
- zero duplicate order events under retry/reconnect tests
- all stop attachment failures end in protected or halted state (never naked)
- recovery scenarios deterministic across at least 20 restarts
- no unresolved `RECOVERING` states longer than configured grace window

