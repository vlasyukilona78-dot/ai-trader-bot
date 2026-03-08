# Pre-Live Validation Runbook (V2)

## Purpose
Run a real Bybit testnet validation pass against the V2 runtime before any capped-live rollout.

## Command

```powershell
python -m app.testnet_validation --execute-orders --run-full-suite --symbol BTCUSDT --max-notional-usdt 20 --soak-seconds 180 --chaos-cycles 8 --report-out logs/testnet_validation_report.json
```

## Required Environment
- `BOT_RUNTIME_MODE=testnet`
- `LIVE_TRADING_ENABLED=false`
- `BYBIT_TESTNET=true`
- `BYBIT_API_KEY` and `BYBIT_API_SECRET` set
- `WS_ENABLED=true`
- `WS_PRIVATE_ENABLED=true`

## Validation Outcomes
The harness writes a JSON report containing:
- startup/testnet preflight checks
- tiny capped-notional lifecycle checks
- websocket normalization and reconnect fallback checks
- restart and recovery scenarios
- soak/chaos checks
- runtime dependency manifest
- go/no-go decisions for paper, testnet, capped-live

## Exit Criteria
- no `FAIL` scenarios
- no `BLOCKED` scenarios for required testnet checks
- `v2_suite_target_runtime` with `returncode=0` and `skipped=0`
- `testnet` decision = `GO`
- `capped_live` decision at most `CONDITIONAL` with no safety blockers
