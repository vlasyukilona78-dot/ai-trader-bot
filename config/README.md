# Configuration

- Runtime secrets are loaded from environment variables only.
- Do not create `config/secrets.env` in source control.
- Use `config/secrets.env.example` only as a template for local, untracked setup.
- Safe defaults are fail-closed: live trading and online retraining disabled unless explicitly enabled.

## Runtime Mode

Set `BOT_RUNTIME_MODE` explicitly to one of:
- `dry_run`
- `paper`
- `testnet`
- `live`

Rules:
- `LIVE_TRADING_ENABLED=true` is allowed only with `BOT_RUNTIME_MODE=live`.
- `BOT_RUNTIME_MODE=live` requires `LIVE_STARTUP_MAX_NOTIONAL_USDT > 0`.
- `testnet/live` require `BYBIT_API_KEY` and `BYBIT_API_SECRET`.

## Dependency Gates

- `FEATURE_RUNTIME_ENABLED=true` requires `numpy` and `pandas`.
- `ML_INFERENCE_ENABLED=true` requires `numpy`, `pandas`, `joblib`, and `scikit-learn`.
- `WS_ENABLED=true` requires `websockets`.
