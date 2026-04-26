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

## Telegram Proxy

Use `TELEGRAM_PROXY_URL` only for Telegram delivery when Telegram is blocked but Bybit works directly.
Supported formats are standard Requests proxy URLs, for example:

- `http://user:pass@host:port`
- `socks5h://user:pass@host:port`
- `http://127.0.0.1:10801` for a local V2RayTun system proxy while it is running

Do not route Bybit through this variable. Bybit should stay direct unless you explicitly configure a separate exchange proxy.

## Coinglass Liquidation Heatmap

Set `COINGLASS_API_KEY` to enable the real Coinglass liquidation heatmap source. The bot keeps a safe fallback to the internal liquidation map when the key is missing, the endpoint times out, or the response cannot be parsed.

Recommended defaults:

- `COINGLASS_LIQUIDATION_ENABLED=true`
- `COINGLASS_LIQUIDATION_EXCHANGE=Bybit`
- `COINGLASS_LIQUIDATION_RANGE=3d`
- `COINGLASS_LIQUIDATION_MODEL=model2`
- `COINGLASS_LIQUIDATION_MIN_INTENSITY=0.12`
- `COINGLASS_LIQUIDATION_MAX_BANDS_PER_SIDE=5`
- `COINGLASS_LIQUIDATION_RATE_LIMIT_COOLDOWN_SEC=900`

Use `COINGLASS_PROXY_URL` only if Coinglass itself needs a proxy. Keep it empty when direct access is stable.
If Coinglass returns `429`, the bot pauses only the external heatmap source for this cooldown and keeps using the internal fallback map instead of spamming the API.
