# V2 Migration Notes

## Active entrypoint
- `main.py` now delegates to `app/main.py`.
- V2 loop enforces exchange reconciliation before each decision.

## Ownership boundaries
- Order placement: `trading/execution/engine.py`
- Risk approvals/sizing: `trading/risk/engine.py`
- Bybit-specific API mapping: `trading/exchange/bybit_adapter.py`
- State transitions/reconciliation: `trading/state/machine.py`

## Legacy modules
- `main_legacy_monolith.py`, `engine/*`, and `core/*` remain for compatibility/reference.
- They are not V2 control-plane modules.

## Security changes
- `config/secrets.env` removed from runtime path.
- `.gitignore` updated to exclude secret/env/data/artifacts.
- `config/secrets.env.example` kept as template only.

## Safe defaults
- Live trading fail-closed unless `LIVE_TRADING_ENABLED=true`.
- Online retraining disabled by default.
- Stop-loss required for entries by default.

## Repo cleanup plan (follow-up)
1. Rotate all historical API keys/tokens that may have ever touched this repo.
2. Remove committed artifacts from git history (`*.pkl`, `data/*.csv`, `signals_history.xlsx`, debug outputs).
3. Move production model artifacts to external storage/object registry.
4. Keep only reproducible source + tests in the main branch.
