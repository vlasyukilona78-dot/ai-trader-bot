# Legacy Runtime Quarantine

Legacy runtime modules are intentionally blocked by default.

Blocked modules include old execution/risk/state implementations and monolith entry scripts.
To import them for archive-only debugging, set:

`ALLOW_LEGACY_RUNTIME=true`

Production runtime path is:

- `main.py` -> `app/main.py`
- `trading/*` V2 modules only
