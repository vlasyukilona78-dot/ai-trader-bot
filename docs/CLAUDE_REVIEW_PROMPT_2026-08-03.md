# Claude Code re-entry and independent review prompt

This file is intentionally self-contained. It lets a fresh Claude Code account
enter either the ordinary project root or the selected MEXC worktree without
depending on prior chat history.

## Exact repository orientation

```text
Project root / preserved Bybit-runtime line:
C:\Users\vlasy\PycharmProjects\koteika_Ultra
branch: feat/phase2-layer1-pump-runtime-alignment
pre-handoff root anchor: d6a69d5

Selected MEXC causal research line:
C:\Users\vlasy\PycharmProjects\koteika_Ultra\.claude\worktrees\codex-project-review-04581e
branch: claude/codex-project-review-04581e
AI foundation anchor: f0b43d6
Phase 1 hardening code tip: e0e4cb4
remote: origin/claude/codex-project-review-04581e
```

The two lines are intentionally unmerged. MEXC is the selected target exchange.
Do not implement the MEXC roadmap in root/Bybit and do not copy code between the
worktrees without an explicit transfer plan.

## Mandatory read-only orientation

Run from PowerShell. Do not switch branches, pull, reset, clean, start Python,
run the scanner, contact MEXC/Telegram, or read `.env` during this review.

```powershell
$ROOT = 'C:\Users\vlasy\PycharmProjects\koteika_Ultra'
$MEXC = Join-Path $ROOT '.claude\worktrees\codex-project-review-04581e'

git -C $ROOT status --short --branch
git -C $ROOT worktree list --porcelain
git -C $ROOT rev-parse HEAD
git -C $ROOT rev-parse '@{upstream}'
git -C $ROOT log -5 --oneline --decorate

git -C $MEXC status --short --branch
git -C $MEXC rev-parse HEAD
git -C $MEXC rev-parse '@{upstream}'
git -C $MEXC log -8 --oneline --decorate
git -C $MEXC diff --stat 1e91ce0..HEAD
git -C $MEXC diff --check 1e91ce0..HEAD
git -C $MEXC merge-base --is-ancestor f0b43d6 HEAD
git -C $MEXC merge-base --is-ancestor e0e4cb4 HEAD
git -C $MEXC ls-files -- .env
```

Expected facts at the recorded checkpoint:

- MEXC local and remote tips match, `f0b43d6` and `e0e4cb4` are ancestors, and the
  worktree is clean;
- current `.env` is not tracked; old history still contained credentials;
- root may contain user-owned `.idea/*` changes — preserve and ignore them;
- no positive trading edge has been established.

If reality differs, report the difference before drawing conclusions. Do not
silently repair or discard it.

## Required reading order

Read every listed file in full unless the item explicitly names a section:

1. Root `AGENTS.md` and root `CLAUDE.md`.
2. MEXC `CLAUDE.md`.
3. MEXC `docs/STRATEGY_AI_MASTER_PLAN_2026-08-03.md` — authoritative plan.
4. MEXC `docs/AI_HANDOFF.md`:
   - `Phase 1 evidence hardening — 2026-08-08`;
   - `Current authoritative checkpoint — 2026-08-03`;
   - `Claude no-edge finding — 2026-07-26`;
   - `Codex independent review of Claude's nine follow-up commits`;
   - `Unified strategy/AI contract foundation — 2026-08-03`.
5. Root `docs/CURRENT_STATE_AND_AI_PLAN_2026-08-03.md` and
   `docs/PROJECT_HANDOFF_FOR_CLAUDE.md` for preservation,
   worktree and historical system context. Where its older MEXC roadmap differs,
   the MEXC master plan supersedes it.
6. Executable implementation and tests listed below.

## Recent change set to inspect commit by commit

```text
0b010e8 fix(backtest): prevent hindsight cohort substitution
3ff8de0 feat(ai): add causal MEXC feature contract
29536f1 docs: unify MEXC strategy and AI roadmap
f0b43d6 docs: record AI foundation publication state
32e8fbe fix(journal): harden causal population evidence
0c32047 fix(backtest): bind replay outcomes to schema-v3 evidence
e0e4cb4 fix(strategy): fail closed without benchmark context
```

The Claude re-entry files are a later documentation-only change. Discover their
exact commit from `git log`; do not hard-code a self-referential latest HEAD.

For each commit, use `git show --stat <hash>` and `git show <hash> -- <file>` as
needed. Do not accept the documentation as proof that the implementation is
correct.

## Code paths that must be inspected

- `backtesting/single_position.py`
- `ai/reversal/feature_contract.py`
- `ai/reversal/population_dataset.py`
- `app/scan.py`
- `core/signal_generator.py`
- `trading/signals/layered_strategy.py`
- `trading/signals/strategy_interface.py`
- `trading/market_data/universe.py`
- `trading/market_data/bar_contract.py`
- `trading/market_data/feed.py`
- `trading/market_data/mexc_client.py`
- `trading/market_data/timeframe_cache.py`
- `trading/metrics/population_journal.py`
- `trading/metrics/cycle_envelope.py`
- `trading/alerts/telegram.py`
- `ai/train.py` only to understand why it is legacy and must not be reused
- `tests/v2/test_reversal_feature_contract_v2.py`
- `tests/v2/test_population_feature_dataset_v2.py`
- `tests/v2/test_single_position_contract_v2.py`
- `tests/v2/test_single_position_schema_v3.py`
- `tests/v2/test_single_position_result_invariants_v2.py`
- `tests/v2/test_journal_cycle_records_v2.py`
- `tests/v2/test_scan_journal_reader_e2e_v2.py`
- `tests/v2/test_scan_source_provenance_v2.py`
- `tests/v2/test_scan_v2.py`
- `tests/v2/test_closed_bar_contract_v2.py`
- `tests/v2/test_population_journal_v2.py`
- `tests/v2/test_layered_strategy_thread_safety_v2.py`
- `tests/v2/test_telegram_alerter_v2.py`
- `tests/v2/test_timeframe_cache_v2.py`
- `tests/v2/test_turnover_units_v2.py`
- `tests/test_strategy_dry_run.py`

Confirm from code and tests, not only prose, that:

1. The single-position selector chooses the top causal candidate before looking
   at future fills, does not substitute a filled runner-up retrospectively, and
   rejects a rehashed outcome that differs from mandatory ReplayEvidence.
2. The feature contract is versioned, has a pinned executable schema hash,
   separates MODEL/PROPOSAL/POLICY/CONTEXT/DIAGNOSTIC roles, and distinguishes
   observed zero from missing/unavailable.
3. Every journal row, including HOLD/error, receives the same captured schema.
4. Funding from the frozen universe snapshot reaches StrategyContext.
5. Strategy/universe config hashes and snapshot/cycle/input IDs are validated.
6. The strict reader rejects malformed/incomplete/drifting cycles and the
   model-input API exposes only the model whitelist.
7. Legacy event-conditioned CSV cannot be mistaken for the new admission
   dataset.

## Required adversarial review

Explicitly verify or challenge these still-open items. They are planned work,
not completed claims:

1. Per-symbol base and per-symbol/per-timeframe HTF timing. Cycle-level source
   provenance and research `entry_eligible_ts` are implemented, but delivery
   timing is not.
2. Exact StrategySpec timeframe semantics, warm-up and physical windows. The
   first research execution bar is reachable; the current Min60 defaults still
   do not express the intended fast-pump horizons.
3. Ledger of all point-in-time MEXC USDT contracts, with inclusion/exclusion
   reasons, rather than only the filtered scan universe.
4. Gate-independent eager feature computation for every valid symbol. Current
   layer trace still stops after failed gates.
5. Separate identities for `MarketFeatureSnapshot`, `RuleEvaluation`,
   `TradeProposal`, `OutcomeLabel` and `ShadowPrediction`.
6. Point-in-time instrument specs: contract size, quantity step, minimums,
   leverage rules, source timestamp and hash.
7. Normalized OI notional and POC/VAH/VAL distances; raw cross-symbol values are
   diagnostic, not model inputs.
8. A cycle envelope/attempt journal that can prove full-universe completeness,
   including empty-universe and pre-scan failures.
9. Atomic/two-pass validation before export or training so a later corrupt cycle
   cannot produce a partially consumed dataset.
10. Single-position outcome labels wired to the population journal with exact
    fees, spread, slippage, funding, stop/TP geometry and concurrency.
11. Purged chronological evaluation, 48-hour purge/embargo where applicable,
    untouched test, symbol/time clustered uncertainty and matched random/rules/
    no-trade baselines.
12. Confirmation lifecycle and cycle/cohort grouping: separate arm-time,
    confirmation-time and proposal-time state without overwriting earlier data.
13. Full runtime parity for causal Fibonacci/overhead, weakness, confluence,
    1h RSI and liquidation context rather than offline-only declarations.
14. The legacy `ai/train.py`, old event-conditioned datasets and old calibrated
    thresholds must remain outside the new admission path.

## Safe secrets/runtime status checks

These checks inspect only existence/index/history metadata. Never open `.env` or
show its historical blobs/content.

```powershell
Test-Path -LiteralPath (Join-Path $ROOT '.env')
Test-Path -LiteralPath (Join-Path $MEXC '.env')
git -C $ROOT ls-files -- .env
git -C $MEXC ls-files -- .env
git -C $ROOT check-ignore -v -- .env
git -C $MEXC check-ignore -v -- .env
git -C $ROOT log --all --format='%H %cs' -- .env
Get-Process -Name python,pythonw -ErrorAction SilentlyContinue
```

Report separately whether `.env` exists locally, is tracked now, existed in
history and has been rotated. Physical ignored files may exist even though the
current Git tips do not track `.env`.

## Model and tooling decision to verify

The proposed order is:

1. logistic/rules/random/no-trade baselines;
2. small LightGBM multiclass candidate plus a separate conditional-payoff/EV
   head after prospective labels mature;
3. CatBoost challenger and XGBoost AFT auxiliary time-to-event model;
4. TCN/Chronos only after tabular evidence;
5. Kimi/OpenAI/Gemini-class LLMs only for timestamped public-text extraction,
   never as the numeric trading or risk engine.

Planned tooling is staged deliberately: Parquet + DuckDB first, local MLflow
second, bounded Optuna only inside train/validation, and Evidently/ONNX/DVC only
after each solves an observed problem. Challenge this order if code/data facts
support a better one, but do not choose a model by popularity or prose quality.

## Validation

After the read-only audit, the allowed local verification command is:

```powershell
& 'C:\Users\vlasy\PycharmProjects\koteika_Ultra\.venv\Scripts\python.exe' `
  -m pytest -q
```

Run it from the MEXC worktree. The recorded result is:

```text
529 passed, 4 skipped, 2 known PytestCollectionWarning (13.85s)
```

Do not run the scanner or any test that requires private credentials/network
delivery. If the suite result differs, identify whether the tree or environment
changed.

## Required response format

Before making any code changes, return one review containing:

1. exact worktree/branch/HEAD and whether local equals remote;
2. a commit-by-commit explanation of what changed;
3. a causal data-flow map from MEXC sources to snapshot, rules, proposal, label,
   model and single-position replay;
4. what is implemented, partially implemented and merely planned;
5. contradictions between docs, code and tests;
6. leakage/look-ahead, population-selection, execution and reproducibility risks
   ranked P0/P1/P2 with file/line evidence;
7. an assessment of the LightGBM + EV / challenger / LLM architecture;
8. exact test evidence and what those tests do not prove;
9. a corrected implementation plan with acceptance criteria;
10. a final operational verdict: bot state, edge status and safest next code task.

Accept a repeat no-edge result. Do not tune thresholds or models until the time,
population, feature and executable-label contracts are proven causal.
