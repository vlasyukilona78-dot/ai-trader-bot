# Claude Code re-entry: independent read-only audit of MEXC evidence schema v6

This prompt is intentionally self-contained. It is for a fresh Claude Code
account reviewing the preserved root/Bybit line and the selected MEXC causal
research line after typed lifecycle, exact per-symbol closed-frame provenance and
population-journal schema v6 were committed.

This task is an **independent read-only audit, not implementation**. Do not edit
files, stage, commit, push, pull, switch branches, reset, clean, repair journals,
launch the bot/scanner, train a model, contact MEXC/Telegram or inspect secret
contents. Return findings and a corrected plan only.

## Non-negotiable status

- MEXC futures is the selected target exchange.
- Root/Bybit and MEXC are intentionally separate lines.
- Bot and scanner are stopped; the selected MEXC line has no private execution.
- No stable generic pump-fade edge has been established after costs.
- No causal model has been fitted, promoted or enabled.
- Current scope is signals-only causal research. Private API, Telegram, testnet
  and live remain forbidden.
- Ignored `.env` files may exist. Credentials occurred in reachable older Git
  history and rotation is not confirmed. Never read or print present or historical
  values.
- Accept repeat no-edge. Do not tune thresholds, windows or models during review.
- The frozen `mexc_strategy_v2` remains Min60/45-hour behavior. No v3 timeframe,
  window or threshold has been selected.
- Gate-conditioned feature missingness still blocks model training. A typed
  lifecycle is not an unconditional market snapshot.

## Exact repository orientation

```text
Root / preserved Bybit-runtime line:
C:\Users\vlasy\PycharmProjects\koteika_Ultra
branch: feat/phase2-layer1-pump-runtime-alignment
known root tip: f01591f2e2872d5af6341580b6d8d44b298cd244

Selected MEXC causal research line:
C:\Users\vlasy\PycharmProjects\koteika_Ultra\.claude\worktrees\codex-project-review-04581e
branch: claude/codex-project-review-04581e
remote: origin/claude/codex-project-review-04581e
AI foundation anchor: f0b43d6
StrategySpec/journal-v5 anchors: bebfd0d / 2d0efcb
frozen behavior anchor: 258c35f
versioned-evidence anchor: 1971b77
frozen-v5 evidence boundary: eb238b2
typed-lifecycle hardening tip: 9ef6b4f
current executable journal-v6 tip: bb1ca13
```

Documentation may be committed after `bb1ca13`. Discover exact current HEAD and
upstream. Treat `bb1ca13` as the executable tip under review, not automatically
as final documentation HEAD. Inventory every worktree and modify none.

## Mandatory read-only orientation

Run from PowerShell:

```powershell
$ROOT = 'C:\Users\vlasy\PycharmProjects\koteika_Ultra'
$MEXC = Join-Path $ROOT '.claude\worktrees\codex-project-review-04581e'

git -C $ROOT status --short --branch
git -C $ROOT worktree list --porcelain
git -C $ROOT rev-parse HEAD
git -C $ROOT rev-parse '@{upstream}'

git -C $MEXC status --short --branch
git -C $MEXC rev-parse HEAD
git -C $MEXC rev-parse '@{upstream}'
git -C $MEXC log -18 --oneline --decorate
git -C $MEXC diff --stat 1971b77..bb1ca13
git -C $MEXC diff --check 1971b77..bb1ca13

git -C $MEXC merge-base --is-ancestor f0b43d6 HEAD
git -C $MEXC merge-base --is-ancestor bebfd0d HEAD
git -C $MEXC merge-base --is-ancestor 2d0efcb HEAD
git -C $MEXC merge-base --is-ancestor 258c35f HEAD
git -C $MEXC merge-base --is-ancestor 1971b77 HEAD
git -C $MEXC merge-base --is-ancestor eb238b2 HEAD
git -C $MEXC merge-base --is-ancestor 9ef6b4f HEAD
git -C $MEXC merge-base --is-ancestor bb1ca13 HEAD
git -C $MEXC ls-files -- .env
```

Expected published checkpoint:

- MEXC descends from every anchor above; executable tip is `bb1ca13`;
- root remains at/upstream `f01591f`; three root `.idea/*` changes, if still
  present, are user-owned and must remain untouched;
- MEXC should be clean and equal to upstream after documentation publication;
- `.env` is not tracked at either selected tip but existed in older history;
- no Python bot/scanner process runs;
- no edge, model-readiness or live-readiness claim exists.

Report any discrepancy before interpreting code. Do not repair it.

## Required reading order

Read in full:

1. Root `AGENTS.md` and root `CLAUDE.md`.
2. MEXC `CLAUDE.md`.
3. MEXC `docs/STRATEGY_AI_MASTER_PLAN_2026-08-03.md` (authoritative plan).
4. MEXC `docs/AI_HANDOFF.md`, especially its newest v6/lifecycle checkpoint and
   the earlier no-edge, StrategySpec, journal-v5 and independent-audit sections.
5. Root `docs/CURRENT_STATE_AND_AI_PLAN_2026-08-03.md` and
   `docs/PROJECT_HANDOFF_FOR_CLAUDE.md` for historical/preservation context. If
   they conflict with the MEXC master plan, report it and use the MEXC plan.
6. Executable code and tests listed below.

## Commits to inspect individually

```text
0b010e8 fix(backtest): prevent hindsight cohort substitution
3ff8de0 feat(ai): add causal MEXC feature contract
32e8fbe fix(journal): harden causal population evidence
0c32047 fix(backtest): bind replay outcomes to schema-v3 evidence
e0e4cb4 fix(strategy): fail closed without benchmark context
bebfd0d feat(strategy): define canonical MEXC strategy spec
2d0efcb feat(journal): chain schema-v5 population evidence
258c35f test(strategy): pin v2 behavioral semantics
1971b77 feat(strategy): preserve versioned evidence compatibility
eb238b2 test(journal): freeze schema v5 evidence boundary
c541eea feat(evidence): define candidate lifecycle contract
a604668 feat(evidence): bind exact closed-frame provenance
8569471 fix(evidence): reject malformed empty frame reads
cf6bc01 feat(strategy): emit typed candidate lifecycle evidence
9ef6b4f fix(evidence): finalize typed lifecycle semantics
bb1ca13 feat(journal): persist typed population evidence in schema v6
```

Use `git show --stat` and targeted `git show <hash> -- <path>`. Explain actual
behavioral/schema changes. Documentation is not proof.

## Expected version and identity matrix

Verify every value from executable code, config and fixtures:

| Boundary | Expected identity |
|---|---|
| MEXC strategy spec | `mexc_strategy_v2` |
| canonical config | `config/mexc_strategy_v2.yaml` |
| StrategySpec contract hash | `9c62b88b7804e9663bae6f0eb429c58c541680b61d307c4f16032cb0b62fe3dd` |
| default StrategySpec instance hash | `9f0b2d7035c2a82ab1b6d8595245b8c3a7a8b9faad17bea8c57f6fcacb189466` |
| current population writer | schema v6 |
| default runtime path | `data/runtime/mexc_population_decisions_v6.jsonl` |
| historical journal | frozen schema v5, readable but read-only |
| cycle envelope | schema v3 |
| cycle identity | version v5 (unchanged by writer-v6 migration) |
| reversal features | `mexc_reversal_features_v2` |
| reversal feature contract hash | `20f9f61d4e2d787c5ad05f54ee3ccd8b7f8ea3a99fe09bc38bbefe09872c496c` |
| lifecycle contract | `candidate_lifecycle_v1` |
| lifecycle contract pin | `cc75c871b7097aa215f9ac88c736b6572e2443318cb0cf9f8bdaf1b0c8cc8551` |
| frame provenance contract | `mexc_closed_frame_provenance_v1` |
| frame provenance pin | `f4004ac933cc1725b2560e93ffbe278c826910424e350059ad29420ed3665dbf` |
| frame hash / bundle contracts | `mexc_closed_ohlcv_hash_v1` / `mexc_raw_frame_bundle_hash_v1` |
| single-position replay | schema v3 with mandatory `ReplayEvidence` |
| frozen strategy behavioral digest | `d5736beda70ca2826dc4868c2d4d95cb17b1289ac2ba03a2a052d9db69587459` |

The StrategySpec hash pins its declarative contract, not all Python bytes. The
instance hash identifies canonical YAML. Lifecycle/provenance pins similarly pin
declared canonical semantics, not external origin authenticity. Verify that the
frozen v5 fixture is byte/semantic evidence for compatibility and that v6 does
not rewrite or append it.

## Core implementation claims to verify

### Typed lifecycle

Verify from `trading/signals/lifecycle_contract.py`,
`core/signal_generator.py` and `trading/signals/layered_strategy.py`:

1. Arm, observation, proposal and terminal event identities are semantic and do
   not include processing wall clock.
2. `CandidateArmV1` binds arm OHLC, effective invalidation, confirmation policy,
   StrategySpec and raw-frame bundle identity.
3. SAME_BAR repeats the exact predecessor/arm market state and counters; later
   observations advance monotonically and exactly once per physical elapsed bar.
4. Invalidation/confirmation/expiry priority is deterministic; terminal events
   cannot acquire successors.
5. Proposal entry binds the correct arm or confirmed observation reference price;
   rejected/not-evaluated proposal states cannot masquerade as executable entry.
6. State mutation is transactional: an evidence/Layer5 failure does not partially
   advance pending lifecycle.
7. Public `evaluate_with_lifecycle(...)` owns exact base, benchmark and HTF
   `FrameRead`, revalidates each, recomputes base indicators from raw base bars and
   does not reread mutable benchmark/cache state.
8. Legacy `generate()` behavior and frozen digest remain compatible; typed API
   hardening did not silently choose new thresholds/timeframes.

### Exact closed-frame provenance

Verify from `trading/market_data/frame_provenance.py`, feed/cache and scanner:

1. Frame identity binds venue, symbol, canonical timeframe, cutoff, exact bar
   range/count, canonical OHLCV(+turnover availability) and hash.
2. Operational request latency is outside market-frame identity but preserved in
   source evidence; evidence is causal (`request_started_at <= received_at <=`
   decision/cycle receipt boundary).
3. `fresh`, `stale`, `no_rows`, `request_failed` and `not_requested` are distinct.
   Malformed HTTP/JSON/API payload is never laundered into empty/no-data.
4. A failed refresh may retain a hashed stale frame only as stale evidence. It
   cannot become current/fresh, and stale base data cannot produce an entry.
5. Base, benchmark and HTF evidence exist per symbol/read; raw bundle hash binds
   their latency-free market/frame identities, while request/receipt/cache facts
   remain in separately persisted operational evidence. Cycle `SourceTiming`
   entries for the
   benchmark/base/HTF bar sources are deterministic projections of that evidence,
   not second free-form claims. Universe and allowed contract-details timings are
   separate cycle-level provenance and are not `FrameRead` objects.
6. Terminal pre-evaluation/empty cycles encode benchmark `not_requested` and do
   not invent a market request.

### Journal schema v6 and reader

Verify from `trading/metrics/population_journal.py`,
`trading/metrics/cycle_envelope.py`, `app/scan.py` and
`ai/reversal/population_dataset.py`:

1. `SCHEMA_VERSION=5` is the frozen compatibility alias;
   `CURRENT_WRITE_SCHEMA=6`; `CYCLE_IDENTITY_VERSION=5` remains separate.
2. A new writer emits v6 only. Existing v5 opens read-only and cannot be appended
   or silently migrated. Legacy v5 model export needs explicit opt-in in addition
   to normal checkpoint policy.
3. V6 header pins evidence contract identities and benchmark evidence. Each row
   carries exact base/HTF evidence, raw bundle identity and optional lifecycle.
4. Entry action vocabulary is strict. An entry requires current data and a typed
   lifecycle/proposal consistent with row action; HOLD/error semantics fail closed.
5. Every row is exactly bound to envelope venue/symbol/cycle/spec/timing and its
   source receipt is no later than decision/cycle completion.
6. Benchmark/base/HTF bar-source timings are exactly re-derived from evidence;
   universe and allowed contract-details timings keep their separate validators.
   Unsupported extra sources or a terminal benchmark that was allegedly requested
   are rejected.
7. Input/snapshot IDs are re-derived rather than trusted. Canonical numeric and
   integer types cannot drift through `1`, `1.0`, bool or non-finite ambiguity.
8. Cross-cycle pending lifecycle chain is validated. A coherently rehashed forged
   predecessor/epoch still fails.
9. `contains_cycle()` refreshes under lock and scanner checks it before strategy
   mutation. `runtime_session()` is a non-blocking process/thread lifetime lock;
   a second scanner owner fails before its first market request and a crash/error
   releases the OS lock.
10. Strict reader validates the complete file before yielding; evidence remains
    top-level metadata and never enters the numeric feature whitelist.

## Exact trust boundary: journal v5 and v6

Do not call either journal generically “tamper-proof.” Verify and state exactly:

1. The unkeyed chain detects accidental corruption, torn/incomplete writes,
   splicing, partial edits and an earlier-cycle change with an unchanged successor.
2. Restart/stale-writer guards reject rollback, fork or rewrite relative to the
   writer's cached observed prefix.
3. A fresh unanchored reader can accept a different complete internally valid
   chain, including clean suffix deletion or a coordinated whole-file rewrite.
4. `JournalCheckpointReceipt` is unsigned. A copy beside the writable journal is
   not independently trusted. It becomes an anchor only when preserved or
   authenticated outside the journal writer's rewrite domain and later supplied
   explicitly.
5. A trusted receipt authenticates only its exact covered prefix (journal ID,
   sequence/tip, byte length and prefix SHA-256), not later tail cycles.
6. Default anchored reading yields the covered prefix. Explicit unanchored-tail
   inclusion may expose validated but unauthenticated later cycles.
7. `model_input_records()` requires a trusted external receipt by default;
   unanchored research needs explicit `allow_unanchored=True`. Frozen v5 needs
   explicit `allow_legacy_v5` as well.
8. Neither pins, chain nor test fixtures authenticate MEXC responses or establish
   trading edge. They enforce internal evidence contracts.

## Executable paths to inspect

- `config/mexc_strategy_v2.yaml`
- `core/mexc_strategy_spec.py`
- `core/signal_generator.py`
- `core/indicators.py`
- `core/volume_profile.py`
- `trading/signals/lifecycle_contract.py`
- `trading/signals/layered_strategy.py`
- `trading/signals/strategy_interface.py`
- `trading/market_data/frame_provenance.py`
- `trading/market_data/source_timing.py`
- `trading/market_data/bar_contract.py`
- `trading/market_data/feed.py`
- `trading/market_data/timeframe_cache.py`
- `trading/market_data/universe.py`
- `trading/metrics/cycle_envelope.py`
- `trading/metrics/population_journal.py`
- `ai/reversal/feature_contract.py`
- `ai/reversal/population_dataset.py`
- `backtesting/single_position.py`
- `app/scan.py`
- `ai/train.py` only to explain why it is legacy/excluded

Inspect at least:

- `tests/v2/test_signal_lifecycle_contract_v1.py`
- `tests/v2/test_strategy_lifecycle_integration_v1.py`
- `tests/v2/test_frame_provenance_v1.py`
- `tests/v2/test_population_journal_v6.py`
- `tests/v2/test_population_journal_v5_compatibility_fixture.py`
- `tests/fixtures/mexc_population_journal_v5_minimal.jsonl`
- `tests/v2/test_population_journal_chain_v5.py`
- `tests/v2/test_scan_v2.py`
- `tests/v2/test_scan_source_provenance_v2.py`
- `tests/v2/test_scan_journal_reader_e2e_v2.py`
- `tests/v2/test_population_feature_dataset_v2.py`
- `tests/v2/test_reversal_feature_contract_v2.py`
- `tests/v2/test_journal_cycle_records_v2.py`
- `tests/v2/test_mexc_strategy_runtime_integration_v2.py`
- `tests/v2/test_mexc_strategy_v2_compatibility_fixture.py`
- `tests/v2/test_indicator_golden_vectors_v2.py`
- `tests/v2/test_volume_profile_golden_vector_v2.py`
- `tests/v2/test_signal_logic_golden_vector_v2.py`
- `tests/v2/test_single_position_contract_v2.py`
- `tests/v2/test_single_position_schema_v3.py`
- `tests/v2/test_causal_cycle_identity_v2.py`

## Required adversarial audit

Confirm tests or independently reason through:

- lifecycle IDs change for every semantic arm/predecessor/price/policy change;
- SAME_BAR with changed OHLC/input/counter fails; backward or skipped physical bar
  transitions fail; invalidation beats same-bar confirmation;
- legacy pending state cannot fabricate a typed predecessor; a failed Layer5 or
  evidence check leaves pending state unchanged;
- public typed API rejects mismatched symbol/timeframe/cutoff/raw bundle and never
  falls back to mutable cached benchmark/HTF;
- reordered/non-finite/gapped bars, invalid high/low, malformed HTTP/JSON/API and
  fake empty data fail; stale refresh remains distinctly stale;
- v6 rejects forged `input_hash`/`snapshot_id`, arbitrary action, wrong venue or
  symbol, cycle-owned field drift, late evidence, source-timing drift, unsupported
  extra source timing, invalid terminal benchmark policy, noncanonical integer/
  numeric types and cross-cycle lifecycle substitution;
- entry without typed lifecycle/proposal, or entry on stale base evidence, fails;
- `contains_cycle()` observes an external append; second runtime owner is rejected;
- v5 fixture reads exactly, refuses append/migration and requires explicit legacy
  model-export opt-in;
- coordinated whole-file rewrite can remain internally consistent without an
  external receipt but is rejected when it changes an externally anchored prefix;
- evidence/checkpoint metadata is not part of numeric predictive features.

Passing tests prove only their specified invariants. They do not prove every
market path, external authenticity, exchange availability, execution latency,
profitability or model value.

## Remaining work to assess, in this order

Typed lifecycle and per-symbol base/benchmark/HTF provenance are **completed
claims to verify**, not the next implementation request.

1. Decide and test restart semantics: append an explicit right-censor before a
   new arm hypothesis or deterministically rehydrate pending state only from an
   externally anchored v6 prefix. Silent continuation is forbidden.
2. Separate persisted identities for `MarketFeatureSnapshot`, `RuleEvaluation`,
   `StrategyProposal` and `ShadowPrediction`; add point-in-time instrument spec
   (contract size, quantity step, minimums, leverage/rounding rules, timestamp,
   content hash).
3. Compute a complete gate-independent causal snapshot and raw-universe ledger
   before rule filtering. Until late-feature missingness no longer encodes the
   rule path, model training is forbidden.
4. Bind versioned proposal to executable single-position labels and exact costs:
   one entry, one SL, one TP, sizing, fees, spread, slippage, funding, horizon and
   concurrency one.
5. Only then collect prospective runtime population with external checkpoint/
   manifest publication and wait for labels to mature.
6. Build purged chronological evaluation with embargo and untouched test;
   rules/no-trade/matched-random/constant-logistic baselines precede LightGBM
   multiclass + separate conditional-payoff/EV head. CatBoost/XGBoost-AFT then
   sequence/forecast challengers only after evidence warrants them.

Separately assess future causal feature parity (1h RSI, overhead/Fibonacci,
weakness/confluence/liquidation proxy, OI notional and relative VP levels) and a
raw point-in-time contract ledger. Do not merge this with threshold calibration.
Any fast-pump timeframe is a separately versioned v3 hypothesis; frozen v2 is the
control.

LLM roles are unchanged: Kimi/OpenAI/Gemini-class models may only transform
timestamped public text into strict offline JSON context or assist audit/research.
They must never own numeric action, sizing, stop/TP, leverage, private credentials
or live execution.

## Safe secret/runtime metadata checks

Never open `.env` or historical blobs. Only metadata:

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

Report existence, current tracked state, reachable-history presence and confirmed
rotation separately. Ignored/untracked does not imply rotated.

## Only allowed executable validation

```powershell
& 'C:\Users\vlasy\PycharmProjects\koteika_Ultra\.venv\Scripts\python.exe' `
  -m pytest -q
```

Recorded full result at executable tip `bb1ca13`:

```text
723 passed, 4 skipped, 2 known PytestCollectionWarning (17.80s)
```

Re-run only this local synthetic suite if needed. Do not run scanner, bot, network
scripts, model training, exchange access or Telegram delivery.

## Required response

Return one read-only report before proposing any edit:

1. exact root/MEXC paths, worktrees, branch, HEAD, upstream and dirty state;
2. commit-by-commit explanation through `bb1ca13` and later docs-only commits;
3. causal data-flow map from sources → FrameRead/provenance → lifecycle/rules →
   journal/checkpoint → future identities/proposal/label/model/replay;
4. independently verified version/hash/pin/path matrix;
5. implemented vs partial vs planned, explicitly treating lifecycle/provenance/v6
   as completed claims and gate-independent snapshot as open;
6. adversarial lifecycle, stale-evidence and journal-v6/v5-compatibility analysis;
7. exact v5/v6 trust boundary, with no unqualified “tamper-proof” claim;
8. contradictions among docs, code and tests;
9. leakage/look-ahead/population/timing/execution/reproducibility risks ranked
   P0/P1/P2 with file/line evidence;
10. exact test evidence and what it cannot prove;
11. assessment of baseline → LightGBM+EV → challenger and restricted-LLM order;
12. corrected ordered plan with measurable acceptance criteria;
13. operational verdict: bot/secret/edge/model/network/live state and safest next
   code slice.

Do not implement fixes in this pass. Do not request permission to implement. The
purpose is a fresh independent audit of the committed v6 evidence boundary and a
precise next plan. Private execution/live remain a separate future project after
rotated credentials and reproducible prospective edge.
