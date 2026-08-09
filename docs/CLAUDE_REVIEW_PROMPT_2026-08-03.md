# Claude Code re-entry and independent read-only review prompt

This prompt is intentionally self-contained. It is for a fresh Claude Code
account reviewing the preserved root/Bybit line and the selected MEXC causal
research line after the canonical StrategySpec, journal-v5 and frozen-v2
behavioral-semantics commits.

The task is an independent audit, not implementation. Do not change files,
stage, commit, push, switch branches, pull, reset, clean, launch the bot or
scanner, contact any external service, or inspect secret contents.

## Non-negotiable project status

- MEXC is the selected target exchange.
- The root/Bybit and MEXC lines are intentionally unmerged.
- The bot and scanner are stopped. MEXC private execution does not exist in the
  selected line.
- No stable generic pump-fade edge has been established after costs.
- No model has been fitted, promoted or enabled in the causal admission path.
- Current work is causal research and signals-only. Testnet, private API and
  live execution remain forbidden.
- Current ignored `.env` files may exist locally. Historical credentials were
  committed in older history and have not been confirmed rotated. Never read or
  print current or historical secret values.
- Accept a repeat no-edge result. Do not tune thresholds, windows or models in
  this review.

## Exact repository orientation

```text
Project root / preserved Bybit-runtime line:
C:\Users\vlasy\PycharmProjects\koteika_Ultra
branch: feat/phase2-layer1-pump-runtime-alignment
verified root tip: f01591f2e2872d5af6341580b6d8d44b298cd244

Selected MEXC causal research line:
C:\Users\vlasy\PycharmProjects\koteika_Ultra\.claude\worktrees\codex-project-review-04581e
branch: claude/codex-project-review-04581e
AI foundation anchor: f0b43d6
earlier Phase 1 hardening tip: e0e4cb4
canonical StrategySpec commit: bebfd0d
journal-v5 executable commit: 2d0efcb
latest executable/test-contract tip: 258c35f
remote branch: origin/claude/codex-project-review-04581e
```

The handoff documents may be a later documentation-only descendant of
`258c35f`. Discover the exact current HEAD from Git. Treat `258c35f` as the
latest executable/test-contract tip, not necessarily as the final documentation
HEAD.

Other worktrees may exist. Inventory them, report them, and do not modify them.
Do not transfer MEXC code into root/Bybit without a separate explicit plan.

## Mandatory read-only Git orientation

Run these commands from PowerShell:

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
git -C $MEXC log -12 --oneline --decorate
git -C $MEXC diff --stat 1e91ce0..HEAD
git -C $MEXC diff --check 1e91ce0..HEAD
git -C $MEXC merge-base --is-ancestor f0b43d6 HEAD
git -C $MEXC merge-base --is-ancestor e0e4cb4 HEAD
git -C $MEXC merge-base --is-ancestor bebfd0d HEAD
git -C $MEXC merge-base --is-ancestor 2d0efcb HEAD
git -C $MEXC merge-base --is-ancestor 258c35f HEAD
git -C $MEXC ls-files -- .env
```

Expected facts at the published checkpoint:

- the current MEXC HEAD descends from all five listed anchors;
- `bebfd0d` and `2d0efcb` are consecutive executable commits;
- `258c35f` follows them and adds frozen v2 behavioral and compatibility
  evidence without changing thresholds or runtime strategy logic;
- after publication, local and upstream should match and the MEXC worktree
  should be clean;
- root and root upstream match at `f01591f`; the three root `.idea/*` changes
  are user-owned noise and must remain untouched;
- current `.env` is not tracked in either selected tip, but older reachable
  history contained it;
- no Python bot/scanner process should be running;
- no positive edge or live-readiness claim should appear.

If reality differs, report the exact discrepancy before interpreting code. Do
not repair, discard or conceal it.

## Required reading order

Read every listed file in full unless a section is named explicitly:

1. Root `AGENTS.md` and root `CLAUDE.md`.
2. MEXC `CLAUDE.md`.
3. MEXC `docs/STRATEGY_AI_MASTER_PLAN_2026-08-03.md` as the authoritative
   strategy/AI plan.
4. MEXC `docs/AI_HANDOFF.md`, especially:
   - `Canonical StrategySpec and anchored journal v5 - 2026-08-08`;
   - `Phase 1 evidence hardening - 2026-08-08`;
   - `Current authoritative checkpoint - 2026-08-03`;
   - `Claude no-edge finding - 2026-07-26`;
   - `Codex independent review of Claude's nine follow-up commits`;
   - `Unified strategy/AI contract foundation - 2026-08-03`.
5. Root `docs/CURRENT_STATE_AND_AI_PLAN_2026-08-03.md` and
   `docs/PROJECT_HANDOFF_FOR_CLAUDE.md` for preservation and historical system
   context. Where they disagree with the MEXC master plan, the MEXC plan wins.
6. The executable code and tests listed below.

## Commits to inspect individually

```text
0b010e8 fix(backtest): prevent hindsight cohort substitution
3ff8de0 feat(ai): add causal MEXC feature contract
29536f1 docs: unify MEXC strategy and AI roadmap
f0b43d6 docs: record AI foundation publication state
32e8fbe fix(journal): harden causal population evidence
0c32047 fix(backtest): bind replay outcomes to schema-v3 evidence
e0e4cb4 fix(strategy): fail closed without benchmark context
bebfd0d feat(strategy): define canonical MEXC strategy spec
2d0efcb feat(journal): chain schema-v5 population evidence
258c35f test(strategy): pin v2 behavioral semantics
```

Use `git show --stat <hash>` and targeted `git show <hash> -- <path>`. Explain
what each commit actually changes. Documentation is not proof of correctness.

## Current version and identity matrix

Verify every value from code and the committed YAML:

| Boundary | Expected identity |
|---|---|
| MEXC strategy specification | `mexc_strategy_v2` |
| canonical MEXC source | `config/mexc_strategy_v2.yaml` |
| StrategySpec contract hash | `9c62b88b7804e9663bae6f0eb429c58c541680b61d307c4f16032cb0b62fe3dd` |
| committed default instance hash | `9f0b2d7035c2a82ab1b6d8595245b8c3a7a8b9faad17bea8c57f6fcacb189466` |
| population journal | schema v5, `data/runtime/mexc_population_decisions_v5.jsonl` |
| cycle envelope | schema v3 |
| reversal feature contract | `mexc_reversal_features_v2` |
| single-position replay | schema v3 with mandatory ReplayEvidence |
| layered strategy v1 behavioral digest | `d5736beda70ca2826dc4868c2d4d95cb17b1289ac2ba03a2a052d9db69587459` |
| frozen v2 envelope canonical payload SHA-256 | `87e3f049ca356f9cd7654464a6fa0cbb12ee319979e145de8aa021c858ee0e5e` |

The StrategySpec contract hash pins the declarative field/layout/adapter
contract. It is not a hash of all Python implementation bytes. The instance
hash identifies the complete canonical YAML payload.

The legacy root `config/config.yaml` is not the MEXC source of truth and contains
different Bybit-era values. Confirm that the MEXC scanner loads the dedicated
YAML once and treats legacy CLI timeframe/candle arguments as assertions, not
unhashed runtime overrides.

## StrategySpec facts that must be verified

Confirm from code and tests:

1. `MexcStrategySpec` uses strict exact-key parsing, canonical interval aliases,
   pinned contract identity and a deterministic canonical instance hash.
2. Strategy, volatility context, base/benchmark requests, HTF cache, indicator
   parameters, volume profile and both history gates are built from the same
   resolved specification.
3. `CycleEnvelope` stores the full canonical payload plus contract/instance
   hashes, rebuilds it strictly, and rejects a timeframe inconsistent with the
   spec.
4. Unknown, missing, duplicate or mistyped YAML values fail closed.
5. Every declared numeric indicator field executes on both base and HTF paths;
   history and volume-profile parameters are live rather than decorative.
6. Still-unwired `min_rsi_1h` and `require_confluence` values are accepted only
   at their inert defaults and reject attempted activation.
7. The committed default preserves previous scanner behavior. Commit `258c35f`
   pins the cumulative VWAP/OBV/CVD modes, volume-profile levels and one stateful
   arm-to-confirm decision/proposal trace, including the behavioral digest
   above. Explain what those finite golden vectors prove and what they cannot
   prove about every possible market path.
8. The frozen `tests/fixtures/mexc_strategy_v2_cycle_envelope_v3.json` rebuilds
   with the pinned v2 contract and instance hashes. Confirm that the fixture is
   canonical and readable; do not infer that a future global version bump will
   remain compatible without a version-dispatched parser.

The present executable timeframe semantics are explicit:

- base and BTC benchmark: Min60, 320 closed bars;
- HTF: Hour4, 120 closed bars, 1800-second request TTL;
- windows remain fixed counts of their source bars.

Therefore the current 45-bar pump window is 45 hours, confirmation wait is 3
hours, recent MSB is 6 hours, relative-strength lookback is 24 hours, and the
12-bar HTF structural anchor is 48 hours. These durations preserve existing
behavior; they do not establish the intended fast-pump horizon. No Min15 or
execution-timeframe feed was invented by this refactor.

Do not treat all `*_bars` fields as interchangeable duration fields. Separate
event horizons (pump, confirmation, recent structure and relative-strength
lookbacks) from estimator sample counts (RSI/EMA/ATR/bands/ADX), volume-profile
sample requirements, warm-up/data budgets and fixed-HTF anchors. Equal elapsed
seconds on Min15 and Min60 do not imply equal features or decisions because the
sampled paths differ.

## Journal-v5 and checkpoint threat boundary

Inspect and explain the precise boundary, without calling the journal generally
"tamper-proof" or "tamper-evident" without qualification.

Expected implementation:

- each new journal receives a random 256-bit `journal_id`;
- cycles use contiguous `sequence_no`, a domain-separated genesis,
  `prev_cycle_commit`, and a footer `cycle_commit` over the canonical header,
  every complete ordered decision row, and footer core;
- restart validates the entire chain and population membership;
- cooperating writers share process-local and OS sidecar locks;
- a stale writer may adopt only an exact extension of its observed prefix and
  rejects rollback, fork or rewrite relative to that cached state;
- `PopulationJournal.checkpoint_receipt()` returns an unsigned detached receipt
  containing journal/cycle identity, sequence, cycle commit, exact prefix byte
  length and prefix SHA-256;
- `verify_population_journal()` exposes whether the file is only internally
  consistent or externally anchored through a supplied receipt;
- the reader validates the complete file first, then checks every second-pass
  cycle ID/commit against the first pass before yielding;
- file stability includes device, inode, size, mtime and ctime.

The security statement must remain exact:

1. The unkeyed v5 chain detects accidental corruption, partial edits, torn or
   incomplete writes, splicing and a changed earlier cycle with an unchanged
   successor.
2. A fresh unanchored reader accepts any complete internally valid prefix. Clean
   suffix deletion/rollback is detectable only relative to a stale writer's
   cached prefix or a trusted receipt that covered the removed cycles.
3. An actor able to coherently rebuild an entire unanchored file can produce a
   different but internally consistent chain.
4. A receipt stored beside the writable journal is not independently trusted.
   It becomes an anchor only after being preserved/authenticated outside the
   journal writer's trust domain and explicitly supplied later.
5. A trusted earlier receipt detects a rewrite of its covered prefix. It does
   not authenticate later tail cycles.
6. With a receipt and default reader behavior, only the anchored prefix is
   yielded. `anchored_only=False` explicitly includes the validated but
   unanchored tail.
7. Without a receipt, the neutral generic reader provides integrity-only access.
   Explicit `anchored_only=True` must fail with
   `trusted_checkpoint_required_for_anchored_read`.
8. `model_input_records()` requires a trusted receipt by default. Unanchored
   model input is available only through the explicit research override
   `allow_unanchored=True`.

## Executable paths that must be inspected

- `config/mexc_strategy_v2.yaml`
- `core/mexc_strategy_spec.py`
- `core/signal_generator.py`
- `core/indicators.py`
- `core/volume_profile.py`
- `trading/signals/layered_strategy.py`
- `trading/signals/strategy_interface.py`
- `trading/market_data/universe.py`
- `trading/market_data/bar_contract.py`
- `trading/market_data/feed.py`
- `trading/market_data/mexc_client.py`
- `trading/market_data/timeframe_cache.py`
- `trading/metrics/cycle_envelope.py`
- `trading/metrics/population_journal.py`
- `ai/reversal/feature_contract.py`
- `ai/reversal/population_dataset.py`
- `backtesting/single_position.py`
- `app/scan.py`
- `trading/alerts/telegram.py`
- `ai/train.py` only to explain why it is legacy and excluded

Inspect at least these tests:

- `tests/v2/test_indicator_golden_vectors_v2.py`
- `tests/v2/test_volume_profile_golden_vector_v2.py`
- `tests/v2/test_signal_logic_golden_vector_v2.py`
- `tests/v2/test_mexc_strategy_v2_compatibility_fixture.py`
- `tests/fixtures/mexc_strategy_v2_cycle_envelope_v3.json`
- `tests/v2/test_mexc_strategy_spec_v2.py`
- `tests/v2/test_mexc_strategy_runtime_integration_v2.py`
- `tests/v2/test_population_journal_chain_v5.py`
- `tests/v2/test_reversal_feature_contract_v2.py`
- `tests/v2/test_population_feature_dataset_v2.py`
- `tests/v2/test_journal_cycle_records_v2.py`
- `tests/v2/test_population_journal_v2.py`
- `tests/v2/test_scan_journal_reader_e2e_v2.py`
- `tests/v2/test_scan_source_provenance_v2.py`
- `tests/v2/test_causal_cycle_identity_v2.py`
- `tests/v2/test_single_position_contract_v2.py`
- `tests/v2/test_single_position_schema_v3.py`
- `tests/v2/test_single_position_result_invariants_v2.py`
- `tests/v2/test_layered_strategy_thread_safety_v2.py`
- `tests/v2/test_market_context_gate_v2.py`
- `tests/v2/test_timeframe_cache_v2.py`
- `tests/v2/test_closed_bar_contract_v2.py`
- `tests/v2/test_turnover_units_v2.py`
- `tests/v2/test_telegram_alerter_v2.py`
- `tests/test_strategy_dry_run.py`

## Causal-path claims to prove from code

1. The single-position selector ranks the causal cohort before future fill data,
   never retrospectively promotes a filled runner-up, groups by `cycle_id`, and
   rejects outcomes whose mandatory ReplayEvidence does not rebuild exactly.
2. The closed-bar contract excludes forming candles and records source timing.
3. The feature contract is versioned, has a pinned schema hash, separates
   MODEL/PROPOSAL/POLICY/CONTEXT/DIAGNOSTIC roles, and distinguishes observed
   zero from missing/unavailable.
4. Every population member, including HOLD/error states, gets the same captured
   feature schema.
5. Frozen-universe funding reaches `StrategyContext`; live ticker data is not
   reintroduced into the closed-bar decision.
6. Strategy/universe hashes and snapshot/cycle/input/envelope/market-feature
   identities are rebuilt rather than trusted.
7. Empty-universe and pre-evaluation failures leave durable zero-row cycles.
8. The strict reader rejects malformed, incomplete, reordered, mixed-schema or
   drifting cycles before an export can partially consume them.
9. Model input exposes only the model whitelist and obeys the checkpoint rules
   above.
10. Legacy event-conditioned CSV and `ai/train.py` cannot be mistaken for the
    new population admission path.

## Required adversarial checks

Reproduce or inspect tests for all of the following without modifying tracked
files:

- StrategySpec missing/unknown/duplicate keys, type confusion, unsupported
  intervals, contract-hash drift and instance-hash drift;
- exact default configuration parity for SignalConfig, history gates, benchmark
  interval and HTF cache, plus numerical golden vectors for cumulative
  indicators and volume profile and the pinned stateful strategy behavior digest;
- for the one frozen default arm/confirm fixture, stable trace, action, stop/TP
  proposal and causal diagnostics are covered by explicit assertions plus a
  digest after 12-decimal float normalization; wall-clock-only identity remains
  excluded. Explain why this locks a representative path, not every strategy
  branch or sub-quantization numerical change;
- the frozen v2 cycle-envelope fixture must retain its canonical payload digest,
  rebuild its exact v2 contract/instance hashes and remain readable after any
  future spec version is introduced;
- rehashed outcome substitution, forged costs/sizing/timing, false replay-input
  hashes and evidence from a different bar;
- malformed rows, full-row/provenance edits, incomplete tails, duplicate cycles,
  wrong ordering and concurrent stale writers;
- changing and rehashing an earlier v5 cycle while leaving its successor
  unchanged;
- a coordinated whole-file rewrite that remains internally consistent but is
  rejected by an earlier trusted external receipt;
- forged receipt journal ID, cycle commit, exact byte length or prefix SHA-256;
- a same-size/same-mtime file replacement between validation passes: no changed
  row may be yielded before rejection;
- explicit anchored reading without a receipt must fail;
- model input without a receipt must fail unless the explicit unsafe research
  override is supplied.

Do not weaken a finding because the current tests pass. State exactly what each
test proves and what it cannot prove.

## Open work and decisions, not completed claims

Rank any additional findings P0/P1/P2. At minimum assess these remaining items:

1. Choose and version the intended physical fast-pump windows. Do not mix that
   research decision with mechanical StrategySpec plumbing or threshold tuning.
   First decide whether the frozen Min60/45-hour hypothesis remains the active
   v2 strategy or a distinct v3 hypothesis is required. A physical-duration
   template resolved for Min60 and Min15 would produce two canonical spec
   instances; it is not literally one unchanged executable YAML because
   `base_interval` belongs to spec identity. Declare field-by-field units,
   divisibility/rounding and bar-boundary rules. Equal duration across different
   sampling intervals is not behavioral parity and needs a new causal evaluation.
   Before changing the global current version, add version-dispatched parsing
   and prove the committed v2 envelope fixture remains readable; the current
   `MexcStrategySpec`/`CycleEnvelope` validation accepts only the current version.
2. Add per-symbol base and per-symbol/per-timeframe HTF provenance. Current
   cycle timing is aggregated and delivery latency is not execution proof.
3. Define typed arm-time, confirmation-time and proposal-time lifecycle without
   overwriting earlier state.
4. Build a point-in-time ledger of every MEXC USDT contract with explicit
   inclusion/exclusion reasons, not only the filtered scan universe.
5. Compute the complete causal feature snapshot before rule gates so missingness
   cannot encode the rule path.
6. Implement causal Fibonacci/overhead, weakness, confluence, 1h RSI and
   liquidation context in the runtime decision path rather than merely declaring
   them offline/planned.
7. Add point-in-time instrument specifications: contract size, quantity step,
   minimums, leverage rules, source timestamp and content hash.
8. Normalize OI to notional and price-relative POC/VAH/VAL distances; raw
   cross-symbol levels remain diagnostic.
9. Separate durable identities for MarketFeatureSnapshot, RuleEvaluation,
   TradeProposal, OutcomeLabel and ShadowPrediction.
10. Connect population cycles to executable single-position labels with exact
    fees, spread, slippage, funding, stop/TP geometry and concurrency.
11. Define a real external checkpoint publication/retention workflow. The
    detached receipt API alone does not create an external trust domain.
12. Build forward-data manifests and only then purged chronological evaluation,
    embargo, untouched test data, symbol/time-clustered uncertainty and matched
    random/rules/no-trade baselines.
13. Keep legacy event-conditioned datasets, old calibration claims and
    `ai/train.py` outside the causal admission path.

## Model and tooling architecture to challenge

The current proposed order is:

1. constant/logistic, rules, matched random and no-trade baselines;
2. small LightGBM multiclass candidate plus a separate conditional-payoff/EV
   head after prospective labels mature;
3. CatBoost challenger and XGBoost AFT auxiliary time-to-event model;
4. TCN/Chronos only after tabular evidence;
5. Kimi/OpenAI/Gemini-class LLMs only for timestamped public-text extraction to
   strict JSON, never as the numeric trading, sizing or risk engine.

Tooling is staged: Parquet + DuckDB first, local MLflow second, bounded Optuna
only inside train/validation, and Evidently/ONNX/DVC only after an observed need.
Challenge this order only with evidence from this data shape and causal contract,
not model popularity.

Do not conflate a proposal-independent direction target with the planned
`tp_first|sl_first|timeout` outcome. Those outcome classes are defined relative
to a particular stop, target and horizon; if proposal geometry varies, it is a
causal conditioning input for the outcome classifier as well as the EV head.
Excluding `FeatureRole.PROPOSAL` is correct only for a separately specified pure
direction head with a proposal-independent label. Audit the current combined
`model_feature_specs()` whitelist, but do not silently change its estimand.
Require each future head to pin an ordered feature list/schema hash, target,
scoring instant and allowed roles; treat CONTEXT explicitly and always exclude
POLICY/DIAGNOSTIC from numeric prediction inputs unless a separately versioned
contract proves otherwise.

## Safe secrets and runtime metadata checks

These commands inspect only existence, index/history metadata and process names.
Never open `.env`, print it, or show historical blob contents.

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

Report separately whether `.env` exists, is tracked now, existed in reachable
history, and has been rotated. Do not infer rotation merely because it is now
ignored/untracked.

## Only allowed executable validation

The only approved Python execution is the local synthetic test suite from the
MEXC worktree:

```powershell
& 'C:\Users\vlasy\PycharmProjects\koteika_Ultra\.venv\Scripts\python.exe' `
  -m pytest -q
```

Recorded result at executable/test-contract tip `258c35f`:

```text
580 passed, 4 skipped, 2 known PytestCollectionWarning (14.99s)
```

The focused StrategySpec/runtime review found no remaining P0/P1. The focused
journal/checkpoint red-team found no remaining P0/P1/P2 after the two-pass
replacement and anchored-reader fixes. Re-audit those conclusions; do not repeat
them merely because they are written here.

Do not run the scanner, bot, ad-hoc network scripts or tests requiring private
credentials, exchange access or Telegram delivery.

## Required response format

Return one read-only review before proposing any edits:

1. exact root/MEXC worktree, branch, HEAD, upstream and dirty-state facts;
2. commit-by-commit explanation through executable/test-contract tip `258c35f`, plus any later
   documentation-only descendants;
3. a causal data-flow map from MEXC sources through StrategySpec, universe,
   closed bars, features, rules, population journal/checkpoint, proposal, label,
   model and single-position replay;
4. the version/hash matrix independently verified from code;
5. what is implemented, partially implemented and only planned;
6. an adversarial analysis of StrategySpec parity and journal-v5 trust boundary;
7. contradictions among documentation, implementation and tests;
8. leakage, look-ahead, population-selection, timing, execution and
   reproducibility risks ranked P0/P1/P2 with file/line evidence;
9. exact test evidence and the limits of that evidence;
10. an assessment of the baseline/LightGBM+EV/challenger/LLM architecture;
11. a corrected next-step plan with measurable acceptance criteria;
12. a final operational verdict covering bot state, secret status, edge status,
    model status and the safest next code decision.

Do not make changes during this pass. The immediate research decision is whether
to keep the frozen Min60/45-hour v2 hypothesis or define a separately versioned
physical-duration hypothesis with explicit field units. Version dispatch and
continued v2 evidence readability are prerequisites to v3, not outcomes to
assume. Typed lifecycle, per-source provenance and unconditional market-feature
plumbing may continue without silently choosing new horizons. Private execution
and live trading remain a separate future project and require a reproducible
edge plus rotated credentials.
