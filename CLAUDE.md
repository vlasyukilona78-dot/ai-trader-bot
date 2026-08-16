# Claude Code entrypoint — Koteika Ultra MEXC

This worktree is the selected MEXC causal research/signals-only line. Do not
infer the current state from the root/Bybit branch or from old positive
expectancy notes.

## Mandatory orientation and reading order

Before editing code, changing branches, starting a process, calling an exchange
API, or giving a strategy verdict:

1. Read `docs/FINAL_BOT_MASTER_PLAN_2026-08-14.md` in full. It is the
   authoritative final-product roadmap after the independent 2026-08-15
   `APPROVE_AS_AUTHORITATIVE` closure verdict.
2. Read `docs/ADR_MEXC_V3_FINAL_BOT_2026-08-15.md` in full. It is the bounded
   v3 architecture/decision contract subordinate to that master.
3. Read `docs/MEXC_V3_PREREGISTRATION_SKELETON_2026-08-15.md` in full before
   collecting admission data, selecting thresholds or evaluating a model.
4. Read the current notice and latest 2026-08-15 checkpoint of
   `docs/AI_HANDOFF.md`. Earlier sections are audit history and may contain
   superseded implementation order or findings.
5. Treat `docs/STRATEGY_AI_MASTER_PLAN_2026-08-03.md` as the frozen
   historical/executable-v2 audit source. Read it before touching or judging a
   v2 contract, codec, fixture or compatibility boundary; do not use it as a
   competing future-product roadmap.
6. `docs/CLAUDE_REVIEW_PROMPT_2026-08-03.md` remains a historical independent
   read-only audit procedure for the v6 evidence checkpoint, not the current
   implementation prompt. Where it conflicts with the newer roadmap/ADR, the
   newer explicitly approved decision governs future v3 work while v2 remains
   frozen.
7. Verify the worktree, branch, local HEAD and remote HEAD yourself. The
   published AI foundation anchor is `f0b43d6` on
   `claude/codex-project-review-04581e`; the 2026-08-08 StrategySpec/journal-v5
   code tip is `2d0efcb`, the behavioral-compatibility tip is `258c35f`, the
   versioned-evidence compatibility tip is `1971b77`, the finalized typed
   lifecycle tip is `9ef6b4f`, and the current journal-v6 code tip is
   `bb1ca13`. The independently approved roadmap was reviewed against the later
   research-builder HEAD `ad30b02` and published at `2a14299`. The completed
   deterministic Min1 aggregation and strict-history contract tips are
   `0ff1b3a` and `36e1446`. The bounded pre-pilot transport and restart-safe
   strict-history-v2 tips are `ba8ea00` and `f8a6b5b`. Current HEAD must descend
   from all of these anchors; discover any later documentation-only publication
   descendant rather than assuming its hash.

## Current product truth

- Target exchange: MEXC futures.
- Scope: public-data, signals-only causal research. There is no production MEXC
  private execution adapter in this line.
- `docs/FINAL_BOT_MASTER_PLAN_2026-08-14.md` is the authoritative path to the
  final Research Monitor, conditionally promoted Actionable Signals Bot and
  separately gated optional Execution Bot. Its v3 design is documentation, not
  deployed runtime or proof of edge; no public-data pilot is implied.
- Frozen-v2 journal/runtime tip and test receipt: `bb1ca13`; `723 passed, 4
  skipped`, with two known pytest collection warnings (`17.80s`). The later
  `ad30b02` dataset-builder commit is parallel research mechanics and does not
  make its Min60/calibration outputs admissible v3/model evidence.
- P1 data-contract checkpoint: `0ff1b3a` adds deterministic Min1 aggregation
  and `36e1446` adds strict, receipt-backed history collection. Independent
  review found no P0/P1. Validation at that code tip is `822 passed, 4 skipped`,
  with the same two known pytest collection warnings (`22.94s`). No network or
  public-data pilot was run.
- Pre-pilot hardening checkpoint: `ba8ea00` adds the bounded injected-executor
  Futures transport contract and `f8a6b5b` adds restart-safe strict-history v2.
  Final validation is `905 passed, 5 skipped`, with the same two known pytest
  collection warnings; code-scope independent red-teams found no P0/P1/P2. No
  network or public-data pilot was run.
- Generic pump-fade has not demonstrated stable positive edge after costs. The
  earlier DCA/positive-expectancy claims are retracted hypotheses, not evidence.
- Phase 0 plus the Phase 1 timing, replay, canonical StrategySpec, typed
  lifecycle and exact closed-frame evidence foundations are complete. The
  current population writer is schema v6; CycleEnvelope v3, causal cycle
  identity v5 and single-position schema v3 remain current. Journal v5 is a
  frozen read-only compatibility format; benchmark context fails closed.
- The current reversal feature contract is `mexc_reversal_features_v2`, pinned
  to `20f9f61d4e2d787c5ad05f54ee3ccd8b7f8ea3a99fe09bc38bbefe09872c496c`.
- Focused review found no open P0/P1 in the implemented StrategySpec/runtime
  scope and no P0/P1/P2 in the journal/checkpoint scope. Its one StrategySpec
  P2 was the lack of numeric behavioral anchors behind declarative revisions;
  `258c35f` closed it. Independent red-team review of the later versioned
  evidence checkpoint `1971b77` found no P0/P1/P2 in that change scope.
- The MEXC runtime loads one strict strategy-spec YAML; its canonical committed
  default is `config/mexc_strategy_v2.yaml`, while `--strategy-spec` may select
  another fully validated file. The pinned contract hash is `9c62b88b...e3dd`;
  the committed default instance hash is `9f0b2d70...9466`. Base/benchmark are
  Min60 and HTF is Hour4. Existing windows deliberately retain fixed-bar
  semantics, so the current 45-bar pump window means 45 hours; choosing a faster
  physical horizon is a later strategy change.
- Behavioral locks now pin cumulative VWAP/OBV/candle-CVD numbers, exact volume
  profile levels, and the default armed-HOLD to confirmed-SHORT decision,
  trace and proposal. A committed CycleEnvelope-v3 fixture also proves that
  frozen `mexc_strategy_v2` evidence remains readable.
- Persisted evidence now uses a version-dispatched reader with a frozen v2
  parser, contract hash and instance-hash path. A future `mexc_strategy_v3`
  must add a separate registered parser/types/config/evidence namespace rather
  than altering the v2 codec or regenerating its fixture.
- For a faster strategy, separate event/state horizons from estimator warm-up
  and sample budgets. `pump_window` and confirmation timing are not the same kind
  of parameter as RSI/ATR/EMA/VP history. A Min15 design is a new strategy
  hypothesis, not a mechanical conversion or override of the frozen Min60 v2.
  The approved final-product roadmap and v3 ADR define the bounded Min1
  acquisition/peak-SLA research path. Its strict local history, aggregation,
  bounded transport and restart-safe per-shard storage contracts are now
  implemented. The pinned `api.mexc.com` fixture remains explicitly
  `candidate_not_u5_verified`; there is no real/default network executor, pilot
  run manifest, global acquisition budget, full-universe scheduler or v3
  strategy/runtime. None may inherit evidence from the frozen Min60 line.
  Historical prose
  coupling “45 bars” to a roughly 20-minute pump remains only a clue about
  earlier intent.
- The current runtime-population path is
  `data/runtime/mexc_population_decisions_v6.jsonl`. Schema v5 fixtures remain
  strictly readable but v5 files cannot be appended; model export from v5 also
  requires an explicit legacy opt-in. Earlier journal schemas and legacy
  event-conditioned CSV are not accepted by the strict reader. Every journal
  file is homogeneous in exact StrategySpec version, contract hash and instance
  hash; restart, append and strict dataset reading reject mixtures.
- Journal v6 binds the exact base, benchmark and higher-timeframe `FrameRead`
  evidence, the canonical hash of their latency-free market/frame identities,
  bar-source timing projections and any
  typed arm/confirmation/proposal lifecycle event. Stale, empty, failed and
  not-requested sources remain explicit evidence outcomes; they are not silently
  promoted to current data. The frame-provenance contract pin is
  `f4004ac933cc1725b2560e93ffbe278c826910424e350059ad29420ed3665dbf`;
  the lifecycle contract pin is
  `cc75c871b7097aa215f9ac88c736b6572e2443318cb0cf9f8bdaf1b0c8cc8551`.
- The scanner now holds one process/thread lifetime lock for the v6 journal and
  one process-local whole-sweep strategy lock. A competing scanner fails before
  its first market request, rather than mutating pending lifecycle state and
  losing a later duplicate-append race.
- Journal v5/v6 chaining rejects corruption, partial edits,
  incomplete writes and stale-writer rewrites relative to the prefix that writer
  already observed. A fresh reader cannot distinguish a clean shorter prefix or
  a coherent rewrite of a wholly unanchored file: only an explicitly supplied
  receipt kept outside the writer's trust domain anchors a prefix and detects
  rollback within it. Model-input export requires that receipt unless an unsafe
  unanchored override is stated explicitly. Strategy identity, exact source
  evidence, raw-bundle identity and lifecycle evidence remain top-level
  non-predictive metadata outside the numeric `features` mapping.
- Do not train or enable a model from the current snapshots. The trace and
  missingness are still gate-conditioned, so current rows are not an admissible
  training population. Intended fast physical windows, point-in-time instrument
  specs and the proposal-label bridge remain unfinished. Scanner restart
  validates the historical lifecycle chain but still does not rehydrate an
  in-memory pending candidate; explicit restart right-censor/rehydration
  semantics remain an open acceptance gate.

## AI boundary

- First numeric champion candidate: small CPU LightGBM multiclass model plus a
  separate conditional-payoff/EV head.
- A proposal-conditioned outcome/payoff target retains proposal geometry such
  as stop/target distances and costs. A pure direction/reversal target is a
  separate experiment with a separate target and feature set that excludes
  proposal-derived fields; the two heads must not be described as one task.
- CatBoost is a tabular challenger; XGBoost AFT is an auxiliary time-to-event
  model; TCN/Chronos are later challengers.
- Kimi/OpenAI/Gemini-class LLMs may only extract timestamped public text context
  into a strict schema. They do not calculate price, risk, sizing, entry, stop,
  TP or order actions.
- Model choice and third-party tooling are open for evidence-based comparison.
  Causality, provenance, deterministic risk and untouched-test discipline are
  correctness requirements, not removable research restrictions.

## Safety and secrets

- Keep the scanner/bot stopped unless the user explicitly requests a run after
  reviewing its network and alert effects.
- The pre-pilot checkpoint contains no real/default network executor. Its
  versioned endpoint fixture is a candidate, not proof of current official or
  live endpoint validity. Public MEXC Min1 collection remains blocked on an
  immutable run manifest, aggregate/global budgets and orchestration, a pinned
  first-request endpoint-verification procedure, a deliberately supplied
  executor, and separate U5 permission. After U5, that bounded verification
  probe must succeed before any acquisition request. Strict-history-v2 storage
  is one exact range request/shard per root; it is not a full-universe run store.
- Do not use private APIs, Telegram delivery, testnet or live execution.
- `.env` is absent from the current tip, but secrets existed in Git history and
  have not been rotated. Treat all historical credentials as compromised and do
  not read, print, copy, test or transmit them.
- Never use destructive Git/filesystem commands or modify the root/Bybit
  worktree without an explicit transfer plan.

## After material work

Update `docs/FINAL_BOT_MASTER_PLAN_2026-08-14.md` when an approved product
decision or acceptance state changes, and append the current section of
`docs/AI_HANDOFF.md` with branch/HEAD, files changed, exact tests,
runtime/network actions, unresolved findings and the next acceptance gate.
Keep `docs/STRATEGY_AI_MASTER_PLAN_2026-08-03.md` frozen except for an explicit
historical-fact correction or compatibility annotation. Preserve historical
conclusions; append a superseding verdict instead of silently rewriting history.
