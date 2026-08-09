# Claude Code entrypoint — Koteika Ultra MEXC

This worktree is the selected MEXC causal research/signals-only line. Do not
infer the current state from the root/Bybit branch or from old positive
expectancy notes.

## Mandatory first read

Before editing code, changing branches, starting a process, calling an exchange
API, or giving a strategy verdict:

1. Read `docs/CLAUDE_REVIEW_PROMPT_2026-08-03.md` in full and follow its
   read-only orientation/review sequence.
2. Read `docs/STRATEGY_AI_MASTER_PLAN_2026-08-03.md` in full. It is the
   authoritative MEXC strategy, data, model and implementation plan.
3. Read the current checkpoint and the final 2026-08-08 section of
   `docs/AI_HANDOFF.md`. Earlier sections are an audit history and contain
   superseded findings.
4. Verify the worktree, branch, local HEAD and remote HEAD yourself. The
   published AI foundation anchor is `f0b43d6` on
   `claude/codex-project-review-04581e`; the 2026-08-08 StrategySpec/journal-v5
   code tip is `2d0efcb`, the behavioral-compatibility tip is `258c35f`, and
   current HEAD must descend from all three anchors.

## Current product truth

- Target exchange: MEXC futures.
- Scope: public-data, signals-only causal research. There is no production MEXC
  private execution adapter in this line.
- Latest executable tip and test receipt: `258c35f`; `580 passed, 4 skipped`,
  with two known pytest collection warnings (`14.99s`).
- Generic pump-fade has not demonstrated stable positive edge after costs. The
  earlier DCA/positive-expectancy claims are retracted hypotheses, not evidence.
- Phase 0 plus the Phase 1 timing, replay, canonical StrategySpec and evidence
  hardening are complete. Population journal schema v5, CycleEnvelope v3 and
  single-position schema v3 are current; benchmark context fails closed.
- Focused review found no open P0/P1 in the implemented StrategySpec/runtime
  scope and no P0/P1/P2 in the journal/checkpoint scope. Its one StrategySpec
  P2 was the lack of numeric behavioral anchors behind declarative revisions;
  `258c35f` closes it with tests and a fixture only, without changing runtime
  algorithms, thresholds, spec version or hashes.
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
- Do not implement a future `mexc_strategy_v3` as a naive global version-literal
  bump. Persisted evidence needs a version-dispatched reader that retains the
  frozen v2 parser/hashes, while v3 types, config and evidence live in a separate
  version namespace. The frozen v2 fixture must remain green.
- For a faster strategy, separate event/state horizons from estimator warm-up
  and sample budgets. `pump_window` and confirmation timing are not the same kind
  of parameter as RSI/ATR/EMA/VP history. A Min15 design is a new strategy
  hypothesis, not a mechanical conversion or override of the frozen Min60 v2.
  Historical prose coupling “45 bars” to a roughly 20-minute pump is only a clue
  about earlier intent, not an accepted timeframe decision.
- The current runtime-population path is
  `data/runtime/mexc_population_decisions_v5.jsonl`; older journal schemas and
  legacy event-conditioned CSV are not accepted by the strict reader.
- Journal v5 is internally chained and rejects corruption, partial edits,
  incomplete writes and stale-writer rewrites relative to the prefix that writer
  already observed. A fresh reader cannot distinguish a clean shorter prefix or
  a coherent rewrite of a wholly unanchored file: only an explicitly supplied
  receipt kept outside the writer's trust domain anchors a prefix and detects
  rollback within it. Model-input export requires that receipt unless an unsafe
  unanchored override is stated explicitly.
- Do not train or enable a model from the current partial snapshots. The trace
  is still gate-conditioned; intended fast physical windows, per-symbol base/HTF
  provenance, arm/confirm lifecycle, point-in-time instrument specs and the
  journal→proposal→label bridge remain unfinished.

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
- Do not use private APIs, Telegram delivery, testnet or live execution.
- `.env` is absent from the current tip, but secrets existed in Git history and
  have not been rotated. Treat all historical credentials as compromised and do
  not read, print, copy, test or transmit them.
- Never use destructive Git/filesystem commands or modify the root/Bybit
  worktree without an explicit transfer plan.

## After material work

Update both `docs/STRATEGY_AI_MASTER_PLAN_2026-08-03.md` and the current section
of `docs/AI_HANDOFF.md` with the branch/HEAD, files changed, exact tests,
runtime/network actions, unresolved findings and next acceptance gate. Keep
historical conclusions intact; append a superseding verdict instead of silently
rewriting history.
