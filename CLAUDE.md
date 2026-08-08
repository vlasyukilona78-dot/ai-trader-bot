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
   `claude/codex-project-review-04581e`; the 2026-08-08 hardening code tip is
   `e0e4cb4`, and current HEAD must descend from both anchors.

## Current product truth

- Target exchange: MEXC futures.
- Scope: public-data, signals-only causal research. There is no production MEXC
  private execution adapter in this line.
- Latest tested tree at the recorded checkpoint: `529 passed, 4 skipped`, with
  two known pytest collection warnings (`13.85s`).
- Generic pump-fade has not demonstrated stable positive edge after costs. The
  earlier DCA/positive-expectancy claims are retracted hypotheses, not evidence.
- Phase 0 and the Phase 1 timing/journal/replay hardening are complete. Journal
  schema v4 and single-position schema v3 are current; benchmark context now
  fails closed by default.
- The current runtime-population path is
  `data/runtime/mexc_population_decisions_v4.jsonl`; older journal schemas and
  legacy event-conditioned CSV are not accepted by the strict reader.
- Do not train or enable a model from the current partial snapshots. The trace
  is still gate-conditioned; StrategySpec/physical intervals, per-symbol
  base/HTF provenance, arm/confirm lifecycle, point-in-time instrument specs and
  the journal→proposal→label bridge remain unfinished.

## AI boundary

- First numeric champion candidate: small CPU LightGBM multiclass model plus a
  separate conditional-payoff/EV head.
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
