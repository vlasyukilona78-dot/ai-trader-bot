# Claude Code collaboration

This repository has two intentionally separate worktrees. Git remote preserves
the current functional commits; an older verified external receipt and optional
same-disk checkpoints preserve ignored datasets/runtime state. None of these
layers alone is a complete copy of every later local artifact.

## Mandatory first read

Before any edit, command that changes state, runtime start, branch switch, or
strategy conclusion:

1. Read `AGENTS.md`.
2. Read `docs/CURRENT_STATE_AND_AI_PLAN_2026-08-03.md` in full. It is the
   current authoritative operational and AI plan.
3. Read `docs/PROJECT_HANDOFF_FOR_CLAUDE.md` for the detailed historical system
   map. Use `docs/AI_HANDOFF.md` only as a historical audit log. It contains
   superseded positive DCA/expectancy findings and must not be treated as the
   current verdict.
4. Run read-only Git orientation (`status`, `worktree list`, branch and HEAD)
   and state explicitly which worktree is being inspected.

## Critical repository state

- The root worktree is the preserved Bybit/runtime line:
  `feat/phase2-layer1-pump-runtime-alignment`; functional anchor `2f7e18f` is
  published to the same remote branch. Fresh validation: 533 passed, 4 skipped.
- `.claude/worktrees/codex-project-review-04581e` is a separate, unmerged
  MEXC/research line; functional anchor `98217df` is published to
  `origin/claude/codex-project-review-04581e`. Fresh validation: 340 passed,
  4 skipped.
- A verified external receipt exists under
  `D:\koteika-preservation\koteika_preservation_20260803_135615_6d24c806`, but
  it captures older source heads `0c38863`/`9f71a86`, not the newer functional
  anchors. Local ignored datasets still require deliberate preservation.
- `scripts/preservation/create_verified_backup.ps1` supports same-disk
  `LocalCheckpoint` without USB and separate-disk/UNC `DisasterResilient` mode.
- The verified current same-disk receipt is
  `C:\koteika-checkpoints\koteika_preservation_20260803_154657_88710bc3`
  (`CHECKPOINT_VERIFIED.json`, root `80e6f2b`, MEXC `1e91ce0`). It is not a
  substitute for separate-disk disaster recovery.
- Do not run `reset`, `clean`, `checkout --`, bulk deletion, or repository-wide
  formatting. Do not assume recovery directories or relocated environments are
  disposable.
- Never edit both worktrees in one task without an explicit transfer plan.

## Product truth

- Koteika Ultra finds possible exhaustion after sharp crypto pumps and
  produces primarily SHORT-oriented WATCH, SETUP, ENTRY, and EXIT signals.
- MEXC is the explicitly selected target research exchange. The MEXC line is a
  public-data, signals-only implementation and has no production private
  execution adapter. Root/Bybit is a preserved runtime line, not the selected
  target venue.
- Positive expectancy is not established. The most recent honest MEXC replay
  found no edge for the tested single-position generic pump-fade after costs.
  Do not revive retracted DCA, `+0.0208 expectancy`, profitable-portfolio, or
  safe-leverage claims.
- Causal scanner/population journaling and an executable one-entry/one-stop/
  one-TP research contract are implemented. The next model path is LightGBM on
  full runtime-population outcomes; causal TCN/Chronos are later challengers.
  LLMs are timestamped context extractors only, never trading decision makers.

## Runtime and trading safety

- Keep the bot stopped unless the user explicitly asks to start it.
- The last verified local snapshot was `BOT_RUNTIME_MODE=paper`, `dry_run=True`.
  Treat this as historical state and re-check config before any future start.
- Do not run even a one-cycle scan automatically: it may call external APIs
  and send alerts.
- Paper mode is not a broker-grade portfolio simulator.
- Do not enable live execution. `LIVE_STARTUP_MAX_NOTIONAL_USDT` is validated
  and logged but is not enforced as an actual per-order cap.
- Before any future live work require explicit user approval, one authoritative
  risk configuration, hard monetary caps, credential validation, emergency
  stop behavior, and fresh end-to-end testnet evidence.
- Never enable high leverage merely because the user generally allowed live
  trading.

## Secrets and evidence

- Treat `.env`, exchange keys, Telegram/Discord credentials, proxy credentials,
  private endpoints, runtime databases, and local datasets as sensitive.
  Never print or copy secrets into reports.
- `.env` is absent from current tips but exists in old Git history/refs.
  Rotation has explicitly not been performed yet. Historical credentials are
  potentially compromised; private API, Telegram, testnet and live remain
  blocked until rotation and fresh validation.
- Tests prove code consistency, not profitability or live safety.
- Distinguish code facts, measured evidence, historical notes, and inference.

## After material work

Update `docs/CURRENT_STATE_AND_AI_PLAN_2026-08-03.md` and, when material,
`docs/PROJECT_HANDOFF_FOR_CLAUDE.md` with:

- date, worktree, branch, and HEAD;
- files changed and why;
- tests/evidence and exact result;
- bot runtime state (`stopped`, `paper`, `demo`, `testnet`, or `live`);
- unresolved findings;
- the safest recommended next step.

If working in the MEXC worktree, also update that worktree's own handoff. Do
not silently rewrite the historical `docs/AI_HANDOFF.md` into a new verdict.
