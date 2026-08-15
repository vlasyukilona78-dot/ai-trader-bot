# Claude Code — independent review of the final-bot master plan

Status: **CONSUMED**. Initial review returned `BLOCK_FOR_CHANGES`; after the four
closure fixes the 2026-08-15 re-check returned `APPROVE_AS_AUTHORITATIVE` with
remaining P0/P1 none. This file is retained as the exact review protocol/receipt,
not as an instruction to rerun review automatically.

Работай только в:

`C:\Users\vlasy\PycharmProjects\koteika_Ultra\.claude\worktrees\codex-project-review-04581e`

Главный объект review:

`docs/FINAL_BOT_MASTER_PLAN_2026-08-14.md`

Это совместный proposed draft Codex + Claude. Не считай его authoritative до
завершения этого review.

## Режим

Выполни независимый **read-only** аудит. Ничего не редактируй, не коммить, не
push, не запускай scanner/bot/model/Telegram/private API и не используй сеть.
Не читай содержимое `.env`. Разрешены только read-only Git, `rg`, `Get-Content`
и чтение локального кода/tests/docs. Pytest в этом проходе не нужен: проверяется
план, а не новая реализация.

Сначала зафиксируй:

```powershell
git branch --show-current
git rev-parse HEAD
git status --short --branch
git worktree list --porcelain
git log -5 --oneline --decorate
```

Ожидаемая base: branch `claude/codex-project-review-04581e`, HEAD `ad30b02`,
local == upstream. Сам plan и этот review prompt могут быть untracked. Другие
worktrees не трогай.

## Обязательные источники истины

Прочитай полностью либо релевантные current/final sections:

1. `CLAUDE.md`
2. `docs/STRATEGY_AI_MASTER_PLAN_2026-08-03.md`
3. top checkpoint и последний authoritative section в `docs/AI_HANDOFF.md`
4. `docs/CLAUDE_REVIEW_PROMPT_2026-08-03.md`
5. фактический код/тесты на HEAD, когда plan делает executable claim
6. новый `docs/FINAL_BOT_MASTER_PLAN_2026-08-14.md` целиком

Не воскрешай withdrawn expectancy/DCA claims. `ad30b02` оценивай только как
legacy/discovery parallel-builder mechanics: его permissive Min60/48h labels не
являются v3/model evidence.

## Зафиксированный product intent

Проверь, что plan нигде не искажает следующее:

```text
venue                         = MEXC Futures
direction                     = SHORT pump exhaustion
market-peak alert upper SLA   <= 600 sec
typical realization           = 1–30 min, иногда дольше
priority                      = минимизировать новый high/adverse loss,
                                а не искусственно закрыть всё через 30 min
margin mode                   = Cross
selected exchange leverage    = 100x if instrument permits
effective account exposure    <= 1.0x equity
reference equity              = 100 USDT
planned account loss budget   <= 5 USDT / 5% equity including stress costs
preferred stop distance       <= 4%
absolute planned trigger cap  <= 5%
global concurrent positions   = 1
DCA                           = prohibited
```

100x не должно трактоваться как 100x account exposure или ликвидация всего
депозита при каждом 1% движении. Одновременно Cross account truth нельзя
гарантировать без dedicated account/attestation либо private reconciliation.

## Review axes

### A. Causality и SLA

- one `fast_cycle_as_of_ts`, per-source cutoffs/receipts;
- conservative peak interval and `provider_accepted - peak_bar_open <= 600`;
- no same-bar OHLC ordering claim;
- full-universe Min1 denominator, not watchlist-only hidden scope;
- episode/level/attempt identity, retest/new-high deadline reset;
- pre-alert named public source and honest provider/ACK clocks;
- manual-entry latency separate from 300-sec validity;
- deterministic restart replay/censor and no wall-clock reset.

### B. Version/storage topology

- frozen StrategySpec v2, feature v2, lifecycle v1, frame provenance v1,
  CycleEnvelope v3, journal v5/v6 and single-position v3 remain readable and
  semantically unchanged;
- v6 stays frozen v2-control;
- FastUniverse, Context, PeakProposal and TradeOutcome ledgers have complete
  denominator, parent commit/trust/freshness binding, canonical chain/lock/fsync/
  restart/checkpoint semantics;
- v3 feature/spec/readers and `single_position_v4` or equivalent new contracts
  are explicit; no reuse of old version names for new semantics.

### C. Features, labels и research

- expensive MTF fetches may be admission-triggered, but features are
  unconditional inside the frozen causal formation population;
- no gate-conditioned missingness or outcome-conditioned dataset;
- unsigned proposal geometry precedes proposal-conditioned EV prediction;
- deterministic fill after preregistered manual latency, no best price in window;
- TP/STOP/HORIZON_EXIT and 4h/24h intervals are executable and distinct;
- common-calendar portfolio, matched-random definitions, purge/embargo, final
  holdout and survivor-bias restrictions are complete;
- verdict taxonomy distinguishes invalid/inconclusive/no-edge/unsafe/confirmed;
- LLM never controls action/risk and is outside SLA-critical path.

### D. Cross/manual operations

- structural and 5% stressed loss include fees/spread/slippage/gap/funding;
- notional/equity <=1 after contract/quantity rounding;
- liquidation/risk tier unknown => abstain;
- dedicated empty subaccount before real-money manual entry;
- durable atomic global slot, authenticated ACK, restart and interprocess races;
- manual out-of-band entry produces a risk-breach state, not a silent warning.

### E. Security/release/final bot

- Research Monitor, Actionable Signals Bot and Optional Execution Bot are not
  conflated;
- all historical credentials rotate before Telegram/private/test/live;
- scanner/LLM cannot access trade credential;
- no runtime dependency on `D:`; off-device backup is DR, not critical-path disk;
- runtime modes, lifetime scanner/account locks, kill-switch actions and rollback
  with open position are executable;
- official API test environment is not silently replaced by live;
- stage/daily caps, allowlists, mandatory scenario matrix and promotion receipts;
- no actionable release without prospective edge and safety/operations gates.

## Required output

Верни строго:

1. Git/worktree facts.
2. `P0`, `P1`, `P2` findings с точными `file:line` и конкретным failure mode.
3. Plan↔code, plan↔historical-plan и internal-plan contradictions.
4. Минимальный exact wording/diff для каждого настоящего finding.
5. Decisions, которые действительно требуют пользователя; не превращай уже
   принятый intent в повторные вопросы.
6. Verdict:
   - `APPROVE_AS_AUTHORITATIVE`, либо
   - `BLOCK_FOR_CHANGES` с минимальным blocking set.
7. Явно подтверди: files unchanged, no commit/push/network/runtime/.env access.

Не предлагай начинать implementation, collection или model fit в этом review.
Если P0/P1 нет, так и напиши; не придумывай замечания ради количества.
