# ADR: MEXC v3 и путь к финальной версии бота

Статус: **ACCEPTED**

Дата: **2026-08-15**

Область: MEXC worktree, `mexc_strategy_v3`, research/signals/execution release path

Авторитетная спецификация: [FINAL_BOT_MASTER_PLAN_2026-08-14.md](FINAL_BOT_MASTER_PLAN_2026-08-14.md)

Этот ADR фиксирует архитектурное решение в компактной форме. Точные схемы,
инварианты, phase gates, определения метрик и acceptance criteria задаёт
авторитетный master plan. При расхождении этот ADR не расширяет и не ослабляет
его требования.

## Контекст

Замороженная линия `mexc_strategy_v2` на Min60/45-часовом окне не установила
generic pump-fade edge после издержек. Она остаётся честным control и
compatibility baseline, но не соответствует новой продуктовой цели: причинно
обнаруживать SHORT pump exhaustion на MEXC Futures и публиковать сигнал не позже
10 минут после наблюдаемого peak/retest attempt.

Пользователь торгует в MEXC Futures Cross. Выбранное на бирже плечо 100x является
настройкой initial margin, а не разрешением увеличить notional. Требуется единый
воспроизводимый путь от point-in-time public evidence до proposal, outcome,
portfolio replay и prospective verdict. Learned model не может подменять
causality, execution safety или risk governor.

## Решение

### 1. Сохранить frozen control без semantic drift

- `mexc_strategy_v2` остаётся frozen Min60/45-hour no-edge control.
- Journal v5 остаётся frozen read-only compatibility schema.
- Journal v6 остаётся текущим writer только frozen v2-линии; v3 semantics,
  features и events в него не добавляются.
- Существующие v2 StrategySpec, feature-contract, lifecycle, provenance и causal
  identity hashes остаются compatibility fixtures.
- Frozen single-position schema v3 не получает новые risk/outcome semantics.
  Они вводятся version-dispatched v4 либо отдельными versioned contracts с
  точным adapter к v3.
- Любой v2 hash/fixture drift является STOP.

### 2. Построить v3 как отдельную causal evidence line

V3 использует full-universe lightweight Min1 discovery и выполняет дорогие
Min5/15/60/4h context reads только после дешёвого causal trigger. Если
full-universe Min1 технически недостижим, заявленная universe policy должна быть
явно сужена либо v3 release остановлен; watchlist нельзя выдавать за всю биржу.

Peak identity разделяется на `peak_episode_id`, `peak_level_id` и
`peak_attempt_id`. Strictly higher tick создаёт новый level/attempt, equal-tick
retest — новый attempt того же level. Deadline каждого attempt:

```text
attempt_deadline_ts = attempt_bar_open_ts + 600 sec
```

OHLC Min1 не доказывает intrabar ordering, поэтому confirmation допускается
только более поздним закрытым Min1 bar, пока нет ordered trade-stream evidence.
Restart восстанавливает materialized state из append-only evidence и причинно
replay-ит пропущенные закрытые бары; gap ведёт к fail-closed состоянию.

### 3. Разделить v3 evidence на четыре ledger

1. `V3FastUniverseLedgerV1` хранит point-in-time raw contract population,
   inclusion/exclusion/not-observed reasons, каждый ожидаемый Min1 cohort и row,
   coverage, detector outcome, missed/overrun state и fast commit.
2. `V3ContextLedgerV1` хранит parent fast commit, point-in-time instrument rules,
   Min5/15/60/4h evidence, aggregation provenance, freshness и безусловный
   `FeatureSnapshotV3`.
3. `V3PeakProposalLedgerV1` хранит `FastCycleEnvelopeV1`, peak/retest/reset,
   confirmation, pre-alert veto/recheck, deterministic rule evaluation,
   proposal, notification evidence и точные parent bindings к fast/context
   ledgers.
4. `V3TradeOutcomeLedgerV1` хранит no-fill/lapse/invalidation, research или
   manual entry, exact costs, rounding, gap overshoot, TP/STOP/HORIZON outcome,
   MAE/MFE/time-to-event и manifests forward evidence.

Все четыре ledger используют canonical serialization, versioned schemas/events,
domain-separated hash chain, interprocess lock, fsync, idempotency,
duplicate/fork rejection, torn-tail policy, restart validation и detached
checkpoint receipt. Trusted admission evidence требует anchor вне writable
runtime trust-domain. Event-oriented peak rows допустимы только при наличии
полного minute denominator в fast-universe ledger.

### 4. Зафиксировать SLA как проверяемую цепочку receipts

Primary signal должен удовлетворять консервативному market-peak upper latency
не более 600 секунд. Формирование, решение, recheck, публикация и operator ACK —
разные clocks. Для выбранного actionable channel:

```text
actionable_ts <= actionable_delivery_at
provider channel:
  actionable_delivery_at = alert_provider_accepted_at
local channel:
  actionable_delivery_at = durable_local_publication_commit_ts
```

`actionable_delivery_at` нельзя выводить из `decision_completed_ts` или
`actionable_ts`. Pre-alert recheck может только veto/invalidate proposal и не
может повышать score, создавать сигнал или исправлять missing feature. Alert
после deadline получает `SLA_INELIGIBLE`. Provider acceptance, operator receipt
и operator ACK не смешиваются. Full-universe coverage и SLA имеют явные
denominators, включая HOLD, error и late/missed cohorts.

### 5. Зафиксировать deterministic user-risk envelope

```text
venue / direction              = MEXC Futures / SHORT pump exhaustion
margin mode                    = Cross
selected exchange leverage     = up to 100x when instrument permits
max effective account leverage = 1.0x
reference equity               = 100 USDT
planned account-loss budget    = <= 5% equity
preferred planned stop         = <= 4% coin move
absolute planned trigger cap   = <= 5% coin move
global executable concurrency  = 1
DCA / averaging                = prohibited
typical realization            = 1-30 min, sometimes longer
```

Position sizing учитывает exact contract/tick/quantity rules, fees, spread,
slippage, funding и stress gap overshoot. Planned 5% trigger не является
гарантией фактической потери ровно 5% при gap. Unknown instrument, account,
position, liquidation-buffer или rounding state означает abstain/STOP. Модель и
LLM не могут расширять deterministic risk/action scope.

### 6. Разделить конечные release outcomes

- **Final Research Monitor** — допустимый законченный результат при
  `NO_EDGE`/`INCONCLUSIVE`; только public evidence и явно non-actionable
  observations, без size/entry/SL/TP.
- **Final Actionable Signals Bot** — разрешён только после
  `EDGE_CONFIRMED_PROSPECTIVE` и `SIGNALS_OPERATION_VALIDATED`; ручной вход,
  deterministic Cross-risk card, local paper ledger и operator workflow, без
  private trade credential.
- **Optional Execution Bot** — отдельный будущий release после нового scope и
  явного разрешения: rotated segmented credentials, dedicated subaccount,
  reconciliation, venue-side protection, kill switches и staged canary/live.

Отсутствие edge не считается основанием публиковать actionable cards или
переходить к live. Model fit запрещён до frozen admissible population, labels и
preregistration; private execution не является частью принятия этого ADR.

### 7. Сохранить строгий release order

P1-P3 сначала доказывают strict history/aggregation, public-data quality и
full-universe acquisition feasibility. Contracts P4/P5 можно исследовать после
P0, но нельзя принять раньше P1-P3; ledger schema и `raw_contract_population` не
замораживаются до результата P3. Model fit начинается не раньше P7. Provider
SLA, operator ACK/manual workflow и private/account truth доказываются на своих
более поздних gates, а не выводятся из local research receipts.

## No-network и STOP boundary

Принятие, хранение или коммит этого ADR **не разрешает** network pilot, scanner,
bot, Telegram, model fit, private API, testnet/demo, live orders, чтение `.env`
или capital risk. Public pilot требует отдельного явного решения U5. Network,
Telegram, model fit, private API и любая торговля требуют раздельных разрешений.
Исторические credentials не ротированы, поэтому Telegram/private/testnet/live
остаются закрытыми.

Полный список STOP conditions задан master plan §20. Обязательные STOP включают:

- future data, hindsight peak/fill/ranking или point-in-time universe breach;
- incomplete/gapped data, превращённые в valid, либо gate-conditioned model
  missingness;
- v2 fixture/hash drift или незаякоренное trusted evidence;
- SLA, coverage, runtime ownership, restart или ledger-integrity failure;
- unknown instrument/account/position state либо sizing/risk breach;
- holdout-driven selection, insufficient power или результат, исчезающий после
  realistic costs/latency;
- secret exposure, unrotated credential use или unexpected network action;
- ambiguous order state, missing venue protection или reconciliation mismatch;
- попытку модели/LLM расширить deterministic scope;
- замену недоступного safe test environment прямым live без отдельного решения.

## Последствия

Положительные последствия:

- v2 no-edge evidence остаётся воспроизводимым control;
- SLA, universe denominator, proposals и outcomes становятся фальсифицируемыми;
- restart, concurrency-one и Cross sizing получают доказательный audit trail;
- отрицательный verdict приводит к безопасному законченному продукту, а не к
  принудительному торговому релизу;
- learned model остаётся challenger над честными baselines, а не источником
  неограниченного риска.

Стоимость решения:

- четыре независимых ledger, causal aggregation и restart replay увеличивают
  реализационную и операционную сложность;
- до P3 нельзя окончательно заморозить v3 population/ledger topology;
- actionable product может не состояться, если edge, SLA, coverage или safety
  gates не подтверждены;
- private/live путь требует отдельной security и operational программы.

## Open decision register

Этот ADR не закрывает U1-U15 и не превращает рекомендации в разрешения.

| ID | Решение | Текущая рекомендация / статус | Требуется до |
|---|---|---|---|
| U1 | Executable max holding | 4h `HORIZON_EXIT`, 24h shadow; user sign-off pending | P6 |
| U2 | Pre-alert source | Continuous public stream preferred; snapshot честно слабее | P3/P4 |
| U3 | Safety upper bounds | Заморозить user/economic caps до candidate performance | P7 |
| U4 | Manual fee schedule | Нужны фактические operator/account fees | P6/P11 |
| U5 | Public pilot network permission | Не предоставлено этим ADR/plan | P2 |
| U6 | Dedicated empty MEXC subaccount | Обязателен до P12B и private/live | P12B/P13 |
| U7 | Peak breach as exit | Пока safety failure, не exit | P6 |
| U8 | TP | Train/dev `{1R, 1.5R, 2R}`, baseline 1.5R | P7 |
| U9 | Final automation scope | Research Monitor гарантирован; Signals требует edge; Execution отдельно | P12/P13 |
| U10 | Manual-entry latency | Primary latency и sensitivity set не выбраны; 300 sec validity не latency | P6 |
| U11 | Versioned risk caps | Research: `min(5 USDT, 5% equity)`; daily/session/live pending | P6/P11/P15 |
| U12 | RPO/RTO/retention | Рекомендации master plan §16; утвердить operational ADR | P11/P13 |
| U13 | Attestation/liquidation buffer | Version, age, pct/ticks не выбраны; unknown means abstain | P6/P12B |
| U14 | Canary promotion evidence | Duration, lifecycle count и zero-incident criteria не выбраны | P15 |
| U15 | Full-universe Min1 fallback | Явно сузить universe либо остановить v3; критерий заморозить до P3 | P3 |

## Изменение решения

Изменение causal clocks, universe scope, peak identity, ledger topology,
deterministic risk envelope, release outcomes или STOP boundary требует нового
versioned ADR либо явной поправки с independent review. Нельзя переопределять
эти contracts конфигом, model artifact, runtime fallback или undocumented
operator action.
