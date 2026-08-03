# Koteika Ultra — единая стратегия, AI-архитектура и новый план

Актуально: **2026-08-03, Europe/Moscow**. Целевая площадка: **MEXC futures**.

Этот документ заменяет идею «добавить одну умную нейросеть, которая почувствует
разворот» на проверяемую систему. Он объединяет исполняемую стратегию, causal
feature contract, single-position PnL, роли моделей, research-инструменты и
правила допуска в shadow. До доказанного edge система остаётся signals-only.

## 1. Итоговое решение

Ядро первой AI-версии — **не Kimi K3 и не другая LLM**. Первым numeric champion
будет небольшой **LightGBM**, который на одном causal snapshot оценивает:

- `p_tp_first` — вероятность, что один TP будет достигнут раньше одного SL;
- `p_sl_first`;
- `p_timeout`;
- ожидаемый net return после fee, spread, slippage и funding — отдельной
  conditional-payoff/EV головой, а не автоматически из class probabilities;
- uncertainty / причину abstain. Здесь uncertainty — это schema/staleness/OOD
  abstain, predictive entropy и dispersion между folds/seeds, а не магическое
  свойство одного дерева.

LightGBM сравнивается с logistic baseline, frozen rules и matched-random. CatBoost
идёт первым challenger; XGBoost AFT — отдельный auxiliary time-to-exit experiment.
TCN и Chronos допускаются только после накопления достаточного prospective
population. LLM разрешена только как асинхронный компилятор публичного контекста.

Финальное действие всегда вычисляет детерминированная policy:

```text
closed point-in-time inputs
    -> unconditional causal features
    -> rules decision + trade proposal
    -> optional numeric shadow probabilities
    -> net-EV policy / abstain
    -> one-entry / one-stop / one-TP contract
```

Ни одна модель не получает право самостоятельно задавать side, leverage, size,
stop, TP, отправлять ордер или читать credentials.

## 2. Что означает «снять ограничения»

Снимаются искусственные блокировки разработки:

- внешний диск больше не требуется для ежедневной работы: локальный verified
  checkpoint уже поддержан штатным инструментом. Он защищает от ошибок Git и
  worktree, но не от физического отказа единственного SSD;
- Claude не является зависимостью проекта;
- исследовательский dataset больше не ограничивается только прошедшими gates или
  только отправленными сигналами;
- нет привязки к одному AI-provider: numeric model локальна, LLM provider сменный;
- старый `auto -> XGBoost`, event-conditioned CSV и tracked pickle не считаются
  обязательным путём;
- публичные MEXC данные, offline replay, локальные тесты и model research можно
  развивать без приватных ключей.

Не снимаются свойства корректной системы: closed-bar causality, data-quality
validation, explicit missingness, single-position risk contract, immutable test,
manual promotion и отсутствие private/live доступа до ротации ключей. Это не
бюрократические ограничения, а определение того, что результат вообще правдив.

## 3. Подтверждённые разрывы текущего кода

### P0 — время и исполнимость

- Scanner по умолчанию использует `Min60`. Поэтому `pump_window_bars=45` — это 45
  часов, `confirmation_max_wait_bars=3` — до трёх часов, `msb_recent_bars=6` —
  шесть часов. Комментарии описывают быстрый intraday pump, то есть физический
  смысл и исполняемая конфигурация расходятся.
- Каждый symbol получает собственный wall-clock `decision_ts` после вычисления.
  Single-position replay требует первую execution candle ровно с этим timestamp,
  что практически невозможно.
- Одновременные candidates сейчас группируются по равенству float timestamp, а
  не по `cycle_id`, поэтому worker latency может менять concurrency-one результат.
- Нужны отдельные `bar_cutoff_ts`, source-specific `request_started_at` /
  `received_at`, `decision_completed_ts`, `cycle_completed_ts`, `actionable_ts`
  и `entry_eligible_ts >= max(all source received_at, cycle completion)`.
- Look-ahead с заменой unfilled top-score на runner-up того же момента уже
  устранён: сначала выбирается лидер по causal score, затем моделируется только
  его fill. Grouping по cycle/cohort всё ещё требует Phase 1.

### P0 — dataset / labels / model

- Population journal писал полный цикл, но до текущей реализации ни один dataset
  builder его не читал.
- Старые builders создают event-conditioned LONG/DCA labels; новый
  `replay_single_short()` ими не вызывается.
- Старый trainer использует `target_win/target_horizon`, row-wise split, выбирает
  XGBoost первым и fit-ит isotonic calibrator на test block.
- MEXC inference к scanner не подключён. Старые pickle без manifest/hash не
  являются допустимыми artifacts и не должны загружаться.

### P0 — feature parity

- Funding и OI уже находились в universe metadata, но funding не передавался в
  strategy context; OI не употреблялся.
- `min_rsi_1h` и `require_confluence` объявлены, но не исполняются.
- Fibonacci, confluence, divergence и estimated liquidation доступны offline,
  но не в runtime decision.
- Weakness layer реализован, но выключен; candle-signed CVD является proxy, а не
  настоящим aggressor trade delta.
- Старый flattener запрашивает несуществующие имена (`level_dist_pct`,
  `msb_confirmed`, `sentiment_index`, `vwap_dist_pct`).
- Trace прекращается после первого failed gate. Отсутствие позднего feature
  означает «слой не вычислялся», а не измеренный ноль.
- Confirmation смешивает armed features с более поздним Layer 5 и не хранит
  L1b/L1c как typed lifecycle. Нужны связанные `CandidateArmSnapshot` и
  `ConfirmationSnapshot`, а момент model scoring должен быть выбран заранее.

### P1 — конфигурация и provenance

- `config/config.yaml` не задаёт фактические defaults signals-only scanner:
  scanner создаёт `LayeredPumpStrategy()` напрямую.
- MEXC OI `holdVol` ещё не имеет закреплённых units, history/delta и age.
- Contract details по умолчанию не загружаются, поэтому отсутствуют точные
  quantity step, minimum quantity, contract size и надёжный max leverage.
- Estimated liquidation map — OHLCV/leverage proxy. Его запрещено называть
  биржевым liquidation feed и смешивать с liquidation price нашей позиции.
- Empty-universe и ошибки до `append_cycle()` пока не имеют отдельного attempt /
  universe-envelope journal.
- Комментарии `measured best`, `validated` и исторические численные thresholds
  считаются только frozen legacy hypotheses: финальный no-edge отменил статус
  доказанного преимущества.

## 4. Реализованный фундамент этого этапа

Добавлен `ai/reversal/feature_contract.py`:

- version `mexc_reversal_features_v1`;
- stable registry и SHA-256 contract hash;
- роли `model_candidate`, `proposal_conditioning`, `deterministic_policy`,
  `context_candidate`, `diagnostic_only`;
- статусы `wired`, `conditional`, `offline_only`, `source_missing`, `planned`;
- для missing сохраняются `null`, observed bit и availability reason;
- executable schema hash отделён от полного registry/roadmap hash и закреплён
  literal-тестом: изменение captured semantics требует новой version;
- model whitelist исключает raw OI, POC/VAH/VAL, policy и diagnostic columns;
- action, confidence, labels и future returns исключены.

Scanner теперь:

- передаёт funding из того же frozen universe snapshot в strategy;
- пишет `feature_snapshot` каждому population record, включая HOLD/error rows;
- различает настоящий `funding_rate=0` и отсутствие источника;
- пишет provenance bits Layer 4 для sentiment/funding/long-short ratio и VP;
- фиксирует strategy-config и universe-policy hashes;
- больше не позволяет подменить unfilled leader runner-up того же cohort с
  использованием будущего результата.

Добавлен строгий `ai/reversal/population_dataset.py`:

- читает только полный ordered population cycle;
- требует точное совпадение contract version/hash/schema, пересчитывает feature
  snapshot из source metadata и повторно проверяет PopulationDecision timing,
  status, bar duration, input/snapshot/cycle IDs;
- сохраняет все HOLD rows;
- не принимает старые CSV;
- нормализует любое `observed=0` в `None`, даже если legacy trace содержит
  placeholder `0.0`.
- имеет отдельный nested `model_input_records()` только по role whitelist, без
  action/status/rule columns.

Это ещё не model-ready dataset: поздние layer features пока structural-missing,
labels и executable entry timing не построены.

## 5. Единая таблица стратегии

| Блок | Текущий runtime | Contract | Что исправить до model fit |
|---|---|---|---|
| Universe | весь **уже отфильтрованный** turnover-band cycle journalled | conditional population boundary | ledger всех USDT contracts: included/exclusion reason, received-at, policy hash, instrument rules |
| Input | один closed cutoff, Min60/320 | hard causal invariant | физические timeframe/окна, hashes base/HTF/BTC |
| L1 pump | recent band event + RSI/volume + move/retrace | candidate features + frozen rule baseline | unconditional values; окна в секундах |
| L1b quality | ATR floor + exact quote turnover; legacy fallback существует | features; data quality отдельно | fallback `close×contract volume` помечать missing, interval-normalized turnover, no early truncation |
| L1c market | BTC RS, 4h RSI, optional overhead/chase | numeric context | explicit missing, fixed 1h/4h sources, fib/confluence ablation |
| L2 weakness | OBV/CVD proxy, default off | challenger features | always measure; tag proxy; collect trades before true flow claim |
| L3 location | extreme distance, MSB, VP | numeric location | unconditional snapshot; armed vs confirmed timestamps |
| L4 crowd | neutral sentiment, VWAP, funding optional | context features | real funding wired; LSR missing; provenance/age required |
| OI/depth/flow | raw `holdVol` только diagnostic | planned | contract-size/notional normalization + age/history; timestamped depth/trades collector |
| Liquidation | offline estimated proxy | optional context | explicit `estimated_ohlcv_proxy`; ablation only |
| L5 proposal | structural stop, POC/1.6R TP; partial TP лишь legacy diagnostic | executable v1: exactly one TP | frozen proposal/contract hash and proposal for all eligible rows |
| Confirmation | up to 3 base bars | separate state transition | `armed_as_of`, `confirmed_as_of`, `actionable_ts` |
| Execution | no MEXC private path | one SHORT, one SL, one TP, concurrency 1 | journal→EntryPlan→label bridge and entry alignment |

### Четыре явные популяции

1. `raw_contract_population`: каждый MEXC USDT contract до policy filters с
   `included`/`exclusion_reason`. Иначе universe-selection bias невидим.
2. `scan_universe_population`: каждый выбранный symbol point-in-time cycle,
   включая no-data/error. Это то, что journal покрывает сейчас.
3. `feature_valid_population`: строки с полным hard input contract. Все causal
   features вычисляются независимо от rule gates.
4. `proposal_eligible_population`: объективно возможно построить SHORT proposal
   `stop > entry > TP` и применить instrument/sizing contract. Именно здесь
   numeric model ранжирует candidates. Rules-qualified — отдельный флаг, а не
   фильтр dataset.

## 6. Модельная стратегия

| Модель | Роль | Сильная сторона | Основной риск | Решение |
|---|---|---|---|---|
| Logistic / constant prior | sanity baseline | прозрачность, выявляет ложную сложность | underfit | обязательна |
| LightGBM multiclass + EV head | первый champion candidate | tabular, missing values, CPU, быстрые ablations | class probabilities не описывают timeout payoff; overfit | **первый fit после prospective maturity** |
| CatBoost | первый challenger | categories и ordered boosting | symbol memorization/unseen listings | после LGBM, с/без symbol ID и unseen-symbol cohorts |
| XGBoost classifier | parity benchmark | зрелая экосистема | старый `auto` скрывал выбор | только explicit experiment |
| XGBoost AFT | time-to-exit auxiliary | censored duration | не решает competing risks сам | отдельная ablation |
| Causal TCN | sequence challenger | локальная динамика формы пампа | data hunger/leakage через windows | только после prospective sample |
| Chronos-2 / Bolt | forecast/embedding challenger | pretrained time-series prior | pretraining provenance и domain mismatch | offline/prospective challenger |
| TabPFN | small-data experiment | быстрый tabular prior | GPU/license/auth и ограничения sample | не champion; optional licensed test |
| LLM context | классификатор ingestion-объектов | новости/listing/incident context | hallucination, latency, look-ahead | provider benchmark, затем последняя ablation |

### Почему не Kimi K3 в ядре

Kimi K3 — большая reasoning/knowledge модель с длинным контекстом. Она подходит
для offline review, кода и разбора текстов, но свечи/turnover/funding — числовая
tabular/time-series задача. Даже официальный troubleshooting Kimi отдельно
предупреждает об ошибках арифметики и рекомендует calculator/tool use. Поэтому:

- Kimi K3 может быть дорогим offline adjudicator;
- Kimi K2.6 non-thinking или текущая low-latency модель другого provider может
  компилировать новости в строгий JSON;
- Kimi K2.6, актуальная low-latency GPT и Gemini сравниваются на одном pinned
  наборе по strict-schema accuracy, правильному abstain, latency и cost;
- provider выбирается этим benchmark, а не названием;
- LLM никогда не считает PnL и не принимает торговое решение.

Допустимый LLM output:

```json
{
  "event_type": "listing|unlock|exploit|campaign|none",
  "direction": "pump_supportive|reversal_supportive|neutral",
  "severity": 0.0,
  "novelty": 0.0,
  "confidence": 0.0,
  "abstain": true
}
```

`published_at`, `first_seen_at`, `retrieved_at`, `source_count`, source identity
и `expires_at` задаёт ingestion/policy слой из фактических метаданных. LLM не
имеет права сочинять эти поля.

Запрещённые поля: action, side, entry, stop, TP, leverage, size, balance,
credentials, private account state.

## 7. Сторонние инструменты

Добавлять только по мере возникновения задачи:

- **Parquet + DuckDB** — versioned datasets, быстрые cohort/time queries без
  отдельного сервера;
- **MLflow local** — параметры, code/data/contract hashes, metrics, artifacts и
  evaluation receipts;
- **Optuna** — только bounded pre-registered search на train/validation; test
  недоступен objective;
- **Evidently** — data/calibration drift после начала shadow;
- **ONNX Runtime** — только после выбора устойчивой модели, если native inference
  окажется узким местом;
- **DVC** — позднее для больших immutable data artifacts; не заменяет verified
  backup и не требует внешнего диска для текущей разработки.

Нельзя одновременно внедрять весь MLOps-стек: сначала Parquet/DuckDB и manifest,
затем MLflow; остальные инструменты должны оправдать собственную сложность.

## 8. Целевая схема данных

### `MarketFeatureSnapshotV2`

```text
snapshot_id, cycle_id, ordinal, size, venue, symbol
bar_cutoff_ts, per-source request_started_at/received_at/exchange_ts
cycle_completed_ts, actionable_ts, entry_eligible_ts
universe_policy_hash
feature_contract_version/hash, source_data_hashes
features, observed, source_age/status
```

Snapshot identity зависит только от market inputs и causal features. Изменение
rules или legacy confidence не должно создавать «новый рынок».

### `RuleEvaluationV1`

```text
snapshot_id, strategy_spec_hash
arm/confirmation linkage, rule result, failed gates, diagnostics
evaluated_at
```

### `TradeProposalV1`

```text
proposal_id, snapshot_id, side=SHORT
decision_reference_price, stop_price, take_profit_price
target_distance_pct, stop_distance_pct, horizon
instrument_spec_hash, execution_contract_hash
proposal_status / rejection_reason
```

### `OutcomeLabelV1`

```text
snapshot_id, proposal_id, contract_hash
label_status=pending|mature|incomplete|unfilled
outcome_class=tp_first|sl_first|timeout
entry_ts, exit_ts, net_pnl_quote
return_on_notional, return_on_risk, funding_pnl
forward_data_manifest_hash
```

### `ShadowPredictionV1`

```text
snapshot_id, artifact_id, predicted_at
p_tp_first, p_sl_first, p_timeout
expected_net_return, uncertainty
status=ok|abstain|schema_error|stale|error
```

RuleEvaluation, TradeProposal и Prediction хранятся отдельно: они не входят в
market snapshot hash. Это позволяет честно сравнивать rules/models на одном
рынке без изменения identity.

## 9. Новый план реализации

### Phase 0 — foundation (выполнено)

- [x] MEXC signals-only, closed-bar full-population journal.
- [x] Single-position mechanics v1.
- [x] Versioned feature registry/hash/missingness.
- [x] Funding universe→strategy.
- [x] Strict population journal reader.
- [x] Strategy/universe config fingerprints and model-role whitelist.
- [x] Unfilled-leader same-cohort look-ahead removed.
- [x] Full regression: `352 passed, 4 skipped, 2 known collection warnings`.
- [x] Изменения разделены на reviewable commits и опубликованы fast-forward:
  `0b010e8`, `3ff8de0`, `29536f1`.

Acceptance: schema стабильна, HOLD не теряются, старый CSV не может случайно
попасть в новый builder.

### Phase 1 — P0 time/spec contract

1. Ввести `StrategySpecV2` и один canonical config hash; YAML и scanner должны
   использовать один объект.
2. Закрепить feature/base/execution/15m/1h/4h intervals отдельно.
3. Выразить окна в секундах либо именованных timeframe bars.
4. Разделить armed, confirmed, cycle-complete, actionable и executable entry.
5. Группировать selection по cycle/cohort, а не float compute timestamp.
6. Добавить canonical serialization/hash single-position contract.
7. Разделить identity `MarketFeatureSnapshot`, `RuleEvaluation`, proposal и
   prediction.
8. Фиксировать source request/received/exchange time; `universe.refreshed_at`
   сейчас ставится до завершения fetch и недостаточен.
9. Получать point-in-time instrument specs: contract size, quantity step,
   minimum quantity/notional, leverage, timestamp и hash.

Acceptance: worker count/order не меняет snapshot, ranking, entry или outcome;
первый execution bar действительно доступен после `actionable_ts`.

### Phase 2 — unconditional causal feature parity

1. Вычислять полный snapshot для каждого feature-valid symbol до rule gates.
2. Journal raw ledger всех MEXC USDT contracts с inclusion/exclusion reason.
3. Подключить настоящие closed Min15/Min60/Hour4 frames.
4. Реализовать explicit 1h RSI, overhead, Fibonacci, confluence, weakness,
   exhaustion/wicks/acceleration; gates оставить выключенными.
5. Добавить funding/OI value, units, observed_at, age, change; raw `holdVol`
   оставить diagnostic до notional normalization.
6. Отдельно спроектировать public depth/trades collector; не подменять true flow
   candle proxy.
7. Estimated liquidation оставить отдельным tagged proxy.

Acceptance: одна и та же schema для всех valid rows; никакой feature не исчезает
из-за раннего failed gate; `value=0` не равен missing.

### Phase 3 — executable labels

1. Создать deterministic TradeProposal для всей proposal-eligible population.
2. Получить point-in-time instrument rules и frozen costs/sizing/horizon.
3. Связать journal с `replay_single_short()`.
4. Settlement funding брать из timestamped future series отдельно от decision
   funding feature.
5. Label хранить append-only отдельно от DecisionSnapshot.

Acceptance: один snapshot получает не более одного label; incomplete horizon не
становится win/loss; изменение любого contract параметра меняет hash.

### Phase 4 — collect + mature prospective population

После ввода новой unconditional schema собрать новые point-in-time cycles и
дождаться maturation labels. Legacy CSV можно использовать только для discovery
и regression, но не для admission первого model candidate.

Acceptance: достаточно независимых time blocks/symbols/regimes; каждый training
row воспроизводится из immutable snapshot + proposal + contract + forward-data
manifest.

### Phase 5 — honest evaluation harness

1. Cohort-grouped chronological train/calibration/test.
2. Purge + embargo не меньше максимального label horizon.
3. Untouched test недоступен fit/calibration/threshold/Optuna.
4. No-trade, rules-only, constant/logistic и несколько заранее seeded
   matched-random прогонов на тех же cohorts, costs и concurrency one.
5. Paired time-block и two-way `time × symbol` uncertainty плюс
   cost/data-quality stress; одного symbol-clustered CI недостаточно.
6. Метрики: net EV/CI, drawdown/tail, coverage, Brier/log loss, calibration,
   PR-AUC; AUC/win rate не являются admission metric.

Acceptance: повторный no-edge принимается; test открывается один раз для frozen
candidate и создаёт immutable evaluation receipt.

### Phase 6 — LightGBM shadow candidate

1. Train fixed small multiclass LightGBM + conditional payoff/EV head; proposal
   geometry является явным conditioning input; без scaler и auto fallback.
2. Separate calibration slice; fit calibrator не видит test.
3. Manifest содержит code/data/feature/label/contract/fold hashes.
4. Inference append-only, exception/schema/drift => abstain.
5. Никакой auto-promotion и online overwrite champion.

Acceptance: prediction не меняет rule decision; artifact с missing/corrupt
manifest/hash не загружается; model лучше matched baselines net of costs.

### Phase 7 — prospective shadow evidence

Собрать несколько независимых режимов и symbols в shadow. Promotion возможен
только при положительной нижней границе paired time/symbol uncertainty, устойчивой
calibration и cost stress, повторяемом rebuild и явном решении оператора.

### Phase 8 — challengers

- CatBoost — первый parity challenger.
- TCN — только causal left-looking windows и train-only normalization.
- Chronos — только frozen prospective predictions; historical backfill не
  считается независимым доказательством.
- LLM context — последняя ablation; numeric baseline сначала оценивается без неё.

### Phase 9 — model-assisted signals-only / paper

После явного promotion model policy может влиять только на signals-only/paper
ranking в контролируемом A/B против frozen rules: append-only decisions,
instant rollback, hard abstain и ручное promotion. Это отдельная ступень между
shadow и любым private execution.

### Phase 10 — private/live (отдельный будущий проект)

Только после rotation/history remediation, устойчивого prospective edge,
MEXC private adapter, test harness, rounding/idempotency/reconciliation, hard
capital caps и emergency stop. Текущее разрешение менять код не является
разрешением рисковать капиталом.

## 10. Обязательные regression/causality tests

1. Полный cycle не теряет HOLD/error rows.
2. Bar sources закрыты к `bar_cutoff_ts`; каждый non-bar source имеет
   authoritative `received_at <= decision/cycle completion`.
3. Config/feature/contract hash меняются при изменении semantics.
4. Cycle не делится между folds.
5. `actionable_ts <= entry_eligible_ts` и entry candle реально доступна.
6. Worker count/order не меняет cohort ranking.
7. Unfilled cohort leader не заменяется runner-up по будущему outcome.
8. Gapped horizon => incomplete, не synthetic outcome.
9. Test labels недоступны fit/calibration/tuning.
10. Missing manifest/hash => abstain; pickle без trusted manifest запрещён.
11. Prediction exception не меняет deterministic action.
12. Matched random имеет те же cohorts, symbols, costs и concurrency.
13. Ingestion context с `first_seen_at > decision_completed_ts` отбрасывается;
    LLM не задаёт timestamps.
14. LLM schema технически запрещает private/action/risk поля.

## 11. Текущий operational verdict

- MEXC остаётся целевой биржей.
- Scanner/бот не запускался в этом этапе.
- Private API, Telegram, testnet/live не используются: ключи не ротированы.
- Старые CSV и tracked ML artifacts — legacy/discovery-only.
- Реализованный feature contract улучшает воспроизводимость, но **не доказывает
  edge** и пока не разрешает model fit.
- Следующая задача: **Phase 1 time/spec contract**, затем unconditional parity.

## 12. Первичные источники для выбора технологий

- OpenAI current model guide: <https://developers.openai.com/api/docs/guides/latest-model>
- Kimi model selection: <https://www.kimi.com/help/kimi-api/api-model-selection>
- Kimi API troubleshooting: <https://www.kimi.com/help/kimi-api/api-troubleshooting>
- Gemini models: <https://ai.google.dev/gemini-api/docs/models>
- LightGBM features: <https://lightgbm.readthedocs.io/en/stable/Features.html>
- CatBoost categorical features: <https://catboost.ai/docs/en/features/categorical-features>
- XGBoost AFT: <https://xgboost.readthedocs.io/en/stable/tutorials/aft_survival_analysis.html>
- Chronos: <https://github.com/amazon-science/chronos-forecasting>
- TabPFN: <https://github.com/PriorLabs/TabPFN>
- DuckDB/Parquet: <https://duckdb.org/docs/stable/data/parquet/overview>
- MLflow tracking: <https://mlflow.org/docs/latest/ml/tracking/>
- Optuna: <https://optuna.readthedocs.io/en/stable/>
- Evidently: <https://docs.evidentlyai.com/>
- ONNX Runtime: <https://onnxruntime.ai/docs/>
