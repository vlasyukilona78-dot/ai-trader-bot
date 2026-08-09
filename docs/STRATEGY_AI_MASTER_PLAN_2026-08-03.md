# Koteika Ultra — единая стратегия, AI-архитектура и новый план

Актуально: **2026-08-09, Europe/Moscow**. Целевая площадка: **MEXC futures**.

Этот документ заменяет идею «добавить одну умную нейросеть, которая почувствует
разворот» на проверяемую систему. Он объединяет исполняемую стратегию, causal
feature contract, single-position PnL, роли моделей, research-инструменты и
правила допуска в shadow. До доказанного edge система остаётся signals-only.

## 0. Исполняемый checkpoint 2026-08-09

Текущее состояние кода, которое заменяет более ранние промежуточные статусы ниже:

- latest executable tip: **`258c35f`** (`test(strategy): pin v2 behavioral
  semantics`); он наследует StrategySpec/journal-v5 tip `2d0efcb`;
- реализован строгий `MexcStrategySpec` version `mexc_strategy_v2` с отдельным
  `config/mexc_strategy_v2.yaml`; production scanner, `LayeredPumpStrategy`, base
  indicators, HTF indicators, volume profile и evidence используют один resolved
  объект, один contract hash и один instance hash;
- legacy `--timeframe` / `--candles` больше не являются вторым источником
  конфигурации: это только fail-closed assertions против YAML;
- все численные indicator/VP/min-history параметры из spec реально доходят до
  исполнения. Объявленные, но ещё не реализованные `min_rsi_1h` и
  `require_confluence` при ненулевом значении отвергаются, а не молча игнорируются;
- актуальная version matrix: population journal **v5**, `CycleEnvelope` **v3**,
  `mexc_reversal_features_v2`, single-position contract **v3**;
- `CycleEnvelope v3` хранит canonical StrategySpec payload, version, contract hash
  и instance hash; его timeframe обязан совпадать со spec по физической длине бара;
- journal v5 образует непрерывную цепочку `journal_id` / `sequence_no` /
  `prev_cycle_commit` / `cycle_commit`, проверяемую при reopen и строгом чтении;
- numeric golden vectors фиксируют семантику cumulative VWAP, close-to-close OBV,
  candle-direction CVD и точные POC/VAH/VAL `core_volume_profile_v1`; отдельный
  golden lifecycle фиксирует default `armed HOLD → confirmed SHORT_ENTRY`, полный
  стабильный trace и proposal, исключая только wall-clock identity;
- frozen fixture `tests/fixtures/mexc_strategy_v2_cycle_envelope_v3.json`
  закрепляет canonical v2 hashes/payload и доказывает чтение исторического
  `mexc_strategy_v2` evidence через `CycleEnvelope v3`;
- focused StrategySpec/runtime review: **нет открытых P0/P1**. Единственный P2 —
  отсутствие численных behavioral anchors за declarative revisions — закрыт
  `258c35f` только тестами/fixture; production algorithms, thresholds, spec
  version и hashes не изменились. Journal/checkpoint red-team также не оставил
  P0/P1/P2;
- полный локальный regression checkpoint: **580 passed, 4 skipped, 2 known
  collection warnings (`14.99s`)**. Это проверка инвариантов, а не доказательство edge.

### Граница доверия journal v5

Внутренняя hash-chain обнаруживает изменение раннего цикла, если последующий хвост
не был согласованно переписан, и не позволяет stale writer продолжить изменённую
историю. Но публичная hash-chain сама по себе **не аутентифицирует полностью
переписанный файл**: скоординированный writer может пересчитать всю цепочку.

Tamper-evidence появляется только при явной передаче `JournalCheckpointReceipt`,
полученного ранее и сохранённого **вне той же границы перезаписи**. Receipt якорит
точный prefix (`journal_id`, sequence, chain tip, byte length и prefix SHA-256).
Файл рядом с journal не становится trusted автоматически; auto-discovery receipt
намеренно отсутствует. Model-input reader требует внешний checkpoint либо явный
unsafe override и по умолчанию выдаёт только заякоренный prefix.

### Текущие physical semantics: event horizons и estimator/sample budgets

`window_semantics=fixed_bar_counts`: смена base interval меняет физическую длину
окна; код не делает скрытого пересчёта в секунды. При текущих `Min60` base и
`Hour4` HTF параметры делятся на два разных класса.

Событийные, state-transition и structural horizons описывают, какую рыночную
историю считает одной гипотезой сама стратегия:

| Event/state parameter | Bars × source timeframe | Физическая длительность v2 |
|---|---:|---:|
| `pump_window_bars` | 45 × Min60 | 45 часов |
| `confirmation_max_wait_bars` | 3 × Min60 | до 3 часов |
| `msb_recent_bars` / `msb_lookback` | 6 / 20 × Min60 | 6 / 20 часов |
| `weakness_lookback` | 4 × Min60 | 4 часа |
| `structural_anchor_htf_bars` | 12 × Hour4 | 48 часов |

Estimator, warm-up и sample budgets задают число наблюдений и устойчивость
оценки. Они имеют физическую длительность при данном source timeframe, но не
являются автоматически «длительностью пампа»:

| Estimator/sample parameter | Bars × source timeframe | Фактический budget v2 |
|---|---:|---:|
| base input frame | 320 × Min60 | до 320 часовых баров |
| `liquidity_lookback_bars` | 12 × Min60 | 12 часов |
| `relative_strength_lookback` | 24 × Min60 | 24 часа |
| base RSI / ATR configured period | 14 × Min60 | 14 часовых баров |
| base ADX configured period | 14 × Min60, применён в двух rolling stages | около 27 баров warm-up; не простой 14-hour window |
| base EMA spans | 20 / 50 × Min60 | span 20 / 50 часовых баров; EMA не имеет конечного окна |
| base BB / Keltner / volume MA periods | 20 / 20 / 20 × Min60 | 20 часовых баров; Keltner содержит EMA |
| VWAP / OBV / candle-CVD | cumulative input frame | до 320 часовых баров текущего frame |
| volume profile | window 120, minimum history/sample 20/24 × Min60 | до 120 часов; отдельные eligibility/sample floors |
| HTF RSI configured period | 14 × Hour4 | 14 четырёхчасовых баров (56 часов) |

Теперь эти значения доказуемо соответствуют исполняемому v2 spec. При проектировании
Min15 нельзя механически делить или сохранять **все** bar counts одним правилом:
event horizons должны соответствовать рыночной гипотезе, а estimator/sample budgets
— достаточной истории, resolution и warm-up. Min15 поэтому является новой стратегией
в отдельной version namespace, а не override frozen `mexc_strategy_v2`.

Исторический комментарий `SignalConfig` связывает «45 bars» с наблюдением о
трёхпроцентном движении примерно за 20 минут; старые collectors также работали с
minute timeframes. Это **историческая подсказка о возможном быстром intent**, а не
принятое решение. Она не доказывает ни выбор Min15, ни 45 баров на Min15, ни иной
конкретный horizon. Менять временной контракт вместе с threshold calibration
запрещено: сначала новая hypothesis получает отдельную версию и acceptance, затем
оценивается без молчаливого переноса claims из no-edge v2.

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

Это **proposal-conditioned** target: модель отвечает, что произойдёт с конкретным
causal proposal и его execution contract. Поэтому proposal features — stop/target
distance, realized risk/reward и cost geometry — сохраняются как conditioning
inputs. Если исследуется чистая вероятность направления/разворота рынка независимо
от сделки, это другой target, другой dataset view и отдельная model head; proposal-
derived features в такой pure-direction задаче запрещены. Эти две постановки нельзя
смешивать в одной метрике или выдавать одну за другую.

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

Раздел сохраняет исторический audit trail. Пункты, закрытые checkpoint 2026-08-08,
не удаляются, а явно помечаются **superseded**; непомеченная часть остаётся
известной roadmap-границей. На executable tip `258c35f` focused review не оставил
открытых P0/P1 **в уже реализованном StrategySpec/runtime scope**. Перечисленные
ниже dataset, lifecycle, provenance и feature-parity работы не стали выполненными,
но они уже явно отделены и блокируют model fit, поэтому не являются новой
необнаруженной P0/P1-регрессией текущего signals-only worktree. Единственный
focused-review P2 (revision literals без numerical behavioral anchors) закрыт
`258c35f` без изменения runtime behavior.

### Исторический P0 — время и исполнимость (часть закрыта; time-hypothesis остаётся)

- **Историческая находка, уточнена (частично superseded):** scanner действительно
  использует `Min60`, поэтому `pump_window_bars=45` — это 45
  часов, `confirmation_max_wait_bars=3` — до трёх часов, `msb_recent_bars=6` —
  шесть часов. Теперь это явно закреплено `StrategySpecV2` как fixed-bar semantics,
  а не скрытый config drift. Нерешённым остаётся выбор: соответствуют ли эти
  физические горизонты задуманному быстрому intraday pump.
- Research timing теперь разделяет cutoff, source responses, ranking completion,
  `actionable_ts`, `entry_eligible_ts` и первый достижимый execution bar.
  Cohorts группируются по `cycle_id`, а не по float wall-clock; unfilled leader
  не заменяется runner-up по будущему outcome.
- Single-position v3 связывает план, полный execution contract, нормализованные
  бары, строго упорядоченный funding и результат. Selector повторяет replay по
  обязательному `ReplayEvidence`, поэтому одного самозаявленного hash недостаточно.
- **Частично superseded:** единый `StrategySpecV2` и явные source-bar windows уже
  реализованы. Для model-ready labels остаются выбор intended physical horizons,
  arm/confirm lifecycle, point-in-time instrument rules и
  journal→proposal→label bridge. Research timing не является доказательством
  своевременной Telegram/live-доставки.

### Исторический P0 — dataset / labels / model (model-ready path остаётся заблокирован)

- **Superseded по версии:** промежуточный population journal v4 заменён journal v5
  с chain/checkpoint semantics. Строгий reader образует проверяемый ordered
  runtime-population источник, но label builder ещё не связан с ним.
- Старые builders создают event-conditioned LONG/DCA labels; новый
  `replay_single_short()` ими не вызывается.
- Старый trainer использует `target_win/target_horizon`, row-wise split, выбирает
  XGBoost первым и fit-ит isotonic calibrator на test block.
- MEXC inference к scanner не подключён. Старые pickle без manifest/hash не
  являются допустимыми artifacts и не должны загружаться.

### Исторический P0 — feature parity (silent activation закрыта; parity не завершена)

- Funding из frozen universe snapshot уже передаётся в strategy context; raw OI
  по-прежнему не нормализован по contract size и остаётся diagnostic/unconsumed.
- **Superseded как silent-ignore дефект:** `min_rsi_1h` и `require_confluence` всё
  ещё не реализованы в decision path, но StrategySpec теперь fail-closed отвергает
  их ненулевые значения.
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

### Исторический P1 — конфигурация и provenance (config drift закрыт; gaps остаются)

- **Superseded:** legacy `config/config.yaml` по-прежнему не является источником
  MEXC defaults, но теперь это намеренно: scanner загружает отдельный строгий
  `config/mexc_strategy_v2.yaml` и передаёт один resolved spec в strategy/data/evidence.
- MEXC OI `holdVol` ещё не имеет закреплённых units, history/delta и age.
- Contract details по умолчанию не загружаются, поэтому отсутствуют точные
  quantity step, minimum quantity, contract size и надёжный max leverage.
- Estimated liquidation map — OHLCV/leverage proxy. Его запрещено называть
  биржевым liquidation feed и смешивать с liquidation price нашей позиции.
- Empty-universe и pre-scan errors теперь получают durable zero-row envelope;
  ошибка записи при включённом journal останавливает цикл fail-closed.
- Base/HTF timing разделён по источникам, но остаётся cycle aggregate, а не
  per-symbol/per-timeframe evidence. Оборванный tail намеренно блокирует reopen
  и пока требует ручного переноса файла вместо автоматического repair.
- Комментарии `measured best`, `validated` и исторические численные thresholds
  считаются только frozen legacy hypotheses: финальный no-edge отменил статус
  доказанного преимущества.

## 4. Реализованный фундамент этого этапа

Добавлен `ai/reversal/feature_contract.py`:

- version `mexc_reversal_features_v2`;
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
- фиксирует полный StrategySpec version/contract/instance identity и
  universe-policy hash;
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
  action/status/rule columns; model inputs требуют explicit trusted checkpoint
  либо явно названный unsafe unanchored режим.

Добавлен `core/mexc_strategy_spec.py` и dedicated YAML:

- strict exact-key/type/range parsing и запрет duplicate YAML keys;
- canonical interval aliases, frozen payload, pinned contract hash и instance hash;
- один executable adapter для `SignalConfig`, volatility, base/HTF indicators,
  volume profile, history bounds и HTF cache;
- runtime-consistency check не позволяет explicit spec описывать другую unbound
  strategy configuration;
- `CycleEnvelope v3` повторно строит spec из payload и связывает его с физическим
  timeframe цикла.

Behavioral/compatibility checkpoint `258c35f` добавляет три исполняемых locks:

- exact numeric vectors для VWAP/OBV/candle-CVD;
- exact volume-profile POC/VAH/VAL vector, фиксирующий tail window, bins, POC и
  descending-volume value-area semantics;
- rounded canonical digest полного default lifecycle `armed HOLD → confirmed
  SHORT_ENTRY`, включая stable decision, trace и proposal поля. Из digest исключены
  только wall-clock `created_at` и `legacy_signal_id`.

Отдельная frozen fixture сохраняет canonical `mexc_strategy_v2` payload, v2 hashes
и `CycleEnvelope v3`. Она является compatibility gate: новый current spec не имеет
права сделать существующее v2 evidence нечитаемым.

Поэтому будущий `mexc_strategy_v3` нельзя реализовать простой заменой глобального
`MEXC_STRATEGY_SPEC_VERSION`/expected literal и текущего parser. Нужен
version-dispatched evidence reader/registry: persisted `strategy_spec_version`
выбирает неизменяемый v2 parser с его hashes либо отдельные v3 types/parser/hash.
V3 config, artifacts и runtime evidence получают отдельную version namespace;
frozen v2 fixture остаётся обязательным regression test.

Population journal v5 добавляет domain-separated cycle commitments, непрерывную
цепочку, restart audit и detached `JournalCheckpointReceipt`. Это обеспечивает
внутреннюю целостность и external-prefix anchoring в заявленной выше границе, но
не превращает незаякоренный файл в криптографически аутентифицированный источник.

Это ещё не model-ready dataset: research entry timing уже построен, но поздние
layer features пока structural-missing, point-in-time instrument contract и
proposal/label bridge отсутствуют, а новая prospective population не накоплена.

## 5. Единая таблица стратегии

| Блок | Текущий runtime | Contract | Что исправить до model fit |
|---|---|---|---|
| Universe | весь **уже отфильтрованный** turnover-band cycle journalled | conditional population boundary | ledger всех USDT contracts: included/exclusion reason, received-at, policy hash, instrument rules |
| Input | один closed cutoff, Min60/320 | `mexc_strategy_v2`, CycleEnvelope v3; frozen v2 compatibility fixture | до Min15 сначала добавить version-dispatched v2 reader и отдельную v3 namespace; event horizons проектировать отдельно от estimator/sample budgets |
| L1 pump | recent band event + RSI/volume + move/retrace | candidate features + frozen rule baseline | unconditional values; 45-hour v2 остаётся frozen, а historical 45-bar/minute clue не выбирает новый horizon |
| L1b quality | ATR floor + exact quote turnover; legacy fallback существует | features; data quality отдельно | fallback `close×contract volume` помечать missing, interval-normalized turnover, no early truncation |
| L1c market | BTC RS, 4h RSI, optional overhead/chase; unsupported 1h/confluence nonzero fail-closed | numeric context | explicit missing, fixed 1h/4h sources, fib/confluence ablation |
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
| LightGBM multiclass + EV head | первый proposal-conditioned champion candidate | tabular, missing values, CPU, быстрые ablations; proposal geometry остаётся conditioning input | class probabilities не описывают timeout payoff; overfit | **первый fit после prospective maturity**; pure direction — отдельный target/head без proposal fields |
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
- [x] **Исторический regression checkpoint, superseded:** `352 passed, 4 skipped,
  2 known collection warnings`; актуальный receipt `580 passed, 4 skipped,
  2 known collection warnings (14.99s)` приведён в §0.
- [x] Изменения разделены на reviewable commits и опубликованы fast-forward:
  `0b010e8`, `3ff8de0`, `29536f1`.

Acceptance: schema стабильна, HOLD не теряются, старый CSV не может случайно
попасть в новый builder.

### Phase 1 — P0 time/spec contract

Slice 1 не был принят независимым ревью: 388 зелёных тестов не покрывали
runtime- и schema-блокеры. Дефекты устранены тремя slice'ами
(`8b03d59`, `577dd9d`, `7290863`). Ниже — состояние после них.

Отозванные утверждения первой попытки:

- «wall-clock не меняет hash» было верно только для `decision_ts`; сам снапшот
  нёс `universe_refreshed_at`, то есть время всё-таки входило в identity.
  Теперь wall-clock вынесен в `feature_provenance` вне `input_hash`.
- «пустой цикл оставляет envelope» — envelope строился, но не сохранялся;
  durable-записи не было. Теперь пишется header+footer, читается после рестарта.
- «worker-order acceptance выполнена» — заявлено преждевременно: холодный старт
  `VolatilityContext` оставался зависимым от порядка, потому что пустой frozen
  snapshot проваливался к живым наблюдениям. Исправлено явным флагом sweep.

Закрыты пункты 4 (частично), 5 и 8:

- [x] `SourceTiming` фиксирует `request_started_at` / `received_at` /
  `source_as_of` / status для universe, benchmark и рыночных данных; отказавший
  источник тоже записывает время ожидания.
- [x] `universe.refreshed_at` больше не выдаётся за момент получения данных: он
  остаётся якорем TTL, а `received_at` замеряется после ответа API.
- [x] `CycleEnvelope` вводит `ranking_ready_ts`, `cycle_completed_ts`,
  `actionable_ts`, `entry_eligible_ts`, `entry_bar_open_ts` и выводит их сам.
- [x] `entry_bar_open_ts` — первая выровненная свеча строго после
  `entry_eligible_ts`; вход на открытии свечи, известной в момент её открытия,
  запрещён.
- [x] Selection группируется по `cohort_id`, а не по равенству float
  `decision_ts`. `EntryPlan` несёт когорту, то есть она известна до replay.
- [x] `replay_single_short` требует первую свечу на `entry_bar_open_ts`;
  `entry_ts` следует за исполнением, а не за решением.
- [x] **Историческая matrix, superseded:** population journal v4,
  `CycleEnvelope` v2, `mexc_reversal_features_v2`, single-position contract v3.
  Актуальная matrix приведена в checkpoint §0; несовместимые schemas fail-closed.
- [x] Пустой и ошибочный цикл пишутся в журнал как header+footer и читаются
  после рестарта.
- [x] Envelope пишется один раз на цикл (header/rows/footer, row count и
  checksum), а не копируется в каждую строку; 300 символов не падают.
- [x] Writer запрещает смешение schema-версий, чужие строки в batch и склейку
  оборванного хвоста.
- [x] Reader пересобирает envelope и сверяет каждый cycle-level факт строки.
- [x] `mexc_reversal_features_v2`: снапшот без wall-clock, провенанс рядом,
  новый executable hash закреплён литералом.
- [x] Тайминги разделены: universe ticker, contract details, benchmark,
  base OHLCV, higher timeframe; кэш тикеров не выдаётся за свежий ответ.
- [x] Холодный старт волатильности инвариантен к порядку воркеров (проверено на
  реальной `LayeredPumpStrategy`, 28 символов, оба порядка).
- [x] **Исторический regression checkpoint, superseded:** `529 passed, 4 skipped,
  2 known collection warnings` (`13.85s`). Следующий исторический checkpoint был
  `576 passed, 4 skipped, 2 known collection warnings`; актуальный результат:
  `580 passed, 4 skipped, 2 known collection warnings (14.99s)`.

Что timing **не** доказывает: `entry_bar_open_ts` помечен
`timing_basis="research_ranking_ready"`. Он обосновывает сравнимость когорты в
research-replay и не утверждает, что живая signals-only доставка успела бы к
этому бару — построение записи, fsync, возврат и канал доставки не измерены.

Hardening 2026-08-08 опубликован тремя отдельными code commits:

- `32e8fbe` — journal v4: envelope hash в header/body/footer, checksum полных
  canonical decision rows, exact ordered population, zero-row terminal cycles,
  полный restart audit, two-pass reader до первого yield и Windows/POSIX
  inter-process append lock;
- `0c32047` — single-position v3: `plan_hash`, `contract_hash`,
  `replay_input_hash`, `result_hash`, обязательный immutable `ReplayEvidence`,
  повторный replay в selector и strict-increasing funding timestamps;
- `e0e4cb4` — benchmark fail-closed по умолчанию; legacy fail-open оставлен
  только как явная ablation `require_benchmark=False`. Это меняет eligibility и
  signal count, но само по себе не является доказательством edge.

Следующий checkpoint того же дня, зафиксированный коммитами `bebfd0d` и
`2d0efcb`, заменяет промежуточный journal v4:

- journal v5: contiguous hash-chain, restart/stale-writer validation, explicit
  detached checkpoint и anchored-only model-input boundary;
- `CycleEnvelope v3`: полный canonical StrategySpec payload и обязательное
  совпадение физического timeframe;
- `mexc_strategy_v2`: dedicated YAML, единые runtime adapters и fail-closed
  проверка spec/runtime drift;
- **исторический regression receipt:** `576 passed, 4 skipped, 2 known collection
  warnings (15.12s)`; superseding tip `258c35f` и его receipt приведены ниже.

Дополнительно fresh/TTL-cache/stale-cache/first-failure сохраняют разные
provenance; contract details имеют отдельный clock; mixed HTF cache не выдаётся
за полностью fresh; закрытый frame больше не делает скрытый live-ticker запрос.
`source_ts` свежего ticker/details — локальный response instant, а не
exchange-supplied server timestamp. `market_feature_hash` связывает symbol,
timeframe и market-only snapshot. При этом `PopulationDecision.input_hash` и
`snapshot_id` намеренно продолжают включать cycle/rule identity, поэтому это не
замена отдельным persisted `MarketFeatureSnapshot` и `RuleEvaluation`.

Behavioral compatibility checkpoint `258c35f` не меняет production behavior:

- indicator и volume-profile semantics получили numerical golden locks; один
  representative default arm→confirm logic/proposal path закреплён явными
  assertions и digest с округлением float до 12 знаков;
- committed v2 envelope fixture фиксирует canonical SHA-256, v2 contract/instance
  hashes и чтение через текущий strict path;
- focused review после закрытия P2: StrategySpec/runtime **P0/P1 none**,
  journal/checkpoint **P0/P1/P2 none**;
- полный receipt: **580 passed, 4 skipped, 2 known PytestCollectionWarning
  (14.99s)**; scanner, network, Telegram, model fit и private APIs не запускались.

Новый remaining order после checkpoint:

1. **[x] Superseded/выполнено:** ввести `StrategySpecV2`, dedicated YAML, один
   canonical instance hash и именованные source-timeframe bars. Текущие durations
   приведены в §0; behavioral semantics и v2 envelope закреплены `258c35f`.
2. **До создания v3 построить compatibility dispatch:** persisted
   `strategy_spec_version` выбирает неизменяемый v2 reader/parser/hash namespace;
   новые v3 types/config/hash/evidence живут отдельно. Naive global bump, после
   которого frozen v2 fixture перестаёт читаться, запрещён.
3. **Выбрать новую intended time hypothesis без калибровки thresholds:** текущий
   `mexc_strategy_v2` остаётся frozen Min60/45-hour baseline. Возможный Min15 path
   является новой стратегией: event/state horizons задаются отдельно от estimator
   warm-up/sample budgets. Историческая связь «45 bars» с minute-scale pump —
   только clue, не принятое значение ни одного нового параметра.
4. Разделить armed и confirmed как typed states и заранее закрепить scoring
   instant; cycle-complete, research-actionable и eligible entry уже разделены.
5. Разделить persisted identity `MarketFeatureSnapshot`, `RuleEvaluation`,
   `TradeProposal` и prediction; одновременно сделать per-symbol base и
   per-symbol/per-timeframe HTF timing, а не только cycle aggregate.
6. Получать point-in-time instrument specs: contract size, quantity step,
   minimum quantity/notional, leverage, timestamp и hash.
7. Завершить unconditional feature parity и raw-contract inclusion ledger до
   построения labels: ранний gate не должен определять missingness.
8. Добавить durable public serialization/reader для single-position v3,
   forward-data manifest и journal→proposal→label bridge.
9. Только после новой schema и external checkpoint procedure начинать
   prospective collection/maturation; затем chronological evaluation и первый fit.

Для каждого фактически используемого journal v5 prefix оператор должен вынести
`JournalCheckpointReceipt` за пределы той же перезаписываемой runtime-директории.
Без этого chain остаётся `internally_consistent_unanchored`, а не trusted evidence.

Acceptance: worker count/order не меняет snapshot, ranking, entry или outcome —
**выполнено для cycle/cohort identity, entry timing и холодного старта
волатильности**; полная acceptance требует ещё безусловного trace (Phase 2),
пока поздние признаки зависят от того, где оборвался gate. Первый бар доступен
после `actionable_ts` **в смысле research-ranking**, не execution.

### Phase 2 — unconditional causal feature parity

1. Вычислять полный snapshot для каждого feature-valid symbol до rule gates.
2. Journal raw ledger всех MEXC USDT contracts с inclusion/exclusion reason.
3. После version-dispatched v2/v3 boundary подключить настоящие closed
   Min15/Min60/Hour4 frames согласно отдельной новой strategy namespace; не
   переопределять frozen v2.
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

1. Train fixed small multiclass LightGBM + conditional payoff/EV head для
   proposal-conditioned outcome; proposal geometry остаётся явным conditioning
   input, без scaler и auto fallback. Pure direction/reversal допускается только
   как отдельный target/head и исключает proposal-derived features.
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
15. Journal chain различает internal consistency и externally anchored prefix;
    coordinated rewrite проходит только без trusted receipt и обнаруживается с ним.
16. Cycle timeframe физически совпадает со StrategySpec; custom indicator/VP/HTF
    параметры меняют реальное исполнение, а не только instance hash.
17. Indicator/VP/logic revision literals имеют numerical behavioral locks; full
    logic digest исключает только wall-clock identity, но сохраняет decision,
    trace и proposal semantics.
18. После появления нового current spec frozen `mexc_strategy_v2` fixture всё ещё
    читается version-dispatched v2 path и сохраняет исходные payload/hashes;
    v3 не переиспользует v2 namespace.

## 11. Текущий operational verdict

- MEXC остаётся целевой биржей.
- Scanner/бот не запускался в этом этапе.
- Private API, Telegram, testnet/live не используются: ключи не ротированы.
- Старые CSV и tracked ML artifacts — legacy/discovery-only.
- Реализованный feature contract улучшает воспроизводимость, но **не доказывает
  edge** и пока не разрешает model fit.
- Phase 1 timing/journal/replay foundation теперь включает journal v5,
  `CycleEnvelope v3`, `mexc_strategy_v2` и single-position v3; benchmark
  fail-closed. Hash-chain без вынесенного receipt остаётся unanchored. Это делает
  evidence строже, но **edge по-прежнему не установлен**.
- Executable tip `258c35f`: `580 passed, 4 skipped, 2 known
  PytestCollectionWarning (14.99s)`; focused implemented-scope verdict — P0/P1
  none, закрытый P2 не менял production behavior. Frozen v2 compatibility и
  indicator/VP и representative default lifecycle locks включены в этот receipt.
- Следующий порядок: version-dispatched immutable v2 reader + отдельная v3
  namespace → отдельная новая time hypothesis с раздельными event horizons и
  estimator/sample budgets (Min15 остаётся кандидатом до явного выбора) → typed
  arm/confirm lifecycle → persisted snapshot/rule identity и per-symbol timings
  → point-in-time instrument specs → unconditional parity/raw ledger →
  proposal/label bridge → externally checkpointed prospective collection. Model
  fit до этого запрещён.

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
