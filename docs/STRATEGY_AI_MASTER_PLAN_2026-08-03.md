# Koteika Ultra — единая стратегия, AI-архитектура и новый план

> [!IMPORTANT]
> С **2026-08-15** финальный product/release roadmap перенесён в
> `docs/FINAL_BOT_MASTER_PLAN_2026-08-14.md`, получивший независимый verdict
> `APPROVE_AS_AUTHORITATIVE`. Этот документ не удалён и не переписан: он остаётся
> frozen historical/executable-v2 audit source для реализованных контрактов,
> checkpoint-ов и причин прежних решений. Будущие v3/product решения берутся из
> нового master; факты и frozen-v2 границы здесь сохраняют доказательную силу.

Актуально: **2026-08-11**. Целевая площадка: **MEXC futures**.

Этот документ заменяет идею «добавить одну умную нейросеть, которая почувствует
разворот» на проверяемую систему. Он объединяет исполняемую стратегию, causal
feature contract, single-position PnL, роли моделей, research-инструменты и
правила допуска в shadow. До доказанного edge система остаётся signals-only.

## 0. Исполняемый checkpoint 2026-08-11

Текущее состояние кода, которое заменяет более ранние промежуточные статусы ниже:

- latest executable tip: **`bb1ca13`** (`feat(journal): persist typed population
  evidence in schema v6`); непосредственный lifecycle hardening tip — **`9ef6b4f`**
  (`fix(evidence): finalize typed lifecycle semantics`). Они наследуют frozen-v5
  boundary `eb238b2`, lifecycle/provenance foundation `c541eea` / `a604668` /
  `8569471` / `cf6bc01`, versioned-evidence checkpoint `1971b77`, behavioral
  checkpoint `258c35f` и StrategySpec/journal-v5 tip `2d0efcb`;
- реализован строгий `MexcStrategySpec` version `mexc_strategy_v2` с отдельным
  `config/mexc_strategy_v2.yaml`; production scanner, `LayeredPumpStrategy`, base
  indicators, HTF indicators, volume profile и evidence используют один resolved
  объект, один contract hash и один instance hash;
- legacy `--timeframe` / `--candles` больше не являются вторым источником
  конфигурации: это только fail-closed assertions против YAML;
- все численные indicator/VP/min-history параметры из spec реально доходят до
  исполнения. Объявленные, но ещё не реализованные `min_rsi_1h` и
  `require_confluence` при ненулевом значении отвергаются, а не молча игнорируются;
- актуальная version matrix: current population writer **v6** по пути
  `data/runtime/mexc_population_decisions_v6.jsonl`; frozen journal **v5** остаётся
  строго читаемым, но read-only; `CycleEnvelope` **v3**, cycle identity **v5**,
  `mexc_reversal_features_v2` (pin
  `20f9f61d4e2d787c5ad05f54ee3ccd8b7f8ea3a99fe09bc38bbefe09872c496c`),
  `candidate_lifecycle_v1`,
  `mexc_closed_frame_provenance_v1`, single-position contract **v3**;
- `CycleEnvelope v3` хранит canonical StrategySpec payload, version, contract hash
  и instance hash; его timeframe обязан совпадать со spec по физической длине бара;
- persisted `strategy_spec_version` теперь выбирает зарегистрированный
  version-specific evidence decoder/parser/hash. Текущий registry содержит только
  `mexc_strategy_v2`; неизвестная или несовпадающая версия отвергается fail-closed,
  а production current-loader отделён от historical evidence decoding;
- writer/restart audit, strict dataset reader и `model_input_records()` требуют
  однородную точную StrategySpec identity `(version, contract hash, instance hash)`
  внутри одного journal. Экспорт повторяет эту identity как metadata вне численных
  model features;
- current journal v6 сохраняет ту же проверяемую цепочку `journal_id` /
  `sequence_no` / `prev_cycle_commit` / `cycle_commit`, но добавляет exact typed
  base/benchmark/HTF evidence, raw-frame bundle identity и optional typed lifecycle;
  v5 fixture не переписывается и не дополняется writer'ом;
- typed lifecycle сохраняет arm, каждое confirmation observation и proposal как
  отдельные immutable semantic events. `CandidateArmV1` связывает arm-bar prices,
  effective invalidation и confirmation policy; SAME_BAR обязан быть точным
  повтором, более поздний переход — монотонным, а proposal entry связан с
  подтверждённым reference price. Contract pin:
  `cc75c871b7097aa215f9ac88c736b6572e2443318cb0cf9f8bdaf1b0c8cc8551`;
- public typed strategy API принимает три явных `FrameRead` — base, benchmark и
  HTF — повторно проверяет их identity/hash/cutoff, пересчитывает base indicators
  из сырого frame и не читает mutable benchmark/cache во время решения;
- exact closed-frame provenance различает `fresh`, `stale`, `no_rows`,
  `request_failed` и `not_requested`, связывает symbol/timeframe/cutoff/rows/hash,
  а benchmark/base/HTF bar-source timings выводятся только из persisted evidence.
  Universe и allowed contract-details timings остаются отдельным cycle-level
  provenance, не представленным как `FrameRead`. Contract pin:
  `f4004ac933cc1725b2560e93ffbe278c826910424e350059ad29420ed3665dbf`;
- scanner держит non-blocking lifetime guard на один runtime-владелец journal,
  проверяет `contains_cycle()` до мутации strategy state, не превращает stale или
  malformed source в fresh evidence и пишет terminal benchmark как
  `not_requested`;
- numeric golden vectors фиксируют семантику cumulative VWAP, close-to-close OBV,
  candle-direction CVD и точные POC/VAH/VAL `core_volume_profile_v1`; отдельный
  golden lifecycle фиксирует default `armed HOLD → confirmed SHORT_ENTRY`, полный
  стабильный trace и proposal, исключая только wall-clock identity;
- frozen fixture `tests/fixtures/mexc_strategy_v2_cycle_envelope_v3.json`
  закрепляет canonical v2 hashes/payload и доказывает чтение исторического
  `mexc_strategy_v2` evidence через version-dispatched `CycleEnvelope v3`;
- focused StrategySpec/runtime review: **нет открытых P0/P1**. Единственный P2 —
  отсутствие численных behavioral anchors за declarative revisions — закрыт
  `258c35f` только тестами/fixture; production algorithms, thresholds, spec
  version и hashes не изменились. Journal/checkpoint red-team также не оставил
  P0/P1/P2;
- полный локальный regression checkpoint на `bb1ca13`: **723 passed, 4 skipped,
  2 known PytestCollectionWarning (`17.80s`)**. Это проверка инвариантов, а не
  доказательство edge. Gate-conditioned missingness по-прежнему блокирует model
  training: полный snapshot ещё не вычисляется независимо от пути rules.

### Граница доверия journal v6 и frozen-v5 compatibility

Внутренняя v5/v6 hash-chain обнаруживает изменение раннего цикла, если последующий хвост
не был согласованно переписан, и не позволяет stale writer продолжить изменённую
историю. Но публичная hash-chain сама по себе **не аутентифицирует полностью
переписанный файл**: скоординированный writer может пересчитать всю цепочку.

Tamper-evidence появляется только при явной передаче `JournalCheckpointReceipt`,
полученного ранее и сохранённого **вне той же границы перезаписи**. Receipt якорит
точный prefix (`journal_id`, sequence, chain tip, byte length и prefix SHA-256).
Файл рядом с journal не становится trusted автоматически; auto-discovery receipt
намеренно отсутствует. Model-input reader требует внешний checkpoint либо явный
unsafe override и по умолчанию выдаёт только заякоренный prefix. V5 reader
сохраняет historical readability, но model export из legacy v5 требует ещё и
явный `allow_legacy_v5`; writer v6 не обновляет v5-файл на месте.

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

Раздел сохраняет исторический audit trail. Пункты, закрытые checkpoints 2026-08-08
и 2026-08-11,
не удаляются, а явно помечаются **superseded**; непомеченная часть остаётся
известной roadmap-границей. На executable tip `bb1ca13` typed lifecycle и exact
per-symbol frame provenance уже выполнены; утверждения ниже об их отсутствии
считаются superseded. На behavioral tip `258c35f` focused review не оставил
открытых P0/P1 **в уже реализованном StrategySpec/runtime scope**. Перечисленные
ниже separate-identity, instrument-spec, unconditional dataset/feature-parity и
label работы не выполнены, но уже явно отделены и блокируют model fit, поэтому не являются новой
необнаруженной P0/P1-регрессией текущего signals-only worktree. Единственный
focused-review P2 (revision literals без numerical behavioral anchors) закрыт
`258c35f` без изменения runtime behavior; `1971b77` добавил versioned evidence
compatibility и identity guards. Коммиты `9ef6b4f` и `bb1ca13` затем добавили
typed lifecycle, exact source evidence и journal-v6 persistence без выбора новой
timeframe/threshold hypothesis.

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
- **Частично superseded:** единый `StrategySpecV2`, явные source-bar windows и
  typed arm/confirm/proposal lifecycle уже реализованы. Для model-ready labels
  остаются separate persisted snapshot/rule/proposal identities, point-in-time
  instrument rules и journal→proposal→label bridge. Research timing не является
  доказательством своевременной Telegram/live-доставки.

### Исторический P0 — dataset / labels / model (model-ready path остаётся заблокирован)

- **Superseded по версии:** промежуточный population journal v4 сначала заменён
  frozen v5 с chain/checkpoint semantics; current writer теперь v6 с typed
  lifecycle/frame evidence, а v5 остаётся read-only. Строгий reader образует
  проверяемый ordered runtime-population источник, но label builder ещё не связан с ним.
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
- **Superseded `9ef6b4f`:** arm, confirmation observations и proposal теперь
  связаны typed immutable lifecycle events с отдельными semantic IDs и строгими
  переходами. Это не устраняет gate-conditioned feature missingness и не создаёт
  отдельные persisted `MarketFeatureSnapshot` / `RuleEvaluation` identities.

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
- **Superseded `bb1ca13`:** cycle header несёт exact benchmark evidence, каждый
  symbol row — exact base/HTF evidence и raw bundle hash, связывающий все три;
  benchmark/base/HTF timings обязаны быть их точной проекцией. Universe и allowed
  contract-details timings остаются отдельными cycle-level clocks.
  Stale/no-rows/failure/not-requested различаются fail-closed.
  Оборванный tail намеренно блокирует reopen и пока требует ручного переноса файла
  вместо автоматического repair.
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

Checkpoint `1971b77` реализует version-dispatched evidence reader/registry:
persisted `strategy_spec_version` выбирает зарегистрированный parser, contract hash
и instance-hash implementation. `CycleEnvelope`, journal writer/restart audit и
dataset reader сверяют полную identity `(version, contract hash, instance hash)`;
`model_input_records()` сохраняет её как metadata и не превращает в predictive
features. Frozen v2 contract/instance hashes не изменились.

Evidence checkpoint `9ef6b4f` + `bb1ca13` завершает следующий causal boundary:

- `candidate_lifecycle_v1` хранит отдельные arm/observation/proposal events,
  связывает каждый semantic ID со всем causal predecessor state и отвергает
  backward, ambiguous, partially advanced или execution-bound переходы;
- typed strategy path владеет exact base/benchmark/HTF `FrameRead`, повторно
  валидирует consumed frames и выполняет state mutation транзакционно;
- `mexc_closed_frame_provenance_v1` сохраняет exact frame identity, request/
  receipt timing, frame range/hash, freshness outcome и raw three-frame bundle;
- current journal writer v6 связывает эти evidence objects с каждой decision row,
  envelope/source timing projection и lifecycle chain. Старый schema-v5 fixture
  остаётся read-only и читается отдельным compatibility path;
- scanner lifetime lock отвергает второго runtime owner до первого market request;
  duplicate-cycle preflight выполняется до `begin_sweep()` и strategy mutation;
- adversarial tests отвергают stale or malformed evidence, forged IDs/actions,
  venue/symbol/cycle drift, поздний receipt, arbitrary source timing, cross-cycle
  lifecycle substitution и неканонические numeric/integer types.

Эта граница доказывает provenance/identity invariants, но не model readiness:
поздние признаки всё ещё могут отсутствовать именно потому, что более ранний gate
завершил rule path. До unconditional gate-independent snapshot fit запрещён.

Это code-enforced compatibility boundary и regression gate, а не абсолютная
неизменяемость внешнего файла или аутентификация его происхождения. Будущий
`mexc_strategy_v3` всё равно обязан получить новые types, config, parser, hash и
отдельную evidence namespace; переиспользовать или редактировать v2 namespace
запрещено. Конкретный v3 timeframe, окна и thresholds пока не выбраны. Frozen v2
fixture остаётся обязательным regression test, а доверие к journal prefix по-прежнему
требует ранее вынесенного внешнего checkpoint receipt.

Frozen population journal v5 ввёл domain-separated cycle commitments, непрерывную
цепочку, restart audit и detached `JournalCheckpointReceipt`; current writer v6
сохраняет эту trust model и добавляет typed lifecycle/frame evidence. Это
обеспечивает внутреннюю целостность и external-prefix anchoring в заявленной выше
границе, но не превращает незаякоренный файл в криптографически
аутентифицированный источник.

Это ещё не model-ready dataset: research entry timing уже построен, но поздние
layer features пока structural-missing, point-in-time instrument contract и
proposal/label bridge отсутствуют, а новая prospective population не накоплена.

## 5. Единая таблица стратегии

| Блок | Текущий runtime | Contract | Что исправить до model fit |
|---|---|---|---|
| Universe | весь **уже отфильтрованный** turnover-band cycle journalled | conditional population boundary | ledger всех USDT contracts: included/exclusion reason, received-at, policy hash, instrument rules |
| Input | один closed cutoff, Min60/320 | `mexc_strategy_v2`, CycleEnvelope v3; version-dispatched frozen v2 evidence | future v3 получает новые types/config/parser/hash/evidence namespace; timeframe и thresholds не выбраны; event horizons проектировать отдельно от estimator/sample budgets |
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
  2 known collection warnings`; актуальный receipt `723 passed, 4 skipped,
  2 known PytestCollectionWarning (17.80s)` приведён в §0.
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
  `576 passed, 4 skipped, 2 known collection warnings`, затем `580 passed,
  4 skipped, 2 known collection warnings (14.99s)`; актуальный результат:
  `723 passed, 4 skipped, 2 known PytestCollectionWarning (17.80s)`.

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

Versioned evidence compatibility checkpoint `1971b77` также не меняет timeframe,
thresholds или runtime decision semantics:

- v2 parser/hash/contract logic зарегистрированы под literal
  `mexc_strategy_v2`, а evidence decoding dispatches по persisted version и
  fail-closed отвергает неизвестную или несовпадающую версию;
- frozen v2 contract hash
  `9c62b88b7804e9663bae6f0eb429c58c541680b61d307c4f16032cb0b62fe3dd`
  и default instance hash
  `9f0b2d7035c2a82ab1b6d8595245b8c3a7a8b9faad17bea8c57f6fcacb189466`
  остались неизменными;
- journal writer/restart audit и dataset/model export требуют одну точную
  StrategySpec identity на файл; model export переносит identity как metadata,
  не как numeric feature;
- полный receipt: **590 passed, 4 skipped, 2 known PytestCollectionWarning
  (18.95s)**. Это доказывает текущие regression-инварианты, но не абсолютную
  неизменяемость evidence, его внешнюю аутентичность или торговый edge.

Typed evidence checkpoint опубликован последовательностью `eb238b2`, `c541eea`,
`a604668`, `8569471`, `cf6bc01`, `9ef6b4f`, `bb1ca13`. Два финальных tips означают:

- `9ef6b4f` завершает строгие causal lifecycle transitions и transactional public
  strategy API на exact base/benchmark/HTF reads;
- `bb1ca13` вводит current writer schema v6, exact evidence-to-envelope binding,
  lifecycle chain, single-scanner runtime guard и read-only v5 compatibility;
- полный receipt: **723 passed, 4 skipped, 2 known PytestCollectionWarning
  (17.80s)**. Network, scanner, Telegram, model fit и private APIs не запускались.

Новый remaining order после checkpoint:

1. **[x] Superseded/выполнено:** ввести `StrategySpecV2`, dedicated YAML, один
   canonical instance hash и именованные source-timeframe bars. Текущие durations
   приведены в §0; behavioral semantics и v2 envelope закреплены `258c35f`.
2. **[x] Выполнено `1971b77`:** persisted `strategy_spec_version` выбирает
   version-specific evidence decoder/parser/hash; v2 hashes и frozen fixture
   сохранены. Future v3 обязан добавить новые types/config/parser/hash и отдельную
   evidence namespace. Ни timeframe, ни окна, ни thresholds v3 не выбраны.
3. **[x] Выполнено `9ef6b4f` + `bb1ca13`:** typed arm/observation/proposal
   lifecycle, exact per-symbol base/benchmark/HTF provenance, evidence-derived
   bar-source cycle timings, stale-evidence policy и single-scanner runtime guard.
   Universe/contract-details clocks остаются отдельным cycle provenance. Frozen
   v2 timeframe/windows/thresholds не менялись.
4. **Следующий ordered slice:** зафиксировать restart policy — explicit
   right-censor перед новой arm-гипотезой либо deterministic rehydration pending
   candidate из externally anchored v6 evidence.
5. Затем разделить persisted identity
   `MarketFeatureSnapshot`, `RuleEvaluation`, `StrategyProposal` и prediction;
   одновременно получать point-in-time instrument spec: contract size, quantity
   step, minimum quantity/notional, leverage rules, source timestamp и content hash.
6. Вычислять gate-independent causal snapshot и raw-universe ledger для каждого
   MEXC USDT contract до rule filtering. Пока missingness кодирует путь gates,
   model training запрещён.
7. Связать versioned proposal с executable single-position contract и labels:
   one entry/SL/TP, sizing, fees, spread, slippage, funding, horizon и concurrency=1.
8. Только после этого начинать prospective runtime-population collection,
   публиковать external checkpoints/manifests и ждать maturation labels.
9. Построить purged chronological evaluation с rules/no-trade/random/logistic
   baselines; только после frozen baseline receipt допускать LightGBM multiclass +
   separate payoff/EV head и challengers.
10. **Только после causal plumbing сравнить intended time hypotheses без
   калибровки thresholds:** текущий `mexc_strategy_v2` остаётся frozen Min60/45-hour
   control. Возможный быстрый вариант является отдельной v3 гипотезой; historical
   minute-scale clue не выбирает Min15 или значение ни одного поля.

Для каждого фактически используемого journal v6 или frozen-v5 prefix оператор
должен вынести `JournalCheckpointReceipt` за пределы той же перезаписываемой
runtime-директории.
Без этого chain остаётся `internally_consistent_unanchored`, а не trusted evidence.

Acceptance: worker count/order не меняет snapshot, ranking, entry или outcome —
**выполнено для cycle/cohort identity, entry timing и холодного старта
волатильности**; полная acceptance требует ещё безусловного trace (Phase 3),
пока поздние признаки зависят от того, где оборвался gate. Первый бар доступен
после `actionable_ts` **в смысле research-ranking**, не execution.

### Phase 2 — restart boundary + separate identities + point-in-time instrument spec

1. Зафиксировать поведение после restart: либо append-only right-censor active
   candidate до новой arm-гипотезы, либо deterministic pending-state rehydration
   только из externally anchored v6 evidence. Silent continuation запрещён.
2. Persist `MarketFeatureSnapshot`, `RuleEvaluation`, `StrategyProposal` и
   `ShadowPrediction` как разные versioned/hash-addressed objects. Rule action и
   proposal geometry не должны менять market-only identity.
3. Зафиксировать scoring instant и exact references между этими objects и уже
   реализованным typed lifecycle.
4. Получать point-in-time MEXC instrument spec: contract size, quantity step,
   minimum quantity/notional, leverage/rounding rules, request/receipt time и
   canonical content hash.
5. Любое отсутствие или несоответствие instrument spec для executable proposal
   должно давать abstain, а не default sizing.

Acceptance: restart не продолжает и не заменяет pending hypothesis без durable
right-censor/rehydration evidence; изменение rules не меняет
`MarketFeatureSnapshot` identity; изменение proposal geometry меняет только
proposal identity; один label в будущем может ссылаться на exact snapshot + rule
+ proposal + instrument-spec tuple.

### Phase 3 — unconditional causal feature parity + raw-universe ledger

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

### Phase 4 — executable single-position proposal and labels

1. Создать deterministic TradeProposal для всей proposal-eligible population.
2. Получить point-in-time instrument rules и frozen costs/sizing/horizon.
3. Связать journal с `replay_single_short()`.
4. Settlement funding брать из timestamped future series отдельно от decision
   funding feature.
5. Label хранить append-only отдельно от DecisionSnapshot.

Acceptance: один snapshot получает не более одного label; incomplete horizon не
становится win/loss; изменение любого contract параметра меняет hash.

### Phase 5 — collect + externally checkpoint prospective population

После ввода новой unconditional schema собрать новые point-in-time cycles и
дождаться maturation labels. Legacy CSV можно использовать только для discovery
и regression, но не для admission первого model candidate.

Acceptance: достаточно независимых time blocks/symbols/regimes; каждый training
row воспроизводится из immutable snapshot + proposal + contract + forward-data
manifest.

### Phase 6 — purged chronological baselines

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

### Phase 7 — LightGBM/EV shadow candidate

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

### Phase 8 — prospective shadow evidence

Собрать несколько независимых режимов и symbols в shadow. Promotion возможен
только при положительной нижней границе paired time/symbol uncertainty, устойчивой
calibration и cost stress, повторяемом rebuild и явном решении оператора.

### Phase 9 — challengers

- CatBoost — первый parity challenger.
- TCN — только causal left-looking windows и train-only normalization.
- Chronos — только frozen prospective predictions; historical backfill не
  считается независимым доказательством.
- LLM context — последняя ablation; numeric baseline сначала оценивается без неё.

### Phase 10 — model-assisted signals-only / paper

После явного promotion model policy может влиять только на signals-only/paper
ranking в контролируемом A/B против frozen rules: append-only decisions,
instant rollback, hard abstain и ручное promotion. Это отдельная ступень между
shadow и любым private execution.

### Phase 11 — private/live (отдельный будущий проект)

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
19. SAME_BAR lifecycle transition повторяет exact predecessor frame/candidate/
    price state; later transition монотонен, terminal event не имеет successor.
20. Base/benchmark/HTF evidence пересобирает frame и raw-bundle hashes; stale,
    malformed, failed, no-rows и not-requested outcomes не взаимозаменяемы.
21. Journal v6 отвергает forged input/snapshot/action, venue/symbol/cycle/timing
    drift, late source receipt, arbitrary extra timing и cross-cycle lifecycle link.
22. Второй scanner owner отклоняется до первого market request; duplicate cycle
    проверяется до мутации strategy state. V5 остаётся readable, но read-only и
    требует explicit legacy opt-in для model export.

## 11. Текущий operational verdict

- MEXC остаётся целевой биржей.
- Scanner/бот не запускался в этом этапе.
- Private API, Telegram, testnet/live не используются: ключи не ротированы.
- Старые CSV и tracked ML artifacts — legacy/discovery-only.
- Реализованный feature contract улучшает воспроизводимость, но **не доказывает
  edge** и пока не разрешает model fit.
- Phase 1 evidence foundation теперь включает current writer journal v6, frozen
  read-only v5, cycle identity v5, `CycleEnvelope v3`, typed lifecycle,
  exact per-symbol base/benchmark/HTF provenance, single-scanner runtime guard,
  `mexc_strategy_v2` и single-position v3. Hash-chain без вынесенного receipt
  остаётся unanchored. Это делает evidence строже, но **edge по-прежнему не установлен**.
- Executable tip `bb1ca13` поверх `9ef6b4f`: `723 passed, 4 skipped, 2 known
  PytestCollectionWarning (17.80s)`. Lifecycle pin `cc75c871…551`, frame-provenance
  pin `f4004ac9…dbf`, reversal-feature pin `20f9f61d…c496c`,
  version-dispatched v2 evidence и frozen behavioral locks входят в этот receipt.
- Этот receipt не делает локальный код или journal абсолютно неизменяемым и не
  аутентифицирует происхождение evidence: внешний checkpoint trust boundary
  сохраняется без изменений.
- Следующий порядок: explicit restart right-censor/rehydration policy → persisted
  `MarketFeatureSnapshot` / `RuleEvaluation` / `StrategyProposal` identities +
  point-in-time instrument spec → unconditional
  gate-independent snapshot и raw-universe ledger → executable single-position
  proposal/labels/costs → externally checkpointed prospective collection → purged
  chronological rules/random/logistic baselines → LightGBM + separate EV head и
  challengers. Min15 остаётся лишь отдельной будущей гипотезой; timeframe и
  thresholds v3 не выбраны. Пока gate path определяет missingness, model fit запрещён.

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
