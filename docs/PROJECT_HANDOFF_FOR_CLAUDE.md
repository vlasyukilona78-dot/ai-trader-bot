# Koteika Ultra: полный контекст проекта для Claude Code

Историческая основа: **2026-07-28**. Актуальное дополнение:
**2026-08-03, Europe/Moscow**.

Этот документ предназначен для новой сессии Claude Code. Его нужно прочитать
полностью до любых изменений, запусков, переключений веток или калибровки
стратегии.

> [!IMPORTANT]
> Начинать нужно с `docs/CLAUDE_REVIEW_PROMPT_2026-08-03.md`. Root/preservation
> состояние уточняет `docs/CURRENT_STATE_AND_AI_PLAN_2026-08-03.md`, а текущий
> MEXC AI/strategy источник истины находится в MEXC worktree:
> `docs/STRATEGY_AI_MASTER_PLAN_2026-08-03.md`, functional anchor `f0b43d6`.
> Все старые hashes, dirty counts, «ahead 9», требования повторно выбрать биржу
> и команды будущего времени ниже являются датированным контекстом.

Актуальные functional anchors:

- root/Bybit runtime: `2f7e18f`, опубликован в
  `origin/feat/phase2-layer1-pump-runtime-alignment`; `533 passed, 4 skipped`;
- target MEXC research AI foundation: `f0b43d6`, опубликован в
  `origin/claude/codex-project-review-04581e`; `352 passed, 4 skipped`, два
  известных collection warnings;
- MEXC подтверждён как целевая площадка, scanner остаётся public-data,
  signals-only, бот остановлен;
- проверенный same-disk checkpoint без USB:
  `C:\koteika-checkpoints\koteika_preservation_20260803_154657_88710bc3`,
  root `80e6f2b`, MEXC `1e91ce0`, manifest SHA-256
  `237204c9b629e48a60b185accf3f3f05491c84a43ca5f11c26e5bc950a0aec89`;
- no-edge verdict не изменён;
- текущие tips не содержат `.env`, но credentials не ротированы и история не
  очищена: private API/Telegram/testnet/live запрещены;
- следующий этап — Phase 1 time/spec contract, затем unconditional feature
  parity и лишь после них executable labels через frozen one-entry/one-stop/
  one-TP contract. LightGBM + отдельный EV-head является первым numeric
  candidate; CatBoost/XGBoost AFT и causal TCN/Chronos — challengers; LLM
  разрешён только как timestamped context extractor.

## 1. Самое важное в десяти пунктах

1. **Koteika Ultra** — криптовалютный сканер и торговый runtime для идеи
   `short on pump`: найти сильный памп, дождаться признаков истощения и слабости,
   выдать SHORT-сигнал, а затем сопровождать его до защитного или управляемого
   выхода.
2. Сейчас проект полезнее рассматривать как **исследовательскую и сигнальную
   систему**, а не как доказанно прибыльного торгового робота.
3. **Положительная математическая доходность не доказана.** Последнее честное
   исследование MEXC показало, что сигналы одной позицией статистически не
   отличались от случайных входов после издержек. Старые положительные выводы
   по DCA официально отозваны.
4. Бот должен оставаться **остановленным**. Локальная конфигурация — `paper`;
   рыночные данные настоящие, но заявки на биржу не отправляются.
5. В репозитории есть **две независимые, неслитые линии**:
   корневой Bybit-runtime и отдельный Claude/MEXC worktree. Их нельзя смешивать
   или воспринимать как последовательные версии одного дерева.
6. Корневой worktree содержит большую незакоммиченную разработку:
   35 изменённых tracked-файлов (`+3665/-315`) и содержательные untracked-файлы.
   Нельзя выполнять `reset`, `clean`, `checkout --`, массовое форматирование или
   удаление «мусора».
7. MEXC-ветка на 9 локальных коммитов впереди GitHub. Чистый clone увидит её
   только до `68e0ff7`, а итог `no-edge` находится в локальном `9f71a86`.
8. `.env`, локальные runtime-базы, журналы, исторические свечи и исследовательские
   CSV не переносятся через Git. `.env` когда-то присутствовал в истории Git,
   поэтому старые ключи и токены нужно считать потенциально скомпрометированными
   и ротировать, если это ещё не сделано.
9. Свежие проверки кода зелёные: корневой worktree — `529 passed, 4 skipped`;
   MEXC-worktree — `287 passed, 4 skipped`. Это доказывает техническую
   согласованность тестов, но **не доказывает edge или безопасность live**.
10. Последнее зафиксированное пользовательское решение: **целевая биржа —
    MEXC, а Claude/MEXC-ветка — база для дальнейшей реализации**. Перед новым
    большим изменением это решение всё равно нужно подтвердить с пользователем,
    потому что текущий активный root-runtime остаётся Bybit.

## 2. Что это за бот и зачем он нужен

Изначальная задача — не «шортить любую зелёную свечу», а находить ситуацию, где
вертикальный импульс уже выдыхается:

- цена быстро выросла и вышла в статистически растянутую область;
- объём, RSI, полосы, поток сделок и структура подтверждают сам памп;
- покупатели теряют инициативу: появляются отклонение от хая, слабые закрытия,
  failed reclaim, дивергенции или перелом структуры;
- вход всё ещё расположен достаточно близко к зоне инвалидации;
- старшие таймфреймы, стакан, спред, ликвидность и деривативный контекст не
  указывают на опасное продолжение squeeze;
- стоп и цель имеют корректную геометрию и проходят риск-фильтры.

Практический продукт состоит из четырёх частей:

1. **Сканер рынка** — автоматически отбирает ликвидные USDT perpetual-контракты
   и анализирует их параллельно.
2. **Сигнальный канал** — отправляет ранние `WATCH`/`SETUP`, основной `ENTRY`,
   графики, уровни, контекст и последующие `EXIT`-уведомления.
3. **Безопасный торговый runtime** — умеет работать в paper/demo/testnet/live,
   согласовывать локальное состояние с биржей, ограничивать риск и
   идемпотентно исполнять ордера. Live сейчас не разрешён.
4. **Исследовательский контур** — хранит решения и результаты наблюдений,
   строит датасеты, replay, отчёты качества и калибровочные сравнения.

Основной реальный сценарий пользователя — получать сигналы, при необходимости
входить вручную и получать уведомления о сопровождении. Поэтому оценка
стратегии должна моделировать **один реально открываемый вход**, если
пользователь явно не согласовал другую схему капитала. Нельзя подменять этот
сценарий скрытым усреднением/DCA.

## 3. Текущее состояние на 2026-07-28

### 3.1 Runtime

- Бот не запускался в ходе подготовки этого документа.
- Последние runtime-логи и рыночные события датированы 2026-07-25.
- Локальный `.env` задаёт:
  - `BOT_RUNTIME_MODE=paper`;
  - `BYBIT_TESTNET=false`;
  - `BOT_SYMBOLS=ALL_BYBIT_LINEAR_USDT`;
  - `CONCURRENT_TASKS=16`;
  - turnover-фильтр от 200 тысяч до 200 миллионов USDT за 24 часа;
  - только perpetual-контракты;
  - alerts включены;
  - `EARLY_PRE_MAIN_ENABLED=true`;
  - tracking сигналов и позиций включён;
  - MTF-контекст для Entry Gate обязателен;
  - online retraining выключен;
  - `RUNTIME_DB_PATH=data/runtime/v2_demo_runtime.db`.
- В `paper` код принудительно устанавливает `dry_run=True`; при
  `BYBIT_TESTNET=false` публичные данные берутся с Bybit mainnet, но ордера не
  отправляются.
- В `.env` также осталось `BYBIT_ENV=demo`, однако явный
  `BOT_RUNTIME_MODE=paper` имеет приоритет. Это устаревшая и сбивающая с толку
  переменная, а не фактический режим.
- Имя явно заданной DB также осталось `demo`, хотя runtime фактически `paper`.
  Это provenance drift: записи из неё нельзя называть demo-execution evidence,
  а перед следующими запусками пути paper/demo/testnet/live нужно развести.
- Последний сохранённый dynamic universe: 348 Bybit linear USDT perpetual
  инструментов после фильтра ликвидности, возраста листинга и качества.

### 3.2 Локальное состояние

Read-only проверка `data/runtime/v2_demo_runtime_main.db`:

- основной файл около 531 МБ, WAL около 47 МБ;
- SQLite `PRAGMA quick_check`: `ok`;
- 467 состояний, все `FLAT`;
- открытых inflight intents: 0; единственная строка имеет статус `completed`;
- duplicate `order_id`: 0;
- duplicate `order_link_id`: 0;
- exchange closures в этой базе: 0.

Это состояние локального runtime, а не независимая проверка биржевого аккаунта.

### 3.3 Свежая валидация

Выполнено без запуска торгового цикла:

```text
scripts/validate_install.ps1:
dependencies=ok
34 passed
Python 3.12.13

root worktree:
529 passed, 4 skipped, 2 collection warnings

Claude/MEXC worktree:
287 passed, 4 skipped, 3 warnings
```

Два общих warning относятся к тому, что pytest видит служебный класс
`TestnetValidationHarness` с `__init__` и не собирает его как тестовый класс.
Дополнительный warning MEXC-worktree — невозможность записать `.pytest_cache`
из-за прав; сами тесты прошли.

### 3.4 Накопленные сигнальные данные root

Локальный журнал содержит:

- 543 записанных кандидата/стратегических решения после интересующих стадий:
  531 `HOLD`, 1 `SHORT_ENTRY`, 11 `EXIT_SHORT`;
- 46 Entry Gate admissions: 1 approved, 45 rejected;
- 54 signal events: 34 `WATCH`, 3 `SETUP`, 1 `ENTRY`, 16 `EXIT`;
- 23 завершённых/просроченных наблюдения: 21 `completed`,
  2 `expired_incomplete`;
- среди 21 полных наблюдений TP был затронут 9 раз, SL — 16 раз, оба уровня —
  6 раз, ни один — 2 раза;
- 19 из 21 полных наблюдений относятся к `pre_main`.

Эти числа **нельзя превращать в win rate**: выборка мала, преимущественно
состоит из ранних сигналов, а факт касания обоих уровней внутри окна не всегда
восстанавливает реальный порядок исполнения. Она лишь подтверждает, что ранний
`WATCH` ещё не готов быть инструкцией немедленного входа.

### 3.5 Последний операторский вердикт калибровки

Для `logs/observation/comparison_pre_main_20260723_1954.json` 2026-07-28
повторно выполнен предписанный triage:

```text
VERDICT: pause_calibration
STOP_REASON: window_size_not_comparable
TOP_COMBINATION: htf_trend_ok + stretched_from_vwap + volatility_regime_ok
```

По `scripts/observation/RUNBOOK_triage.md` это non-comparable-window stop
(код 13). Практическое решение: **thresholds не менять**, собрать сопоставимое
по размеру окно и только затем сравнивать like-for-like. Этот вердикт важнее
желания увеличить частоту сигналов.

## 4. Две линии разработки

### 4.1 Корневой Bybit/runtime worktree

```text
path:   C:\Users\vlasy\PycharmProjects\koteika_Ultra
branch: feat/phase2-layer1-pump-runtime-alignment
HEAD:   0c38863523cc4cf0f021677f04d08349be0c3aca
date:   2026-05-24
```

Это текущая рабочая папка и наиболее функциональный Bybit-runtime. Её remote-ref
имеет тот же hash, но upstream для локальной ветки не настроен.

Незакоммиченное состояние:

- 35 modified tracked-файлов;
- staged-файлов нет;
- tracked diff примерно `+3665/-315`;
- основная разработка находится в `app/main.py` (`+1987/-199`);
- содержательные untracked-файлы включают новые trackers, тесты, launch/validate
  scripts и документацию;
- тысячи остальных untracked-файлов принадлежат резервной
  `.venv_relocated_20260723` и `.claude` worktrees.

Корень использует partial clone (`blob:none`) и sparse checkout. В sparse-набор
входят каталоги исходников, но `.github` не выведен в рабочее дерево, хотя
workflow-файлы существуют в Git. Не объявлять CI «отсутствующим» только потому,
что каталога не видно; его можно читать через `git show HEAD:.github/...`.

### 4.2 Отдельный Claude/MEXC worktree

```text
path:   .claude/worktrees/codex-project-review-04581e
branch: claude/codex-project-review-04581e
HEAD:   9f71a866b413bba4f1ab3a21603219dfe61f16fd
date:   2026-07-26
remote: origin/claude/codex-project-review-04581e = 68e0ff7
```

Эта ветка была создана от мартовского `origin/main`, а не от майской root-ветки.
Общий предок двух линий — `efac6eaf`. Между ними нет merge.

Ветка содержит:

- MEXC public market-data client и dynamic universe;
- исторический collector;
- построение pump datasets;
- causal/runtime dataset builder;
- pathwise replay и temporal validation;
- MEXC signals-only scanner `app/scan.py`;
- исправления turnover units, warm-up, label timing, purge и gate ordering;
- signal observation wiring;
- итоговый no-edge audit.

Она **не содержит готового MEXC private execution adapter**. Реальная торговля
на MEXC не реализована и не разрешена; `app/scan.py` — public-data,
signals-only путь.

Последние девять коммитов существуют только локально:

```text
8cc31fc fix look-ahead in the labelled dataset
ead2aa1 pathwise P&L replay and real temporal validation
aae0384 turnover units, purge leakage, feature warm-up
27aa443 MEXC signals-only scanner and request pacing
694055f execution ordering, fail-open gates, order-dependent thresholds
6cde693 runtime-pipeline dataset and observation wiring
f7352f2 volatility-scaled stop buffer and .env loading
f689e42 gate-calibration finding
9f71a86 no-edge finding; old expectancy claims retracted
```

Новый clone с GitHub не получит эти коммиты, ignored datasets или итоговое
исследование без отдельного сохранения/пуша.

### 4.3 `main` не является актуальной версией

Локальный и remote `main` остаются на мартовском `efac6ea`. Тег
`baseline-testnet-go-20260308-170045` исторически фиксирует «14 PASS,
0 blockers», но это не сертификат текущего кода и не разрешение live.

## 5. Активная архитектура root/Bybit

Упрощённый runtime-flow:

```text
main.py / start_bot_clean.ps1
    -> instance/supervisor lock
    -> app.bootstrap.load_runtime_config()
    -> BybitAdapter + MarketDataFeed + WS/poll reconciliation
    -> RuntimeStore + StateMachine + RiskEngine + ExecutionEngine
    -> dynamic symbol universe
    -> parallel OHLCV/ticker analysis
    -> FeaturePipeline
    -> LayeredPumpStrategy / Ultra fast path
    -> EntryGate
    -> RiskEngine approval and sizing
    -> ExecutionEngine (paper ignores real order placement)
    -> Telegram/Discord + charts + JSONL/SQLite telemetry
    -> observation and virtual-position tracking
```

### 5.1 Entry points

- `main.py` — Windows-friendly root launcher and dual-profile supervisor.
  Запуск без аргументов создаёт дочерние профили `main` и `early`.
- `app/main.py` — фактический V2 control-plane и торговый цикл.
- `app/bootstrap.py` — runtime modes, secrets/env, symbol discovery, Bybit
  adapter config и fail-closed startup validation.
- `scripts/start_bot_clean.ps1` — рекомендуемый переносимый launcher; по
  умолчанию запускает профиль `main`, умеет `-Once`, пишет отдельные логи.

Неактуальные/legacy entrypoints:

- `main_legacy_monolith.py`;
- `main.py.bak_working`;
- `trading_loop.py`;
- `engine/*`;
- часть старых корневых clients/trainers.

`core/*` нельзя целиком называть legacy: хотя control-plane находится в
`trading/*`, активный V2 импортирует из `core` индикаторы, market data,
feature engineering, volume profile, regime и `SignalGenerator`.

### 5.2 Market data

Активный Bybit feed умеет:

- OHLCV через Bybit V5;
- bulk ticker snapshot на весь universe;
- mark/last/index price и spread;
- funding из ticker;
- по финальному кандидату — forced refresh с long/short ratio, open interest,
  public trade flow, native 15m/1h frames и liquidation context;
- Coinglass heatmap при наличии отдельного ключа, с внутренним fallback;
- live-price overlay перед отправкой сигнала;
- data-quality checks и MTF readiness flags.

На первой массовой стадии дорогой derivatives context по умолчанию не
запрашивается для каждого символа; он принудительно обновляется перед
кандидатным решением. Это компромисс скорости и полноты.

Важное ограничение root: `MarketDataClient.fetch_sentiment_index()` существует,
но текущий `MarketDataFeed.fetch_frame()` его не вызывает. В фактических
decision events sentiment/news остаются `None`/unavailable. Не писать, что
реальный sentiment уже подключён к root, пока call site не появится и не будет
покрыт тестом.

### 5.3 Feature pipeline

Основные признаки:

- OHLCV и live mark overlay;
- RSI, EMA20/EMA50, MACD/histogram, ATR, ADX;
- Bollinger и Keltner;
- VWAP и distance from VWAP;
- OBV и proxy-CVD;
- volume spike;
- native/resampled MTF RSI и trend на 5m/15m/1h;
- volume profile POC/VAH/VAL;
- market regime;
- funding, OI, long/short ratio;
- spread, expected slippage, depth/imbalance и aggressor exhaustion;
- liquidation clusters и external heatmap для графика/контекста.

`FeaturePipeline` обязан отбрасывать unusable/non-finite/stale frames. Нельзя
заменять missing data нейтральным значением без явного quality/degraded flag.

### 5.4 Стратегия short on pump

Основной алгоритм реализован в:

- `core/signal_generator.py`;
- `trading/signals/layered_strategy.py`;
- `trading/signals/entry_gate.py`;
- `trading/signals/ultra_short_entry.py`;
- `trading/signals/ultra_v2.py`.

Логика слоёв:

1. **Regime filter** — запрещает опасный рыночный режим, сильный HTF-uptrend,
   недостаточную волатильность, отсутствие растяжения от VWAP или news veto.
2. **Layer 1: pump detection** — подтверждает clean pump, RSI/bands/volume и
   создаёт устойчивый `pump_id`.
3. **Layer 2: weakness confirmation** — ищет lower close/high, rejection,
   failed reclaim, OBV/CVD divergence, RSI/MACD rollover и слабость после пика.
4. **Layer 3: entry location** — проверяет близость к хаю/VAH/POC, sweep/reclaim,
   bearish MSB, fresh reaction и запрещает позднее преследование цены.
5. **Layer 4: fake/continuation filter** — использует VWAP, funding, crowding,
   OI, sentiment quality и structural override. При degraded data требует более
   сильную структуру.
6. **Layer 5: TP/SL geometry** — строит stop выше структурной инвалидации и
   ATR-buffer, цели по volume profile/support/RR и несколько partial targets.
7. **Entry Gate** — отдельный admission слой: валидирует price geometry, RR,
   stop distance в ATR, chase distance, полный MTF-контекст, continuation risk,
   spread/slippage/depth/imbalance и итоговый score.

Стратегия short-only по умолчанию: `allow_long_entries=False`; long возможен
только при явном `ENABLE_LONG_SIGNALS`.

`Ultra` — более быстрый путь раннего short-кандидата. Он не обходит Entry Gate:
если Ultra rejected, код продолжает проверять более консервативный layered
setup.

Генератор рассчитывает `partial_tps`, но активный execution engine не исполняет
настоящую многоступенчатую схему фиксации. В runtime остаётся один итоговый TP.
Не описывать partial targets как уже реализованное управление ордерами.

### 5.5 Профили сигналов

- `early` — `WATCH`/`SETUP`, наблюдение и уведомления. Профиль monitor-only:
  non-HOLD intent получает `early_profile_monitor_only`.
- `main` — подтверждённый `ENTRY`, риск-проверка, возможное исполнение в
  demo/testnet/live и managed exits.
- `both` — оба потока в одном процессе.
- Root `main.py` без аргументов запускает два отдельных дочерних процесса
  (`main` и `early`) под одним supervisor lock.

Два важных нюанса:

- настоящий обычный pre-main fallback требует
  `EARLY_PRE_MAIN_ENABLED=true` (либо legacy-флаг). Без этого обычный early-путь
  часто оформляет уже одобренный `SHORT_ENTRY`, а наиболее ранним независимым
  путём остаётся Ultra Radar/Confirm;
- early не выставляет ордер, но live-refresh кандидата переиспользует общий
  exchange snapshot/reconcile. Поэтому профиль observation-only по исполнению,
  но не полностью изолирован от чтения состояния биржи.

Не считать `WATCH` немедленной инструкцией входа. Открытый продуктовый вопрос:
должен ли `WATCH` быть только informational или ручной entry cue. До ответа
калибровать его как торговую сделку нельзя.

### 5.6 Risk, execution и state

`RiskEngine` проверяет:

- daily loss и consecutive-loss halt;
- cooldown;
- max concurrent positions;
- max total and per-symbol notional;
- stop-loss presence and geometry;
- confidence, turnover и spread;
- risk-based position sizing;
- leverage/notional cap;
- liquidation buffer;
- costs/slippage buffer;
- запрет pyramiding по умолчанию.

`ExecutionEngine` содержит:

- idempotency keys и journal;
- instrument-rule normalization;
- bounded retries;
- pre-entry orderbook depth/slippage guard;
- stop attachment;
- stale-order cleanup;
- restart recovery;
- polling/WS reconciliation;
- protective close и halt при невозможности защитить позицию;
- external intervention detection.

Критический недочёт: `LIVE_STARTUP_MAX_NOTIONAL_USDT` проверяется как
положительный и логируется, но **не применяется как фактический hard cap к
размеру заявки**. Поэтому существующие startup gates недостаточны для live.
Live нельзя включать, пока это не исправлено и не покрыто integration tests.

Exchange-side closures могут согласовываться обратно в risk state через closed
PnL reconciliation, но локальная demo-база пока не содержит записанных
`exchange_closures`; real-world путь всё равно должен пройти testnet evidence.

Liquidation-buffer также не является полноценной pre-trade моделью ликвидации:
если для нового входа точная liquidation price ещё неизвестна, эта конкретная
проверка проходит. Это ещё одна причина не трактовать наличие RiskEngine как
доказательство безопасности высокого плеча.

### 5.7 Persistence и observability

`RuntimeStore`/SQLite хранит:

- state records/transitions;
- inflight intents;
- order decisions;
- idempotency keys;
- signal admissions;
- risk session;
- exchange closures.

Дополнительные JSONL/JSON:

- `data/runtime/decision_events.jsonl` — pump-кандидаты и отклонённые решения;
- `data/runtime/signal_events.jsonl` — жизненный цикл сигнала;
- `signal_observations*.json*` — no-look-ahead оценка последующего движения;
- `signal_positions*.json*` — виртуальные/ручные позиции и exits.

`SignalObservationTracker` использует только уже появившиеся бары. Если символ
исчез из universe, `expire_stale()` переводит запись в
`expired_incomplete`; такую строку нельзя считать завершённой калибровочной
выборкой.

`SignalPositionTracker` — shadow/virtual сопровождение:

- основной доставленный SHORT может автоматически открыть виртуальную запись;
- ручную позицию можно зарегистрировать через
  `scripts/manual_signal_position.py`;
- managed exit и TP/SL barrier формируют уведомление;
- при конфликте TP и SL в одном баре применяется консервативное правило:
  считается, что первым сработал stop;
- закрытие виртуальной ручной позиции **не закрывает сделку пользователя на
  бирже**.

Alerts:

- Telegram и Discord;
- карточки с entry/TP/SL/confidence/reason;
- отдельные 1m и HTF-графики;
- symbol-copy button;
- dedupe/cooldowns и cross-process reservations;
- proxy autodetection;
- redaction токенов и proxy credentials из логов;
- отдельные выключатели обычных и Ultra alerts.

### 5.8 ML

В проекте есть старые `model_win`/`model_horizon`, scaler, model bundle,
governance и online-learning код.

Текущее правило:

- ML используется только в **shadow/advisory** режиме;
- prediction добавляется в telemetry и не меняет решение стратегии;
- в последних событиях artifact помечался `ungoverned_artifact`;
- `ML_ONLINE_RETRAIN_ENABLED=false`;
- auto-promotion выключен;
- нельзя дать ML право блокировать/разрешать сделки до temporal,
  out-of-sample и cost-aware доказательства на корректном feature contract.

Текущие bundles неполны: main содержит model-файлы без полноценного
manifest/registry, а early — manifest/registry без самих model-файлов. Кроме
того, startup dependency-check и фактическая shadow-загрузка используют разные
флаги (`ML_INFERENCE_ENABLED` и `ML_SHADOW_ENABLED`). Не считать ML
воспроизводимым или governed до сведения этого контракта к одному источнику.

## 6. Конфигурация и приоритет источников

### 6.1 `.env`

Содержит runtime mode, exchange credentials, Telegram credentials, universe,
профили, Entry Gate, risk/execution toggles. Файл локальный и не должен
печататься, цитироваться или коммититься.

### 6.2 `config/config.yaml`

Через `core/settings.py` реально питает часть:

- strategy settings;
- market-data URL/timeouts;
- ML paths/toggles;
- alert chart setting.

Важный config drift: блок `risk:` в YAML загружается в `AppSettings`, но
активный `trading/risk/engine.py` получает `RiskLimits` из env через
`trading/risk/limits.py`. Нельзя считать YAML risk limits фактическими runtime
лимитами.

При отсутствии risk env активны defaults загрузчика:

- risk per trade 1%;
- daily loss 3%;
- max leverage/notional-to-equity 3x;
- max 2 concurrent positions;
- per-symbol notional 25%;
- total notional 60%;
- four consecutive losses to halt;
- 15-minute cooldown;
- stop required;
- pyramiding disabled;
- modeled entry/exit fee 5.5 bps each и slippage buffer 6 bps.

Это кодовые defaults, а не рекомендация включать реальную торговлю.

### 6.3 Runtime modes

| Mode | Реальные данные | Реальные заявки | Назначение |
|---|---|---|---|
| `dry_run` | да | нет | безопасный smoke |
| `paper` | да | нет | наблюдение/сигналы |
| `demo` | Bybit demo | да | demo execution |
| `testnet` | Bybit testnet | да | integration validation |
| `live` | mainnet | да | запрещён до снятия blockers |

`demo` поддерживается кодом, хотя старый `config/README.md` перечисляет не все
режимы.

`paper` не является полноценным виртуальным брокером: dry-run adapter не хранит
долгоживущие exchange positions, fills и margin. Долгое сопровождение ведёт
отдельный `SignalPositionTracker`, причём виртуальная позиция создаётся только
после реально доставленного alert (`sent > 0`). Поэтому paper-телеметрию нельзя
автоматически приравнивать к broker PnL или точному портфельному backtest.

### 6.4 Воспроизводимость окружения

Текущее окружение работает на Python 3.12.13, но декларации зависимостей
расходятся:

- `requirements.txt`: `pandas>=2.2,<3.0`;
- `config/runtime_constraints.lock.txt`: `pandas==3.0.1`;
- `scripts/validate_install.ps1` запускает pytest, но `pytest` отсутствует в
  `requirements.txt`.

Значит, чистая установка только по `requirements.txt` пока не гарантирует тот
же validation result. До миграции нужно согласовать requirements/lock,
зафиксировать способ установки dev/test dependencies и повторить
`validate_install.ps1` плюс полный suite.

## 7. Карта репозитория

| Путь | Роль |
|---|---|
| `main.py` | root launcher/supervisor |
| `app/main.py` | главный runtime loop |
| `app/bootstrap.py` | modes/env/universe/startup validation |
| `app/testnet_validation.py` | testnet harness |
| `trading/exchange/` | Bybit HTTP/WS adapter и schemas |
| `trading/market_data/` | feed, snapshots, reconciliation |
| `trading/features/` | feature pipeline/validators |
| `trading/signals/` | strategy adapter, Entry Gate, Ultra, audits |
| `trading/risk/` | limits, sizing, liquidation, approval |
| `trading/execution/` | orders, idempotency, recovery |
| `trading/state/` | state machine, SQLite persistence, signal trackers |
| `trading/alerts/` | Telegram/Discord presentation |
| `core/` | активные алгоритмы: indicators, market data, signals, regime, VP |
| `ai/` | artifacts, shadow inference, train/online learning |
| `backtesting/` | старый backtest; в MEXC-ветке также новый replay/validation |
| `scripts/observation/` | наблюдение, анализ качества, comparison triage |
| `tests/` и `tests/v2/` | unit/e2e/safety regression suite |
| `data/runtime/` | локальные DB/JSONL/state; не коммитить |
| `logs/runtime/` | локальные runtime logs; не коммитить |
| `docs/AI_HANDOFF.md` | детальный исторический аудит, не текущий обзор |
| `.claude/worktrees/...` | отдельные Git worktrees; не считать обычными папками |

## 8. Что было сделано по этапам

### Февраль–март 2026

- исходный monolithic bot и модели;
- GitHub CI и training workflow;
- переход к V2 control-plane;
- state isolation, WS normalization, lifecycle hardening;
- пятислойная стратегия, diagnostics и sentiment fallback;
- startup reconciliation, RuntimeStore, RiskEngine, ExecutionEngine;
- observation audit payloads и health logging;
- calibration control и operator triage runbook.

### Апрель–май 2026

- online demo/early learning;
- liquidation map и Coinglass integration;
- clean signal cards и launch scripts;
- execution/risk/reconciliation hardening;
- signal- и exit-quality analyzers;
- ML/backtest/persistence improvements;
- отдельный Entry Gate, scoring и strategy versioning;
- replay audit;
- Ultra Short / Ultra V2;
- удаление `.env` из текущего tree и cleanup временных артефактов;
- committed root checkpoint `0c38863`.

Многие коммиты называются `Full project sync`; темы восстановлены по точному
дереву файлов, а не по хорошим commit messages.

### 22–25 июля: восстановление и root-доработки

- репозиторий перенесён/восстановлен на Windows;
- сохранены recovery artifacts:
  `.git_corrupt_20260723`, `.git_mixed_20260723`,
  `.venv_relocated_20260723`, `.venv_flash_20260723`,
  `recovery_snapshot_20260723.zip`;
- восстановлена рабочая `.venv`;
- сделан переносимый validate/start path;
- добавлена защита от второго экземпляра;
- параллельный full-universe scan, bulk tickers и progress logs;
- native MTF, trade flow, data readiness;
- улучшены Layer 1/Entry Gate/volume profile/MACD;
- early lifecycle и live refresh перед alert;
- полные signal cards и два графика;
- decision/signal journals;
- ML shadow;
- no-look-ahead observations;
- virtual/manual position tracker;
- managed/barrier exits и защита от повторных alerts;
- Telegram proxy/redaction и alert-disable tests.

Recovery directories нельзя удалять до отдельного, подтверждённого пользователем
архивирования. Они не являются актуальным исходным кодом.

### 25–26 июля: MEXC и независимый аудит

- выбран MEXC как предполагаемая целевая биржа;
- создан public MEXC data/universe/history stack;
- перестроены части Layer 1/3/5;
- построены pump datasets и первоначальная DCA-аналитика;
- добавлены market-relative volatility, liquidity/lot filters, structure,
  multi-timeframe features и scanner;
- Codex провёл независимый аудит причинности, labels, turnover, replay,
  temporal validation и runtime wiring;
- проблемы были исправлены серией локальных коммитов;
- финальный анализ показал отсутствие доказанного edge для фактически
  протестированной generic pump-fade логики.

## 9. Итог исследования MEXC и отменённые выводы

### 9.1 Что сначала выглядело положительно

Runtime calibration:

- 31 295 candidate rows / 278 symbols;
- после исключения equity/commodity proxies и cadence gaps:
  26 887 rows / 226 symbols / 133 days;
- DCA-модель показывала положительный mean и высокий win rate.

### 9.2 Почему это оказалось ложным

Положительность возникала из учёта до шести добавлений через каждые 8% и
нормализации результата на deployed capital. Усреднение проигрывающей позиции
и выход на 3% ниже blended entry создавали martingale-like высокий win rate,
который не соответствовал фактическому ручному использованию сигнала.

При replay как **одна позиция, один вход, без усреднения**:

- все 28 комбинаций target 2/3/5/8% и stop 3–50% были отрицательными;
- лучший mean был примерно `-0.14%`;
- использованный round-trip cost — `0.217%`;
- сигналы статистически не отличались от random entries на тех же
  symbols/windows;
- доверительные интервалы пересекали отсутствие преимущества.

Для 338 rule-filtered test signals uncensored adverse excursion до 3% target:

```text
p50  3.1%
p90 27.0%
p95 45.7%
p99 118.0%
max 132.9%
```

Следовательно, старые claims о `+0.0208 expectancy`, прибыльном filtered
portfolio, безопасном DCA и «100% recovery» нельзя повторять. Их отменяет
commit `9f71a86`.

### 9.3 Что именно было протестировано

No-edge относится к generic band-breakout + RSI + volume pump fade, а не к
полной задуманной технике. В протестированном decision path:

- Fibonacci не участвовал;
- overhead level gate был disabled;
- weakness layer был disabled;
- liquidation map влиял на картинку, а не на decision;
- `require_confluence` и `min_rsi_1h` были объявлены, но не потреблялись.

Корректный вывод: **текущая реализация не доказала edge; полный задуманный
набор условий ещё не был честно реализован и протестирован**. Это основание
для нового causal исследования, но не повод считать будущую версию прибыльной
заранее.

### 9.4 Найденные исследовательские ошибки

- исходный часовой look-ahead: kline timestamp был open time, а решение известно
  только после close;
- отсутствие полного 48h purge/embargo между train/validation/test;
- смешение 3% labels с 5% reward;
- fixed-horizon MAE/MFE смешивались с trade-path DCA labels;
- intrabar high/low ordering;
- gap stop fills по недоступной цене;
- Min1/Min5/Min60 thresholds применялись к разным горизонтам;
- MEXC `vol` — contracts, а не USDT turnover; точный `amount` отбрасывался;
- current-universe survivorship/future-selection bias;
- недостаточный 7d/30d warm-up;
- cadence gaps трактовались как непрерывные bars;
- tokenized equities/ETF попадали в crypto population;
- threshold search был невоспроизводим;
- gates калибровались на уже отобранной event population.

Любая новая калибровка должна устранить эти классы ошибок до обсуждения
результата.

## 10. Подтверждено и не подтверждено

### Подтверждено кодом и тестами

- безопасные paper/dry-run defaults;
- state machine, persistence, reconciliation и restart recovery;
- idempotency/retry/protective execution paths;
- dynamic Bybit universe и parallel scan;
- 5-layer diagnostics, Entry Gate и Ultra fast path;
- MTF/data-quality guards;
- risk sizing and limits;
- rich alerts, dedupe и charts;
- decision/signal journals;
- no-look-ahead signal observations;
- shadow/manual position tracking и managed exit notifications;
- ML shadow не управляет admission;
- public MEXC signals-only research path;
- обе рабочие линии проходят свои test suites.

### Не подтверждено

- положительная expectancy;
- реальный channel PnL/Sharpe/equity curve;
- безопасность высокого leverage;
- корректность `WATCH` как immediate entry;
- production-grade MEXC execution;
- partial TP execution;
- фактический hard cap `LIVE_STARTUP_MAX_NOTIONAL_USDT`;
- current root sentiment/news feed;
- testnet readiness текущего dirty root после настоящего order lifecycle;
- переносимость ignored datasets/caches;
- применимость старых thresholds к текущей runtime population.

## 11. Блокеры и открытые вопросы

### P0 — до любой новой разработки

1. Сохранить root dirty worktree, девять локальных MEXC-коммитов и нужные
   ignored research datasets. Не полагаться на GitHub.
2. Не запускать и не включать live.
3. Не повторять отозванные DCA/expectancy claims.
4. Подтвердить с пользователем, что MEXC остаётся целевой площадкой и что
   продолжать нужно от локального `9f71a86`.
5. Убедиться, что все исторические API/Telegram credentials ротированы.

### P1 — перед новым исследованием

1. Сформулировать точный торговый контракт:
   - один вход или DCA;
   - sizing basis;
   - fixed stop/managed stop;
   - TP и partial exits;
   - leverage/margin mode;
   - fees, spread, slippage, funding;
   - concurrency/capital occupancy;
   - horizon и success definition.
2. Реализовать **фактически задуманную** стратегию:
   - causal Fibonacci/levels;
   - overhead resistance gate;
   - weakness confirmation;
   - liquidation context в decision, а не только chart;
   - confluence и 1h RSI consumption;
   - explicit closed-bar contract.
3. Построить point-in-time universe и exact MEXC quote turnover (`amount`).
4. Валидировать cadence и warm-up каждого timeframe.
5. Группировать одинаковые decision timestamps и делать global 48h
   purge/embargo.
6. Сравнивать rule с random/naive baselines и считать symbol-clustered
   confidence intervals.
7. После comparison JSON всегда использовать prescribed calibration triage.

### P1 — перед live/testnet

1. Применить реальный per-order cap вместо одной проверки
   `LIVE_STARTUP_MAX_NOTIONAL_USDT > 0`.
2. Добавить явный проверяемый операторский emergency stop. Внутренние
   `HALTED`/circuit-breaker/protective-close полезны, но не заменяют аварийную
   кнопку.
3. Свести risk config к одному источнику истины.
4. Сделать observation collector fail-closed в `paper`: сейчас он наследует
   внешний runtime mode и в demo/testnet/live способен инициировать действия
   соответствующего режима.
5. Исправить testnet verdict: сценарии со статусом `SKIP` сейчас не попадают в
   blockers, поэтому прогон без `--execute-orders` может показать `testnet=GO`,
   хотя order lifecycle фактически не проверялся.
6. Учесть, что testnet harness по умолчанию перезаписывает tracked
   `config/runtime_constraints.lock.txt` результатом `pip freeze`.
7. Реализовать и протестировать MEXC private adapter, только если MEXC
   подтверждён.
8. Пройти настоящий tiny-notional lifecycle, partial fill, stop attach,
   restart, WS chaos и intervention tests.
9. Получить явное пользовательское согласование monetary limits и emergency
   stop.

### P2 — maintainability

- разбить `app/main.py` (сейчас более 10 тысяч строк) на scanner, lifecycle,
  alerting, tracking и orchestration services;
- убрать config/docs drift;
- улучшить commit messages и разделить локальный шум от исходников;
- архивировать recovery artifacts после подтверждения;
- вывести `.github` в sparse checkout, если нужно редактировать CI;
- убрать legacy modules из обычного runtime tree после отдельной проверки.

## 12. Рекомендуемый следующий технический этап

Если пользователь подтверждает MEXC, следующий этап должен быть не очередной
подбор порогов, а **causal implementation parity**:

1. Зафиксировать executable single-position contract.
2. На MEXC signals-only ветке реализовать недостающие intended features и
   closed-bar input contract.
3. Сохранить точные provenance manifests: venue, contract, universe snapshot,
   request failures, timeframe coverage, feature schema, costs и commit hash.
4. Построить runtime-population dataset, а не event-conditioned dataset.
5. Выполнить purged chronological evaluation против random baseline.
6. Принять результат, включая повторный no-edge, без подгонки test set.
7. Только если edge устойчив и воспроизводим — обсуждать paper observation.
8. Private execution и live остаются отдельным последующим проектом.

## 13. Безопасные команды для ориентации

Ни одна команда запуска бота не должна выполняться автоматически.

```powershell
# Сначала только состояние
git -c safe.directory=C:/Users/vlasy/PycharmProjects/koteika_Ultra status --short --branch
git -c safe.directory=C:/Users/vlasy/PycharmProjects/koteika_Ultra worktree list
git -c safe.directory=C:/Users/vlasy/PycharmProjects/koteika_Ultra branch --all --verbose

# Проверка установки без trading loop
powershell -ExecutionPolicy Bypass -File .\scripts\validate_install.ps1

# Полный root suite
.\.venv\Scripts\python.exe -m pytest -q
```

Основной launcher, **только по явной просьбе пользователя**:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_bot_clean.ps1 -SignalProfile main
powershell -ExecutionPolicy Bypass -File .\scripts\start_bot_clean.ps1 -SignalProfile early
```

Один scan-cycle всё равно может обратиться к публичным API и отправить alerts,
поэтому `-Once` также не запускать без согласования.

`scripts/observation/collect_observation_window.ps1` также нельзя считать
безопасным только из-за слова observation: скрипт многократно запускает runtime,
но сам не принуждает `paper` и не отключает alerts/orders. Перед разрешённым
окном режим и внешние действия нужно задать fail-closed явно.

Для готового calibration comparison JSON обязательно:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\observation\triage_calibration_result.ps1 -ComparisonJson <comparison.json>
```

Exit codes и полный flow описаны в
`scripts/observation/RUNBOOK_triage.md`.

Ручная shadow-позиция:

```powershell
.\.venv\Scripts\python.exe .\scripts\manual_signal_position.py status BTCUSDT
```

Команды `open`/`close` меняют локальное tracker state и должны выполняться
только по явной просьбе пользователя.

## 14. Правила работы для новой Claude Code

1. Всегда сначала читать `AGENTS.md`, `CLAUDE.md`, этот документ и релевантную
   часть `docs/AI_HANDOFF.md`.
2. Сразу фиксировать:
   - cwd/worktree;
   - branch и HEAD;
   - `git status`;
   - runtime state (`stopped/paper/demo/testnet/live`);
   - какие выводы являются code facts, а какие inference.
3. Не раскрывать содержимое `.env`, токены, ключи, proxy credentials или
   приватные endpoints.
4. Не очищать и не «нормализовать» dirty tree.
5. Не редактировать одновременно root и MEXC-worktree без явного плана переноса.
6. Не утверждать, что тесты доказывают прибыльность.
7. Не оптимизировать thresholds по test set и не выбирать красивую метрику
   задним числом.
8. Не использовать DCA-модель для оценки single-entry сигнала.
9. Не включать ML admission или auto-promotion.
10. Не запускать бот, observation window, testnet orders или alerts без запроса.
11. Перед live требовать отдельное явное разрешение, hard monetary caps и
    доказательства testnet lifecycle.
12. После материальной работы обновлять handoff:
    - worktree/branch/HEAD;
    - изменённые файлы и причина;
    - tests/evidence;
    - runtime state;
    - unresolved findings;
    - следующий рекомендуемый шаг.

## 15. Что нужно перенести при смене аккаунта/машины

Если меняется только Claude-аккаунт на этом же компьютере, локальные файлы
останутся, а новый Claude должен открыть корень проекта и прочитать
`CLAUDE.md`.

Если создаётся новый clone или меняется компьютер, GitHub **недостаточно**.
Нужно отдельно и безопасно сохранить:

- root dirty diff и содержательные untracked source/tests/docs;
- локальные commits `8cc31fc..9f71a86`;
- итоговый handoff MEXC-worktree;
- нужные ignored MEXC datasets и history caches вместе с hash/manifests;
- локальные runtime observations, если они нужны для анализа;
- `.env` через безопасный секретный канал, не через Git/чат;
- сведения о том, какие credentials уже ротированы.

Во вложенных `.claude/worktrees` обнаружены ещё две локальные `.env`. Их
содержимое не выводилось. Они исключены локально, но локальные exclude-правила
не являются переносимой защитой: нельзя копировать worktree целиком или делать
слепой `git add -A`.

Не переносить как «исходники»:

- `.venv*`;
- `.git_corrupt*`/`.git_mixed*`, если уже сделан проверенный архив;
- `.claude/settings.local.json`;
- runtime logs/caches без явной исследовательской необходимости.

До осознанного checkpoint/commit/push перенос проекта нельзя считать
завершённым.

## 16. Источники истины

При конфликте описаний приоритет:

1. фактический код и tests выбранного worktree;
2. текущий `git status`/HEAD;
3. этот документ;
4. `docs/AI_HANDOFF.md` и handoff внутри MEXC-worktree;
5. README/runbooks;
6. старые комментарии, commit messages и временные отчёты.

`docs/AI_HANDOFF.md` содержит ценный подробный аудит, но это исторический журнал
нескольких срезов. Его ранние положительные DCA/expectancy разделы отменены
финальным `Claude no-edge finding`.

---

Короткая формула проекта для новой нейросети:

> Мы строим безопасную, проверяемую систему поиска истощения пампа и SHORT-входа,
> но текущий generic pump-fade не доказал преимущества после издержек. Сначала
> нужно сохранить две локальные линии, реализовать полный задуманный causal
> decision path и честно проверить single-position торговый контракт. До этого
> бот остаётся остановленным, paper-only и без live-риска.
