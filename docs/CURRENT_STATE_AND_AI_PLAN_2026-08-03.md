# Koteika Ultra — текущее состояние и план causal AI

Актуально: **2026-08-03, Europe/Moscow**.

Это основной документ для продолжения работы новой нейросетью. Он имеет приоритет
над датированными preservation-планами и историческим `docs/AI_HANDOFF.md`.

## 1. Короткая формула проекта

Koteika Ultra — исследовательская система, которая ищет истощение резкого пампа
криптовалюты и возможный SHORT-разворот. Целевая площадка подтверждена:
**MEXC futures**. Текущая MEXC-линия использует только публичные рыночные данные,
формирует сигналы и журналирует всю наблюдаемую популяцию; она не умеет отправлять
приватные MEXC-ордера.

Бот не «чувствует» рынок и не должен обещать предсказание разворота. Его допустимая
задача — в момент принятия решения оценить калиброванные вероятности нескольких
исходов и ожидаемый PnL по заранее замороженному торговому контракту.

Главная исследовательская правда: generic pump-fade **не показал устойчивого edge
после издержек**. Ранее опубликованные положительные DCA/expectancy-выводы были
отозваны после устранения look-ahead и замены условного score на pathwise PnL.
Повторный no-edge является допустимым итогом; подгонять модель до положительной
цифры запрещено.

## 2. Неподвижные границы безопасности

- Бот остановлен. Рабочее состояние — research / signals-only / paper.
- Live, testnet, приватные API и отправка Telegram сейчас запрещены.
- `.env` удалён из актуальных root и MEXC tips и игнорируется, но существовал в
  Git-истории и других refs. История не очищена, credentials не ротированы по
  решению пользователя.
- Старые Bybit keys, Telegram token и proxy credentials считаются потенциально
  скомпрометированными до ротации. Их нельзя читать, печатать, переносить в prompt,
  вызывать или использовать для проверки доставки.
- Публичные MEXC-запросы без credentials, offline-replay и локальные тесты
  допустимы. Автоматически запускать scanner нельзя: даже signals-only запуск
  является внешним сетевым действием и может затронуть alerts/config.
- Никакая ML/LLM-модель не получает право менять entry, sizing, stop, TP или
  execution, пока не пройдёт shadow-валидацию на будущих данных.

## 3. Две линии разработки

| Линия | Назначение | Functional anchor | Remote | Состояние |
|---|---|---|---|---|
| Root | сохранённый Bybit/runtime Phase 2, observation и operational tooling | `2f7e18f` | `origin/feat/phase2-layer1-pump-runtime-alignment` = `2f7e18f` | `533 passed, 4 skipped, 3 warnings` |
| MEXC | целевая causal research/signals-only реализация | `98217df` | `origin/claude/codex-project-review-04581e` = `98217df` | `340 passed, 4 skipped, 2 warnings` |

Docs-only коммиты после этих anchors могут изменить tip, но не меняют указанные
functional checkpoints. `main` и root/Bybit не следует смешивать с MEXC без
отдельного transfer plan.

### Что реализовано в root

- Phase 2/V2 control plane, параллельное наблюдение universe, MTF и live-refresh
  перед сигналом.
- Persistent `SignalObservationTracker` и `SignalPositionTracker`.
- Entry Gate, layered strategy, feature/market-data pipeline, Telegram client,
  metrics и observation tooling.
- Portable Windows install validation и bounded keep-awake helper.
- Проверенный backup-инструмент с двумя честно различимыми режимами.

### Что реализовано в MEXC

- Исправлены look-ahead в labels, turnover units, warm-up, purge, execution
  ordering, pathwise PnL replay и research verdict.
- Публичный MEXC signals-only scanner с pacing и полным universe.
- Один causal cutoff на цикл, только закрытые base/HTF bars, last closed close
  вместо live ticker, freshness/cadence/OHLCV validation.
- Полный population journal: по одной записи на каждый point-in-time symbol,
  включая HOLD и ошибки; стабильные SHA-256 IDs, полный цикл, dedup и безопасные
  error codes без exception text.
- Thread-safe strategy state и подтверждённая, а не предполагаемая, delivery
  semantics.
- Исполняемый research-контракт одной SHORT-позиции в
  `backtesting/single_position.py`.

## 4. Preservation и работа без внешнего диска

### Подтверждённая disaster-resilient копия

Существует проверенная копия на отдельном физическом USB/FAT32-диске:

```text
D:\koteika-preservation\koteika_preservation_20260803_135615_6d24c806
marker: VERIFIED_OK.txt
payload: 2387 files / 2,840,300,975 bytes
MANIFEST_SHA256.csv:
3f86780a487d2f6f02e3603d83e741e5665fb0f0cdc91cb6cadaa1f71274a075
source root HEAD: 0c38863523cc4cf0f021677f04d08349be0c3aca
source MEXC HEAD: 9f71a866b413bba4f1ab3a21603219dfe61f16fd
```

Эта копия зафиксировала состояние **до** новых commits и поэтому не заменяет Git
для `2f7e18f`/`98217df`. Она не содержит `.env`, но была создана до ужесточения
исключения runtime logs и должна считаться чувствительной: не публиковать и не
подключать к недоверенной машине.

### Штатный режим без USB

`scripts/preservation/create_verified_backup.ps1` поддерживает:

- `LocalCheckpoint` — копия на том же физическом диске, маркер только
  `CHECKPOINT_VERIFIED.json`;
- `DisasterResilient` — требует другой physical disk либо non-local UNC, маркер
  только `VERIFIED_OK.json`.

Локальный checkpoint защищает от ошибочного Git-действия и порчи worktree, но не
от отказа SSD, кражи или шифровальщика. Он является нормальным ежедневным режимом,
поэтому внешний диск больше не обязателен для продолжения разработки.

Preflight без записи:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\preservation\create_verified_backup.ps1 `
  -Mode LocalCheckpoint `
  -BackupBase C:\koteika-checkpoints `
  -RootPath C:\Users\vlasy\PycharmProjects\koteika_Ultra `
  -MexcPath C:\Users\vlasy\PycharmProjects\koteika_Ultra\.claude\worktrees\codex-project-review-04581e `
  -PythonPath C:\Users\vlasy\PycharmProjects\koteika_Ultra\.venv\Scripts\python.exe `
  -PreflightOnly
```

Инструмент запрещает destination внутри Git-worktree, не перезаписывает старые
runs, исключает `.env*`, `.git`, virtualenv/recovery/cache/IDE, runtime system log,
raw WAL/SHM и stale runtime lock. SQLite сохраняется через Online Backup API.
Source/destination перечитываются с SHA-256, Git-state проверяется повторно, а
success marker записывается атомарно последним. При активном Python процесса
проекта скрипт останавливается.

## 5. Замороженный single-position contract v1

`98217df` создаёт независимый от старого DCA модуль. Это механика будущих labels,
а не торговая рекомендация.

Контракт требует:

1. Только SHORT.
2. Ровно один market-entry на open первой полной свечи с
   `bar.open_ts == decision_ts`.
3. Ровно один абсолютный stop и один абсолютный take-profit, известные в момент
   решения; DCA отсутствует.
4. Полный gap-free forward horizon из закрытых свечей.
5. Stop имеет приоритет, если одна OHLC-свеча задела stop и TP.
6. Gap через stop исполняется по худшему open; gap через TP не получает
   оптимистичное улучшение цены. Если уровень уже пройден до entry, setup получает
   `unfilled`, а не искусственную прибыль.
7. Quantity ограничена risk budget, max notional, max leverage, quantity step,
   min quantity и min notional.
8. Entry/exit fees, half-spread и directional slippage заданы явно.
9. Funding задаётся timestamped событиями. Положительная ставка платит SHORT;
   для внутрисвечного выхода не засчитывается funding, который мог произойти
   позже выхода.
10. Глобальная concurrency равна одному. Одновременные кандидаты выбираются по
    causal score с детерминированным tie-break; позиция блокирует новые входы до
    своего выхода.

Результат содержит fills, exit reason/time, quantity/notional/risk budget, gross
PnL, fees, funding, net PnL, return on notional и return on risk.

## 6. Какой ИИ выбран

### Champion baseline: LightGBM

Первым кандидатом должен быть небольшой CPU **LightGBM**, а не Kimi/ChatGPT и не
большая нейросеть. Причины:

- вход — преимущественно табличные causal features с пропусками и нелинейными
  взаимодействиями;
- реальная runtime-population сначала будет относительно небольшой;
- обучение и inference воспроизводимы локально и не требуют API key;
- низкая latency позволяет оценивать весь universe;
- feature importance/SHAP и простые ablations облегчают поиск leakage;
- сильный табличный baseline нужен до сравнения с более сложными моделями.

Существующий `ai/train.py` **не является новым trainer**: он обучает исторические
`target_win/target_horizon`, предпочитает XGBoost в `auto`, использует простой
80/20 split и калибрует probability на test-блоке. Его нельзя подключать к
решению. Новый pipeline должен быть отдельным и fail-closed.

### Цели модели

Для каждого point-in-time symbol/cutoff модель должна оценивать как минимум:

- `P(TP first)`;
- `P(SL first)`;
- `P(timeout)`;
- условный и безусловный `net_pnl_quote`/`return_on_risk` после всех costs;
- calibrated uncertainty / abstain, когда вход вне обученной области.

Практический первый вариант — три competing-risk probability outputs плюс
отдельный robust regressor для net EV. Торговое ранжирование использует только
out-of-fold calibrated outputs; сырой training prediction запрещён.

### Challenger-модели, но не сразу

1. **Causal TCN** на коротких последовательностях закрытых candles — первый
   sequence challenger после устойчивого LightGBM baseline.
2. **Chronos-2** — отдельный foundation-model shadow experiment для возвратов и
   quantiles, без прямого права на entry.
3. **TimesFM** можно оставить исследовательской альтернативой, если его forecast
   target и latency будут сопоставимы с Chronos.
4. **DeepLOB** допустим только после появления собственного point-in-time L2
   order-book collector. Строить его из OHLCV нельзя.

### Почему не Kimi K3 в торговом hot path

Kimi K3 — крупная reasoning/agent model с огромным контекстом, а не модель
микроструктурного прогноза. Open weights всё равно слишком велики для практичного
локального low-latency inference на этом компьютере; cloud-вызов добавляет latency,
стоимость, недетерминизм и внешний failure mode. Языковая убедительность не равна
калиброванной вероятности разворота.

Если позже понадобится новостной контекст, LLM разрешён только как асинхронный
context extractor:

```text
timestamped public news/text
→ strict JSON event extraction
→ age/source/confidence validation
→ numeric context features
→ LightGBM shadow input
```

Для массовой extraction-задачи разумнее Kimi K2.6 non-thinking либо компактная
high-volume модель; Kimi K3 — только редкий offline escalator для сложного
документа. Альтернативно можно использовать локальный FinBERT для sentiment.
Никакому cloud LLM нельзя передавать `.env`, приватные позиции, account state,
полную proprietary strategy или право генерировать entry/sizing/stop/TP.

## 7. Идеальная последовательность реализации

### Phase A — preservation и causal input contract — выполнено

- обе линии закоммичены и опубликованы;
- создан и проверен portable backup-tool;
- scanner принимает только закрытые bars с общим cutoff;
- runtime population journal полон и дедуплицирован;
- single-position mechanics заморожены и протестированы.

### Phase B — versioned label builder — следующий кодовый этап

1. Привязать каждый population row к последующим Min5 bars и timestamped funding.
2. Воспроизводить outcome только через `SinglePositionContract` v1.
3. Сохранять contract/config hash, feature schema hash, universe snapshot ID,
   input hash и data-quality flags.
4. Не удалять HOLD/error/no-data строки до явной политики sampling.
5. Добавить hand-calculated fixtures, boundary/gap/funding tests и повторный build
   с byte-stable output.

### Phase C — сбор runtime-population

Сначала controlled signals-only observation. До запуска отдельно проверить config,
alerts disabled и отсутствие private adapters. Контролировать:

- полный cycle size и dedup rate;
- долю `no_data`, stale/gap/data errors;
- point-in-time universe coverage;
- warm-up отдельно для base/HTF;
- availability и возраст funding/OI/liquidation context;
- schema/config drift.

### Phase D — честная оценка rules и random

- expanding chronological folds;
- purge/embargo не меньше label horizon (48h) плюс одна base-bar boundary;
- scaler/imputation/feature selection fit только на train;
- probability calibration только на более позднем past-only validation block;
- untouched future test один раз на зафиксированный кандидат;
- symbol-clustered confidence intervals;
- paired random-entry и rules-only baselines на той же population/concurrency;
- cost stress: fees, spread, slippage и funding хуже базового сценария;
- метрики: net EV, CI, profit factor, drawdown/tail, coverage, Brier/log loss,
  calibration error и PR-AUC, а не только win rate/AUC.

### Phase E — LightGBM shadow

Модель не создаёт сделку. Она пишет versioned score рядом с rules decision.
Promotion разрешён только если на нескольких untouched future folds одновременно:

- нижняя граница symbol-clustered CI net EV выше нуля;
- результат лучше paired random и rules-only после costs;
- calibration не деградирует по времени и regime;
- edge выдерживает cost/data-quality stress;
- достаточны число независимых symbols, событий и календарная длительность;
- повторный build/retrain даёт тот же вывод.

### Phase F — challengers и LLM context

Только после Phase E сравнить causal TCN/Chronos по тому же frozen evaluation.
Добавление LLM-контекста — отдельная ablation: baseline должен быть посчитан без
него, timestamps должны доказывать доступность текста до decision.

### Phase G — private execution/live — отдельный будущий проект

До него нужны rotation/history remediation, MEXC private adapter, testnet-like
execution harness, exchange filters/rounding, idempotency, hard monetary caps,
emergency stop, reconciliation, alert isolation и повторяемый edge. Общее
разрешение менять код не является разрешением рисковать капиталом.

## 8. Acceptance и STOP-условия

Немедленно остановить развитие модели, если:

- feature/label использует forming bar или данные позже `decision_ts`;
- dataset снова event-conditioned вместо runtime-population;
- одинаковый cutoff даёт разные causal IDs без изменения input/config;
- split смешивает label horizon с будущим fold;
- calibration или threshold выбирается по test;
- model comparison использует разные costs, universe или concurrency;
- LLM output напрямую влияет на ордер/risk;
- положительный результат исчезает при clustered CI или cost stress;
- для продолжения требуется читать старые credentials.

## 9. Первые конкретные задачи следующей сессии

1. Добавить versioned label-builder вокруг `replay_single_short()`.
2. Зафиксировать, откуда на decision-time берутся stop/TP для каждого HOLD-кандидата;
   если levels отсутствуют, строка не должна получать выдуманный label.
3. Добавить point-in-time funding series и точные MEXC instrument rules
   (`quantity_step`, minima) в dataset contract.
4. Исправить training/evaluation pipeline: purged folds, separate calibration,
   LightGBM first, random/rules baselines, clustered CI.
5. Только затем накопить observation window и обучить shadow candidate.

## 10. Первичные технические источники для выбора моделей

- LightGBM paper: <https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html>
- Causal TCN evaluation: <https://arxiv.org/abs/1803.01271>
- DeepLOB: <https://arxiv.org/abs/1808.03668>
- Chronos: <https://github.com/amazon-science/chronos-forecasting>
- TimesFM: <https://github.com/google-research/timesfm>
- FinBERT: <https://arxiv.org/abs/1908.10063>
- Kimi model selection: <https://www.kimi.com/help/kimi-api/api-model-selection>
- Kimi capabilities: <https://www.kimi.com/help/kimi-api/api-model-capabilities>
- Kimi K3 repository: <https://github.com/MoonshotAI/Kimi-K3>
- OpenAI model guide: <https://developers.openai.com/api/docs/guides/latest-model>
- MEXC contract API: <https://mexcdevelop.github.io/apidocs/contract_v1_en/>

## 11. Правило для новой нейросети

Сначала прочитать `AGENTS.md`, этот документ, затем handoff нужного worktree.
Перед изменениями назвать worktree/branch/HEAD, проверить status и не смешивать
root с MEXC. Факты кода, измерения, гипотезы и торговые решения всегда разделять.
Тесты доказывают согласованность реализации, но не прибыльность.
