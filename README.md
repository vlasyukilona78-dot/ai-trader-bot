# Koteika Ultra — causal Short on Pump research

Бот исследует затухание пампа и возможный SHORT-разворот. Целевая площадка —
**MEXC**, где отдельная ветка работает с публичными данными в signals-only режиме.
Root сохраняет Bybit/Phase 2 runtime, ранние и основные сигналы и виртуальное
сопровождение позиции.

Edge после издержек пока не доказан. Бот остановлен, live/testnet/private API и
Telegram заблокированы до ротации исторически раскрытых credentials и отдельной
проверки безопасности. Актуальный контекст, hashes и AI-roadmap:
`docs/CURRENT_STATE_AND_AI_PLAN_2026-08-03.md`.

## Быстрая проверка после переноса Windows

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\validate_install.ps1
```

Проверка использует локальную `.venv`, компилирует приложение и запускает
критические smoke-тесты.

## Запуск root runtime

`main` — имя сигнального профиля, а не гарантия paper-mode. Перед любым запуском
нужно отдельно проверить локальную конфигурацию и отключение outbound alerts.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_bot_clean.ps1 -SignalProfile main
```

Ранние сигналы запускаются отдельно:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_bot_clean.ps1 -SignalProfile early
```

Логи каждого запуска сохраняются в `logs/runtime`. Текущие Git tips не содержат
`.env`, но старый файл остаётся в истории; credentials ещё не ротированы. Последнее
проверенное состояние было `paper`, однако его нужно подтвердить заново перед
каждым запуском. Никаких ордеров сейчас отправлять нельзя.

## Как устроено решение

1. Layer 1 определяет сам памп и его устойчивый `pump_id`.
2. Layer 2 ищет истощение импульса и слабость покупателей.
3. Layer 3 проверяет качество уровня и реакцию цены.
4. Layer 4 отбрасывает ложные/продолжающиеся пампы с учётом funding, OI,
   order book, потока сделок и старших таймфреймов.
5. Entry Gate допускает только согласованный сетап.
6. После сигнала трекер позиции следит за лучшей ценой, возвратом прибыли,
   структурным разворотом и защитным выходом.

Целевая AI-архитектура: сначала локальный LightGBM на полной causal
runtime-population и PnL-метках frozen single-position contract; затем causal TCN
и Chronos как отдельные challengers. LLM допускается только для асинхронного
timestamped news-context и не имеет права задавать entry, sizing, stop, TP или
execution. Существующий legacy trainer к новому решению не подключён.

## Ручная позиция

Основной сигнал автоматически открывает виртуальное сопровождение. Если вход
был по другой цене или бот перезапускался, состояние можно задать вручную:

```powershell
.\.venv\Scripts\python.exe .\scripts\manual_signal_position.py open BTCUSDT --entry 1.234 --stop 1.28 --take-profit 1.10 --leverage 1 --replace
.\.venv\Scripts\python.exe .\scripts\manual_signal_position.py status BTCUSDT
.\.venv\Scripts\python.exe .\scripts\manual_signal_position.py close BTCUSDT --price 1.15 --reason manual
```

Параметр leverage в этом примере — metadata виртуального tracker, а не торговая
рекомендация. Состояние и события хранятся в `data/runtime`. Выход по виртуальной
позиции является уведомлением: бот не закрывает вручную открытую сделку на бирже.

## Проверенный checkpoint без внешнего диска

Ежедневную локальную копию можно создать на C: без USB:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\preservation\create_verified_backup.ps1 `
  -Mode LocalCheckpoint -BackupBase C:\koteika-checkpoints `
  -RootPath $PWD `
  -MexcPath "$PWD\.claude\worktrees\codex-project-review-04581e" `
  -PythonPath "$PWD\.venv\Scripts\python.exe"
```

`LocalCheckpoint` проверяет SHA-256 и SQLite, но находится на том же SSD и не
защищает от его отказа. Для другого диска/UNC используется режим
`DisasterResilient`. `.env*`, runtime system log, Git metadata, virtualenv и raw
WAL/SHM не копируются.

## Наблюдение и калибровка

Каждый памп-кандидат, включая отклонённые входы, записывается в
`data/runtime/decision_events.jsonl`. Сигналы и их жизненный цикл — в
`data/runtime/signal_events.jsonl`.

Готовый вердикт по окну наблюдения:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\observation\triage_calibration_result.ps1 -ComparisonJson <comparison.json>
```

Подробности: `scripts/observation/RUNBOOK_triage.md`.

Высокое плечо резко увеличивает риск ликвидации. Код не обещает прибыль:
перед реальной торговлей изменения должны пройти достаточное demo-наблюдение
и проверку на данных, не использованных при настройке.
