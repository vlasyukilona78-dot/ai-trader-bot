# Preservation Plan — 2026-07-30 (CORRECTED)

> [!CAUTION]
> **SUPERSEDED — DO NOT EXECUTE.** Этот план содержит ошибки в обработке
> tracked `.env`, порядке коммитов и backup-командах. В частности, нельзя
> выполнять из него старые `git rm --cached`, `C:\backup`, `sqlite3.exe`, staging,
> push и hash-проверки. Актуальный результат и дальнейшая процедура:
> `docs/CURRENT_STATE_AND_AI_PLAN_2026-08-03.md`.

**Статус:** архивный pre-execution plan. Preservation было выполнено по
перепроверенной последовательности 2026-08-03; этот файл сохранён только как
аудит истории решений.

---

## Критические поправки к предыдущей версии

1. ✅ **Remote ref подтверждён:** `origin/claude/codex-project-review-04581e` (не `droid/...`)
2. ✅ **MEXC worktree НЕ чистый:** `docs/AI_HANDOFF.md` имеет +175 uncommitted строк
3. ✅ **`.env` tracked в индексе и истории** root и MEXC — `.gitignore` НЕ защищает
4. ✅ **Манифесты → внешний архив:** реальная копия + SHA-256 проверка после копирования
5. ✅ **SQLite Online Backup API** вместо live-копирования DB/WAL
6. ✅ **48 файлов:** 31 modified + 17 new (14 test, 3 docs/scripts); `test_volume_profile_v2.py` дубликат удалён
7. ✅ **PowerShell команды:** все команды валидны для PowerShell, не bash
8. ✅ **Split commits:** 6 логических коммитов вместо одного mega-commit
9. ✅ **Handoff обновляется последним** после получения commit hashes и push state

---

## Подтверждённые факты

### Git-состояние

**Root worktree:**
- Branch: `feat/phase2-layer1-pump-runtime-alignment`
- HEAD: `0c38863`
- Remote HEAD: `0c38863` (совпадает)
- Status: 35 modified, 18 untracked (48 содержательных)
- Tests: 5 modified + 9 new = 14 файлов в `tests/v2/`

**MEXC worktree:**
- Branch: `claude/codex-project-review-04581e`
- Local HEAD: `9f71a86`
- Remote HEAD: `68e0ff7` ← **9 коммитов отстаёт**
- Remote ref: `origin/claude/codex-project-review-04581e` (подтверждено `git branch -vv`)
- Status: **DIRTY** — `docs/AI_HANDOFF.md` (+175 строк uncommitted)
- Tracking: `[origin/claude/codex-project-review-04581e: ahead 9]`

### Tracked `.env` — критический риск

```
Root: .env IS TRACKED (5+ commits in history)
MEXC: .env IS TRACKED (в том же диапазоне коммитов)
.gitignore: .env, .env.* (НЕ защищает уже tracked файлы)
```

История содержит credentials: `1eb2e99`, `16aa3a0`, `42c0322`, `399e79d`, `35a8dad`.

### Data inventory

**MEXC `data/`** — 853 МБ:
- `processed/`: 63 МБ (9 CSV: pump datasets, runtime calibration)
- `history/`: 790 МБ (1159 CSV: per-symbol OHLCV cache)

**Root `data/runtime/`** — 558 МБ:
- `v2_demo_runtime_main.db` + `.db-wal` + `.db-shm`: 553 МБ (WAL активен!)
- JSONL: `decision_events` (4.1 МБ, 543 строки), `signal_*` (126 КБ, 110 строк)

**Root `logs/observation/`** — 48 КБ:
- `comparison_pre_main_20260723_1954.json` (46 КБ) — единственный calibration result

---

## P0 — Credentials rotation (до любых операций с remote)

`.env` tracked в индексе и присутствует в public GitHub history. **Старые credentials потенциально скомпрометированы.**

### Обязательные действия (вручную, не автоматизировано):

1. **Ротировать немедленно:**
   - Bybit API key + secret
   - Telegram bot token
   - Любые proxy credentials в `.env`

2. **Удалить из индекса** (после ротации, перед любым commit):
   ```powershell
   # Root:
   git rm --cached .env

   # MEXC worktree:
   git -C .claude\worktrees\codex-project-review-04581e rm --cached .env
   ```

3. **Очистка истории — ОТДЕЛЬНОЕ решение:**
   - `git filter-repo` или `BFG Repo-Cleaner` на всех ветках
   - Требует force-push всех веток
   - Все клоны становятся несовместимыми
   - **Обсудить с пользователем отдельно; НЕ включать в preservation-план**

### Блокировка до ротации

Шаги 4 (push root) и 5 (push MEXC) **заблокированы** до выполнения пунктов 1–2 выше.

---

## Шаг 0 — Pre-flight checks (read-only)

```powershell
# Убедиться что bot не запущен:
Get-Process python -ErrorAction SilentlyContinue | Where-Object {$_.Path -like "*koteika*"}

# Проверить git status (не должно быть неожиданных файлов):
git status --short
git -C .claude\worktrees\codex-project-review-04581e status --short

# Проверить tracked .env (должен быть в списке):
git ls-files .env
git -C .claude\worktrees\codex-project-review-04581e ls-files .env

# Проверить последний коммит handoff'а (должен быть 9f71a86):
git -C .claude\worktrees\codex-project-review-04581e log -1 --oneline -- docs/AI_HANDOFF.md
```

**STOP условие:** если Python-процесс запущен, если появились незнакомые untracked файлы, если `.env` не в tracked — сообщить пользователю и остановиться.

---

## Шаг 1 — Patch .gitignore (fix coverage gap)

**Файл:** `C:\Users\vlasy\PycharmProjects\koteika_Ultra\.gitignore`

**Изменение:** добавить в секцию `# Virtual envs`:

```
.venv_relocated_*/
```

**НЕ коммитить** ещё. Это будет первый коммит в следующем шаге.

---

## Шаг 2 — Root commits (6 логических коммитов)

### Commit 1: fix(gitignore): add .venv_relocated_* exclusion

```powershell
git add .gitignore
git commit -m "fix(gitignore): add .venv_relocated_* exclusion

Prevent accidental staging of .venv_relocated_20260723/ backup directory."
```

### Commit 2: feat(core): strategy and signal generation updates

```powershell
git add `
  core/signal_generator.py `
  core/volume_profile.py `
  core/feature_engineering.py `
  core/indicators.py `
  core/market_data.py `
  core/settings.py `
  trading/signals/layered_strategy.py `
  trading/signals/entry_gate.py `
  trading/signals/models.py `
  trading/signals/runtime_source_adapter.py

git commit -m "feat(core): strategy and signal generation updates

- signal_generator: layered pump strategy improvements (+84 lines)
- volume_profile: POC/VAH/VAL calculation refinements (+80)
- feature_engineering: MTF and data quality flags (+31)
- indicators: new technical indicators (+26)
- market_data: derivatives context and funding (+41)
- layered_strategy: weakness confirmation and entry location
- entry_gate: MTF context validation and scoring
- runtime_source_adapter: feature contract alignment"
```

### Commit 3: feat(runtime): V2 control plane and market data overhaul

```powershell
git add `
  app/main.py `
  trading/market_data/feed.py `
  trading/features/pipeline.py `
  trading/alerts/signal_card_clean.py `
  trading/metrics/logging.py `
  alerts/telegram_client.py

git commit -m "feat(runtime): V2 control plane and market data overhaul

- app/main.py: +2186 lines — parallel universe scan, MTF integration,
  live-refresh before alert, managed exit notifications
- market_data/feed: bulk ticker snapshots, forced refresh for candidates,
  data quality checks
- features/pipeline: unusable frame rejection, explicit degraded flags
- signal_card_clean: rich card format with entry/TP/SL/confidence
- metrics/logging: structured cycle logging
- telegram_client: proxy detection, credential redaction"
```

### Commit 4: feat(observation): signal trackers and no-look-ahead observations

```powershell
git add `
  trading/state/signal_observation_tracker.py `
  trading/state/signal_position_tracker.py

git commit -m "feat(observation): signal trackers and no-look-ahead observations

- SignalObservationTracker: no-look-ahead post-signal movement tracking,
  expire_stale() for symbols leaving universe
- SignalPositionTracker: shadow/manual position tracking, managed exit
  notifications, TP/SL barrier handling, conservative same-bar rule"
```

### Commit 5: test(v2): comprehensive test suite for Phase 2 runtime

```powershell
git add `
  tests/test_strategy_dry_run.py `
  tests/v2/test_alert_disable_v2.py `
  tests/v2/test_early_lifecycle_v2.py `
  tests/v2/test_entry_gate_v2.py `
  tests/v2/test_feature_pipeline_v2.py `
  tests/v2/test_market_data_feed_v2.py `
  tests/v2/test_market_parallelism_v2.py `
  tests/v2/test_metrics_logging_v2.py `
  tests/v2/test_ml_shadow_integration_v2.py `
  tests/v2/test_observation_signal_quality_v2.py `
  tests/v2/test_runtime_instance_lock_v2.py `
  tests/v2/test_signal_observation_tracker_v2.py `
  tests/v2/test_signal_position_tracker_v2.py `
  tests/v2/test_telegram_client_v2.py `
  tests/v2/test_volume_profile_v2.py

git commit -m "test(v2): comprehensive test suite for Phase 2 runtime

14 test files covering:
- Lifecycle: early_lifecycle, alert_disable
- Parallel execution: market_parallelism, runtime_instance_lock
- Trackers: signal_observation_tracker, signal_position_tracker
- Pipelines: entry_gate, feature_pipeline, market_data_feed
- Integrations: ml_shadow_integration, telegram_client
- Quality: observation_signal_quality
- Components: volume_profile, metrics_logging

Full suite: 529 passed, 4 skipped."
```

### Commit 6: chore(env): scripts, config, and root launcher updates

```powershell
git add `
  main.py `
  requirements.txt `
  config/config.yaml `
  config/secrets.env.example `
  README.md `
  scripts/start_bot_clean.ps1 `
  scripts/validate_install.ps1 `
  scripts/manual_signal_position.py `
  scripts/keep_awake_night.ps1 `
  scripts/observation/analyze_recent_signal_quality.py `
  scripts/observation/collect_observation_window.ps1

git commit -m "chore(env): scripts, config, and root launcher updates

- main.py: Windows-friendly dual-profile supervisor
- scripts/start_bot_clean.ps1: portable launcher with -Once, separate logs
- scripts/validate_install.ps1: dependency check + 34 smoke tests
- scripts/manual_signal_position.py: manual position registration
- scripts/observation: quality analysis and window collection
- config: new settings for MTF, observation, shadow ML
- requirements.txt: dependency updates
- README: updated for Phase 2 architecture"
```

**НЕ включать в коммиты:**
- `.idea/` файлы (tracked IDE noise — НЕ стейджить)
- `logs/system.log` (tracked runtime log — НЕ стейджить)
- `CLAUDE.md`, `docs/AI_HANDOFF.md`, `docs/PROJECT_HANDOFF_FOR_CLAUDE.md` (обновятся в шаге 7)

---

## Шаг 3 — MEXC commit (uncommitted AI_HANDOFF.md)

```powershell
$MEXC = ".claude\worktrees\codex-project-review-04581e"

git -C $MEXC add docs/AI_HANDOFF.md

git -C $MEXC commit -m "docs: record root-worktree state and quality findings

+175 lines documenting:
- Root worktree Phase 2 development (2026-07-25)
- Signal observation expiration for symbols leaving universe
- Quality findings: WATCH signals not ready for live entry
- Calibration verdict: pause_calibration, window_size_not_comparable

Bot remains stopped, paper mode only."
```

---

## Шаг 4 — Push root (after credentials rotation)

**BLOCKED:** выполнить только после ротации credentials и `git rm --cached .env`.

```powershell
# Проверить что .env больше не tracked:
git ls-files .env
# Должен вернуть пустой результат. Если .env всё ещё в списке — STOP.

# Push (fast-forward, --force НЕ нужен):
git push origin feat/phase2-layer1-pump-runtime-alignment
```

Это добавит 6 новых коммитов сверху `0c38863`.

---

## Шаг 5 — Push MEXC (after credentials rotation)

**BLOCKED:** выполнить только после ротации credentials и `git rm --cached .env` в MEXC worktree.

```powershell
$MEXC = ".claude\worktrees\codex-project-review-04581e"

# Проверить что .env больше не tracked:
git -C $MEXC ls-files .env
# Должен вернуть пустой результат. Если .env всё ещё в списке — STOP.

# Push (fast-forward с 68e0ff7 → 9f71a86 + новый коммит):
git -C $MEXC push origin claude/codex-project-review-04581e
```

Это загрузит 9 локальных коммитов (`8cc31fc..9f71a86`) + 1 новый = 10 коммитов.

---

## Шаг 6 — External archive with hash verification

### 6.1 SQLite Online Backup (root runtime DB)

**НЕ копировать live DB/WAL командами ОС.** Использовать SQLite Online Backup API:

```powershell
# Создать backup-директорию:
New-Item -ItemType Directory -Force -Path "C:\backup\koteika_2026-07-30\root_runtime"

# Online backup через sqlite3 CLI:
$sourceDB = "C:\Users\vlasy\PycharmProjects\koteika_Ultra\data\runtime\v2_demo_runtime_main.db"
$targetDB = "C:\backup\koteika_2026-07-30\root_runtime\v2_demo_runtime_main_backup.db"

sqlite3 $sourceDB ".backup '$targetDB'"

# Проверка целостности backup:
sqlite3 $targetDB "PRAGMA integrity_check;"
# Должно вернуть: ok

# Verify record count:
sqlite3 $targetDB "SELECT COUNT(*) FROM state_records;"
# Должно вернуть: 467 (или близко, если bot запускался после snapshot)
```

### 6.2 MEXC processed datasets (63 МБ — критичны)

```powershell
$backupRoot = "C:\backup\koteika_2026-07-30"
$mexcData = "C:\Users\vlasy\PycharmProjects\koteika_Ultra\.claude\worktrees\codex-project-review-04581e\data\processed"

# Copy:
Copy-Item -Path "$mexcData\*" -Destination "$backupRoot\mexc_processed\" -Force

# Generate SHA-256 manifest:
Get-ChildItem "$mexcData\*.csv" | ForEach-Object {
    $hash = (Get-FileHash $_.FullName -Algorithm SHA256).Hash
    "$hash  $($_.Name)"
} | Out-File -Encoding utf8 "$backupRoot\mexc_processed\MANIFEST_SHA256.txt"

# Verify после копирования:
Get-ChildItem "$backupRoot\mexc_processed\*.csv" | ForEach-Object {
    $hash = (Get-FileHash $_.FullName -Algorithm SHA256).Hash
    $expected = Select-String -Path "$backupRoot\mexc_processed\MANIFEST_SHA256.txt" -Pattern $_.Name
    if ($expected -match $hash) {
        Write-Host "OK: $($_.Name)"
    } else {
        Write-Host "FAIL: $($_.Name) — hash mismatch!" -ForegroundColor Red
    }
}
```

### 6.3 MEXC history cache (790 МБ, 1159 CSV — опционально)

History может быть перезагружен с MEXC API, но потеря = дни ожидания.

**Минимум:** сохранить SHA-256 manifest без архива:

```powershell
$mexcHistory = "C:\Users\vlasy\PycharmProjects\koteika_Ultra\.claude\worktrees\codex-project-review-04581e\data\history"

Get-ChildItem "$mexcHistory\*.csv" | ForEach-Object {
    $hash = (Get-FileHash $_.FullName -Algorithm SHA256).Hash
    "$hash  $($_.Name)"
} | Out-File -Encoding utf8 "$backupRoot\mexc_history_MANIFEST_SHA256.txt"
```

**Рекомендуется:** полная копия + verify (долго, ~5–10 минут):

```powershell
Copy-Item -Path "$mexcHistory" -Destination "$backupRoot\mexc_history\" -Recurse -Force

# Verify sample (первые 10 файлов для smoke-check):
Get-ChildItem "$backupRoot\mexc_history\*.csv" | Select-Object -First 10 | ForEach-Object {
    $originalHash = (Get-FileHash "$mexcHistory\$($_.Name)" -Algorithm SHA256).Hash
    $backupHash = (Get-FileHash $_.FullName -Algorithm SHA256).Hash
    if ($originalHash -eq $backupHash) {
        Write-Host "OK: $($_.Name)"
    } else {
        Write-Host "FAIL: $($_.Name)" -ForegroundColor Red
    }
}
```

### 6.4 Root runtime JSONL/observations

```powershell
$runtimeData = "C:\Users\vlasy\PycharmProjects\koteika_Ultra\data\runtime"
$observationData = "C:\Users\vlasy\PycharmProjects\koteika_Ultra\logs\observation"

# Copy JSONL:
Copy-Item -Path "$runtimeData\*.jsonl" -Destination "$backupRoot\root_runtime\" -Force
Copy-Item -Path "$runtimeData\*.json" -Destination "$backupRoot\root_runtime\" -Force

# Copy observation comparison:
Copy-Item -Path "$observationData\*.json" -Destination "$backupRoot\root_observation\" -Force

# Generate manifest:
Get-ChildItem "$backupRoot\root_runtime\*", "$backupRoot\root_observation\*" | ForEach-Object {
    $hash = (Get-FileHash $_.FullName -Algorithm SHA256).Hash
    "$hash  $($_.Name)"
} | Out-File -Encoding utf8 "$backupRoot\root_data_MANIFEST_SHA256.txt"
```

### 6.5 Backup summary

После завершения шага 6:

```powershell
Write-Host "`n=== BACKUP INVENTORY ==="
Get-ChildItem -Path $backupRoot -Recurse -File | Measure-Object -Property Length -Sum | ForEach-Object {
    Write-Host "Total files: $($_.Count)"
    Write-Host "Total size: $([math]::Round($_.Sum / 1MB, 2)) MB"
}

Write-Host "`n=== MANIFESTS ==="
Get-ChildItem "$backupRoot\*MANIFEST*.txt"
```

---

## Шаг 7 — Update handoff docs (последний шаг)

После получения новых commit hashes от шагов 2–3 и push state от шагов 4–5.

### 7.1 Получить актуальные hashes

```powershell
$rootHead = git rev-parse HEAD
$rootHeadShort = git rev-parse --short HEAD
$mexcHead = git -C .claude\worktrees\codex-project-review-04581e rev-parse HEAD
$mexcHeadShort = git -C .claude\worktrees\codex-project-review-04581e rev-parse --short HEAD

Write-Host "Root HEAD: $rootHead ($rootHeadShort)"
Write-Host "MEXC HEAD: $mexcHead ($mexcHeadShort)"
```

### 7.2 Update CLAUDE.md

**Файл:** `C:\Users\vlasy\PycharmProjects\koteika_Ultra\CLAUDE.md`

**Изменения:**
- Секция `## Critical repository state` → обновить root HEAD на новый hash из шага 2
- Секция `## Critical repository state` → обновить MEXC HEAD на новый hash из шага 3
- Добавить дату последнего preservation: `Last preservation: 2026-07-30`

### 7.3 Update PROJECT_HANDOFF_FOR_CLAUDE.md

**Файл:** `C:\Users\vlasy\PycharmProjects\koteika_Ultra\docs\PROJECT_HANDOFF_FOR_CLAUDE.md`

**Изменения:**
- Строка 3: `Актуальность снимка: **2026-07-30, Europe/Moscow**.`
- Раздел `## 4.1 Корневой Bybit/runtime worktree` → обновить HEAD
- Раздел `## 4.2 Отдельный Claude/MEXC worktree` → обновить HEAD
- Раздел `## 4.2` → убрать "9 локальных коммитов" (теперь запушено)
- Раздел `## 3.1 Runtime` → добавить: `- Последний preservation checkpoint: 2026-07-30`

### 7.4 Commit handoff updates

```powershell
git add CLAUDE.md docs/PROJECT_HANDOFF_FOR_CLAUDE.md

git commit -m "docs: update handoff after preservation checkpoint 2026-07-30

- CLAUDE.md: updated root/MEXC HEAD hashes, preservation date
- PROJECT_HANDOFF_FOR_CLAUDE.md: snapshot date 2026-07-30,
  confirmed MEXC commits pushed, runtime checkpoint recorded

All Phase 2 work preserved:
- 6 root commits: gitignore, core/strategy, runtime, observation,
  tests, environment
- 10 MEXC commits pushed (9 prior + 1 AI_HANDOFF update)
- External backup: SQLite online backup, processed datasets verified,
  runtime JSONL/observations archived"

# Push:
git push origin feat/phase2-layer1-pump-runtime-alignment
```

---

## Validation checklist (после всех шагов)

```powershell
# 1. Git status чистый (кроме .idea/, logs/system.log — tracked noise):
git status --short
# Ожидается: только .idea/, logs/system.log если они были modified

# 2. .env не в индексе:
git ls-files .env
git -C .claude\worktrees\codex-project-review-04581e ls-files .env
# Ожидается: пустой вывод

# 3. Remote синхронизирован:
git fetch origin
git log --oneline origin/feat/phase2-layer1-pump-runtime-alignment..HEAD
# Ожидается: пустой вывод (всё запушено)

$MEXC = ".claude\worktrees\codex-project-review-04581e"
git -C $MEXC fetch origin
git -C $MEXC log --oneline origin/claude/codex-project-review-04581e..HEAD
# Ожидается: пустой вывод

# 4. Backup существует и полон:
Test-Path C:\backup\koteika_2026-07-30\mexc_processed\MANIFEST_SHA256.txt
Test-Path C:\backup\koteika_2026-07-30\root_runtime\v2_demo_runtime_main_backup.db
Test-Path C:\backup\koteika_2026-07-30\root_data_MANIFEST_SHA256.txt
# Все должны вернуть: True

# 5. SQLite backup integrity:
sqlite3 C:\backup\koteika_2026-07-30\root_runtime\v2_demo_runtime_main_backup.db "PRAGMA integrity_check;"
# Должно вернуть: ok
```

---

## Риски и митигация (обновлено)

| Риск | Уровень | Митигация |
|---|---|---|
| `.env` leaked в public history | **P0 — Критический** | Ротация credentials обязательна ДО push; `git rm --cached` в шагах 4–5; очистка истории — отдельное решение |
| CRLF warnings при каждом git-операции | Низкий | Добавить `.gitattributes` с `* text=auto eol=lf` (P2, отдельный PR) |
| SQLite WAL несовместим с live DB при копировании | Высокий | SQLite Online Backup API в шаге 6.1 вместо `Copy-Item` |
| Backup без проверки хэшей | Средний | Явная верификация SHA-256 после каждого copy в шаге 6 |
| Один mega-commit затрудняет review/revert | Средний | 6 логических коммитов в шаге 2 |
| `test_volume_profile_v2.py` staged дважды | Низкий (исправлено) | Исправлено в списке файлов |
| Handoff обновлён до получения hashes | Средний | Handoff коммитится последним в шаге 7 |
| Push до ротации credentials | **P0 — Критический** | STOP-условия в шагах 4–5 с явной проверкой `git ls-files .env` |

---

## Execution time estimate

- Шаг 0: 2 мин (pre-flight checks)
- Шаг 1: 1 мин (patch `.gitignore`)
- Шаг 2: 10 мин (6 коммитов с проверкой `git status` между ними)
- Шаг 3: 2 мин (MEXC commit)
- **P0 блокировка:** ротация credentials — зависит от пользователя
- Шаг 4: 2 мин (push root после ротации)
- Шаг 5: 2 мин (push MEXC после ротации)
- Шаг 6: 20–30 мин (SQLite backup 5 мин, MEXC processed 2 мин, history 10–20 мин, JSONL 2 мин)
- Шаг 7: 5 мин (handoff update + commit + push)

**Общее время:** ~45–60 минут (включая ожидание push, без учёта ротации credentials).

---

## Next steps after preservation

После успешного завершения preservation-плана:

1. **Подтвердить MEXC как целевую биржу** — пользователь должен явно согласовать продолжение от `9f71a86`
2. **Сформулировать executable single-position contract:**
   - Один вход или DCA
   - Sizing basis (% equity, fixed USDT)
   - Stop: fixed или managed trailing
   - TP: single target или partial exits
   - Leverage, margin mode (isolated/cross)
   - Fees, spread, slippage, funding model
   - Concurrency, capital occupancy
   - Success definition (hit TP, hold 48h, realized PnL)
3. **Начать causal implementation parity** — реализовать недостающие intended features (Fibonacci, overhead levels, weakness, liquidation-in-decision, confluence, 1h RSI)

---

**Статус:** План готов. Ничего не выполнено. Ожидает подтверждения пользователя.
