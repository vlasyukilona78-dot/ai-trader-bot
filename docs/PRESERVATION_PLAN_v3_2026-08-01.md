# Koteika Ultra — preservation plan v3

Актуальность исходного аудита: **2026-08-03, Europe/Moscow**.

> [!IMPORTANT]
> **EXECUTED / ARCHIVED — DO NOT RE-RUN AS A COMMAND LIST.** Фактический receipt,
> текущие hashes, локальный checkpoint без USB и AI-roadmap находятся в
> `docs/CURRENT_STATE_AND_AI_PLAN_2026-08-03.md`. Плановое тело ниже сохранено для
> аудита и содержит исходные hashes/STOP-условия, которые уже не являются текущим
> состоянием.

Фактический outcome:

- disaster-resilient backup подтверждён в
  `D:\koteika-preservation\koteika_preservation_20260803_135615_6d24c806`;
  `VERIFIED_OK.txt`, 2387 payload files, 2,840,300,975 bytes, manifest SHA-256
  `3f86780a487d2f6f02e3603d83e741e5665fb0f0cdc91cb6cadaa1f71274a075`;
- receipt фиксирует pre-mutation heads `0c38863`/`9f71a86`, а не новые tips;
- root functional anchor `2f7e18f` опубликован, тесты
  `533 passed, 4 skipped, 3 warnings`;
- MEXC functional anchor `98217df` опубликован, тесты
  `340 passed, 4 skipped, 2 warnings`;
- `.env` и runtime system logs удалены из текущих tips; история не очищена и
  credentials не ротированы, поэтому private API/Telegram/testnet/live закрыты;
- `scripts/preservation/create_verified_backup.ps1` теперь различает
  `LocalCheckpoint` на C: (`CHECKPOINT_VERIFIED.json`) и настоящий
  `DisasterResilient` на другом диске/UNC (`VERIFIED_OK.json`).
- текущий `LocalCheckpoint` успешно создан в
  `C:\koteika-checkpoints\koteika_preservation_20260803_154657_88710bc3`:
  root `80e6f2b`, MEXC `1e91ce0`, 2571 файлов / 2,861,567,291 bytes, manifest
  `237204c9b629e48a60b185accf3f3f05491c84a43ca5f11c26e5bc950a0aec89`.

Этот документ заменяет `docs/PRESERVATION_PLAN_2026-07-30.md`.

## 1. Цель и критерий завершения

Цель — без потери и без смешивания двух линий разработки сохранить:

1. незакоммиченный root/Bybit Phase 2 runtime;
2. девять локальных MEXC-коммитов;
3. незакоммиченный независимый MEXC-аудит;
4. root runtime observations и согласованный SQLite snapshot;
5. полный MEXC research dataset/history;
6. актуальные handoff-документы для следующей AI-сессии;
7. доказательства того, что сохранённые commits прошли свежие тесты и remote
   действительно получил те же hashes.

Preservation завершён только когда одновременно выполнены условия:

- внешний backup находится не на том же физическом томе, что проект;
- backup полностью проверен SHA-256 и SQLite integrity check;
- MEXC final tip больше не содержит `.env`;
- root и MEXC local HEAD равны live remote tips;
- root и MEXC прошли свежие test suites с exit code 0;
- root index пуст, содержательные изменения закоммичены;
- MEXC worktree чист;
- bot остаётся остановленным, `paper`, без live/testnet/demo действий;
- выпущен финальный execution receipt с hashes и результатами.

## 2. Подтверждённое исходное состояние

### 2.1 Git

| Линия | Local | Live remote | Состояние |
|---|---|---|---|
| root/Bybit | `0c38863523cc4cf0f021677f04d08349be0c3aca` | тот же hash в `feat/phase2-layer1-pump-runtime-alignment` | 31 содержательный modified + 19 содержательных untracked |
| MEXC | `9f71a866b413bba4f1ab3a21603219dfe61f16fd` | `68e0ff77373db2b48c19e3dcf09f1cbb7d569e47` в `claude/codex-project-review-04581e` | ahead 9; `docs/AI_HANDOFF.md` имеет `+175/-0` |

Дополнительно в root намеренно остаются четыре tracked-noise файла:

- `.idea/koteika_Ultra.iml`;
- `.idea/misc.xml`;
- `.idea/vcs.xml`;
- `logs/system.log`.

Новых/изменённых тестов: **15** — 6 modified + 9 new.

Появление этого v3-плана уже учтено: всего сохраняется **50 содержательных
root-файлов** — 31 modified + 19 untracked.

### 2.2 `.env` и security

Точный факт:

- root Phase 2 index, root `HEAD` и целевая root remote-ветка `.env` **не
  содержат**;
- root-команду `git rm --cached .env` выполнять нельзя;
- MEXC local `HEAD` и `origin/claude/codex-project-review-04581e` `.env`
  содержат;
- `origin/main` и несколько исторических remote-веток также содержат `.env`;
- `.gitignore` не удаляет уже tracked-файл.

Удаление `.env` из нового MEXC tip не очищает историю. Ротация credentials —
обязательная защита; history rewrite и remediation старых веток являются
отдельной разрушительной задачей.

### 2.3 Runtime и данные

- Подходящих запущенных Python-процессов проекта не найдено.
- `data/runtime/bot_runtime.lock` существует и может быть stale; автоматически
  удалять его нельзя.
- Runtime остаётся `paper`; запускать бот, `-Once`, observation collector,
  alerts, testnet или live в рамках preservation запрещено.
- Python: 3.12.13; stdlib SQLite: 3.50.4 с `Connection.backup()`.
- `sqlite3.exe` в PATH отсутствует.
- MEXC `processed`: 9 файлов / 65,488,677 bytes.
- MEXC `history`: 1,159 файлов / 825,012,437 bytes.
- Полное MEXC `data/`: 1,170 файлов / 890,646,683 bytes; сюда также входят
  `raw/BTCUSDT_1m.csv` и `runtime/v2_runtime.db`.
- Root runtime: около 583 MB; observation comparison: один JSON около 46 KB.
- В root runtime есть 611 `alert_locks` — это persisted dedupe state, а не
  cache; их нужно сохранять. `data/runtime/matplotlib/` является восстановимым
  cache и в канонический backup не входит.
- Sanitised snapshot tracked MEXC tree без `.env`: 510 файлов, около 1.41 GB.

## 3. Неприкосновенные правила

На всех этапах запрещены:

- `git reset`, `git clean`, `checkout --`, force-push и массовое форматирование;
- `git add -A`, `git add .` и staging по wildcard всего worktree;
- чтение или вывод значений `.env` в чат/лог/manifest;
- архивирование `.claude/worktrees` или всего проекта целиком;
- включение `.env`, `.venv*`, recovery-каталогов, `.git*`, runtime logs или IDE
  metadata в remote commits;
- raw-копирование активной SQLite DB/WAL как канонического snapshot;
- history rewrite в рамках этого плана;
- любые торговые или alert-вызовы.

После любого неожиданного status, hash, staged path, test failure, secret-scan
hit, backup mismatch или remote divergence действует правило: **STOP, ничего не
исправлять автоматически, показать пользователю факт**.

## 4. Границы полномочий

План делится на три отдельных пользовательских разрешения:

1. **Backup approval** — разрешение создать проверенную копию на указанном
   внешнем носителе/UNC.
2. **Local commit approval** — разрешение patch/stage/commit после успешного
   backup.
3. **Push approval** — разрешение отправить commits только после ротации
   credentials, свежих тестов и dry-run push.

Ротацию Bybit/Telegram/OpenAI/Coinglass/proxy credentials выполняет или явно
подтверждает пользователь. Новые значения не передаются AI.

## 5. Этап A — read-only preflight

Выполнять в одном PowerShell-сеансе:

```powershell
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = 'C:\Users\vlasy\PycharmProjects\koteika_Ultra'
$Mexc = Join-Path $Root '.claude\worktrees\codex-project-review-04581e'
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$RootRemoteRef = 'refs/heads/feat/phase2-layer1-pump-runtime-alignment'
$MexcRemoteRef = 'refs/heads/claude/codex-project-review-04581e'
```

Проверки:

1. `$Root`, `$Mexc`, `$Python` существуют.
2. Нет project-related `python.exe/pythonw.exe` по `CommandLine` или
   `ExecutablePath`.
3. Оба Git index пусты.
4. Root HEAD точно `0c38863...`, MEXC HEAD точно `9f71a86...`.
5. Live remote tips, полученные через `git ls-remote`, всё ещё `0c38863...` и
   `68e0ff7...`.
6. Root `.env` отсутствует в index/HEAD; MEXC `.env` присутствует в index/HEAD.
7. MEXC ahead ровно 9 и dirty-path ровно `docs/AI_HANDOFF.md`.
8. Root meaningful sets точно совпадают с совокупными allowlists разделов 9 и
   11.
9. Capacity gate вычислен по текущему измеренному payload:
   `required = measured_payload * 1.25 + 512 MiB`; при этом свободно должно
   быть не менее 4 GiB. Для UNC, где размер нельзя надёжно получить, требуется
   отдельное ручное подтверждение ёмкости.

Проверка процесса должна использовать `Get-CimInstance Win32_Process`, а не
только `Get-Process.Path`: системный Python может запускать проект через
command-line.

## 6. Этап B — внешний backup до любых Git-изменений

### 6.1 Обязательный внешний destination

На машине сейчас обнаружен только том `C:`. `C:\backup` защищает от ошибки Git,
но не от потери диска и поэтому **не считается полным preservation**.

Пользователь должен указать:

- другой физический том, например `E:\koteika-backups`; либо
- UNC, например `\\nas\backups\koteika`.

```powershell
$BackupBase = '<USER_SUPPLIED_EXTERNAL_OR_UNC_PATH>'
```

STOP, если путь не задан, недоступен, совпадает с исходным томом или содержит
уже существующий run-directory. Run-directory создаётся уникально:

```text
koteika_preservation_YYYYMMDD_HHMMSS_<8-char-guid>
```

Backup считается чувствительным: хотя `.env` исключён, source snapshots и
research data содержат операционный контекст.

### 6.2 Что сохранить

До первого patch/stage сохранить одновременно:

1. metadata:
   - timestamp/timezone;
   - root/MEXC branch, local HEAD, live remote tips;
   - `git status --porcelain=v2 --branch`;
   - точные allowlists;
2. root:
   - полный binary diff относительно `HEAD`;
   - полные версии всех 31 meaningful modified файлов;
   - полные версии всех 19 meaningful untracked файлов;
3. MEXC:
   - binary patch `docs/AI_HANDOFF.md`;
   - девять `format-patch --binary --full-index` файлов диапазона
     `origin/claude/...HEAD`;
   - sanitised snapshot всех tracked-файлов, кроме `.env`;
4. SQLite:
   - каждый ненулевой `*.db` через Python `sqlite3.Connection.backup()`;
   - нулевой early DB как точную zero-byte copy;
   - JSON-report: SQLite version, integrity result и counts всех таблиц;
5. Root research/runtime:
   - top-level `data/runtime/*.json` и `*.jsonl`;
   - полное `data/runtime/alert_locks/**` с source-before/source-after и
     destination manifests;
   - `logs/observation/*.json`;
   - stale `data/runtime/bot_runtime.lock` только как metadata, с явной пометкой
     `stale`; автоматически удалять или считать активным runtime state нельзя;
6. MEXC research:
   - **полное** дерево `data/`, включая processed, history и raw artifacts;
   - `data/runtime/v2_runtime.db` через SQLite Online Backup как канонический
     snapshot, а не обычную файловую копию.

Не сохранять `.env`, raw WAL/SHM, `.venv*`, `.git*` или recovery directories в
канонический backup. `data/runtime/matplotlib/**` явно исключить как
восстановимый cache. Raw-copy MEXC `data/runtime/v2_runtime.db` не считать
каноническим snapshot и не смешивать с Online Backup output.

### 6.3 Безопасный SQLite snapshot

Использовать `$Python`, URI `mode=ro` и `Connection.backup()` для **обоих**
источников:

- root `data/runtime/*.db`;
- MEXC `data/runtime/*.db`, включая `v2_runtime.db`.

Для каждого DB:

1. destination не должен существовать;
2. выполнить online backup;
3. на backup выполнить `PRAGMA integrity_check`;
4. получить список пользовательских таблиц и counts;
5. сравнить source и destination counts;
6. записать результат в `sqlite_backup_report.json`;
7. не выполнять checkpoint и не считать `.db-shm` частью backup.

Read-only SQLite connection может коснуться временного `.db-shm`; это допустимо
и должно быть отмечено в execution receipt. Основные `.db`/`.db-wal` не должны
измениться.

### 6.4 Копирование и SHA-256

Для деревьев данных использовать `robocopy /E /COPY:DAT /DCOPY:DAT /R:2 /W:1
/XJ`; exit codes 0–7 являются успешными, 8+ — STOP.

Для каждого дерева обязательна схема:

```text
source-before manifest
    -> copy
source-after manifest
destination manifest
    -> compare source-before == source-after == destination
```

Manifest содержит repo-relative path, byte length и SHA-256, отсортированные по
path. Проверяются **все** файлы, не sample. Manifest-only без копии не считается
preservation.

До создания run-directory измерить ожидаемый payload, включая sanitised MEXC
tracked snapshot. Известный минимум на 2026-08-03 — около 2.64 GiB ещё до
patches/manifests/reports; поэтому прежний фиксированный лимит 3 GiB отменён.
Применяется `max(4 GiB, measured_payload * 1.25 + 512 MiB)`.

После component checks создать общий `MANIFEST_SHA256.csv`, отдельный hash
самого manifest и `VERIFIED_OK.txt`. До появления `VERIFIED_OK.txt` запрещены
patch/stage/commit.

## 7. Этап C — rotation gate и MEXC security commit

После подтверждённого backup пользователь ротирует все когда-либо
использовавшиеся значения:

- Bybit keys/secrets, включая profile-specific main/early;
- Telegram bot tokens;
- OpenAI и Coinglass API keys;
- proxy credentials и любые переиспользованные credentials.

Chat IDs не являются auth-secret, но считаются sensitive metadata.

После подтверждения ротации:

```powershell
git -C $Mexc rm --cached -- .env
```

Ожидания до commit:

- локальный `$Mexc\.env` физически остался;
- `git -C $Mexc ls-files -- .env` пуст;
- staged diff содержит **ровно** `D .env`;
- `.env` покрыт ignore-rule;
- содержимое staged deletion никогда не выводится в чат.

Отдельный commit:

```text
chore(security): stop tracking local environment file
```

После commit:

- `HEAD:.env` отсутствует;
- локальный ignored `.env` существует;
- новые rotated values вносятся пользователем локально, не через AI;
- root `git rm --cached .env` не выполняется.

`origin/main` и старые ветки/история всё ещё содержат старый файл. После ротации
это остаётся отдельным P0 remediation-проектом, но не блокирует preservation
текущих tips.

## 8. Этап D — MEXC audit commit, validation и push

Закоммитить единственный текущий dirty-файл:

```text
docs/AI_HANDOFF.md
```

Commit message должен соответствовать фактическим `+175` строкам:

```text
docs(audit): record independent review of nine MEXC commits
```

В body перечислить только фактические темы:

- forming/closed candle mismatch;
- observation backfill и delivery accounting;
- shared mutable strategy/thread-safety;
- fail-open/stale context;
- runtime-dataset parity;
- replay/DCA/reproducibility defects;
- no live authorization.

После двух новых commits MEXC будет ahead remote на 11, а не на 10.

До push:

1. MEXC status чистый; local ignored `.env` допустим.
2. Compile exit code 0.
3. Полный MEXC pytest exit code 0; фактическую summary сохранить, не
   предсказывать число заранее.
4. Staged/current-tip secret gate пройден.
5. Live remote повторно fetched.
6. Remote tip является ancestor local HEAD.
7. `git push --dry-run` успешен.

Push только explicit, без force:

```powershell
git -C $Mexc push --porcelain origin `
  HEAD:refs/heads/claude/codex-project-review-04581e
```

После push проверить live `ls-remote` == local HEAD и отсутствие `.env` в
remote tip tree.

## 9. Этап E — root commits

### Общее правило каждого commit

Перед staging index должен быть пуст. Staging выполняется только exact
PowerShell-array allowlist.

После staging:

1. `git diff --cached --name-only` точно равен ожидаемому set;
2. `git diff --cached --name-status` и `--stat` просмотрены;
3. `git diff --cached --check` успешен;
4. staged filename denylist не содержит `.env`, private keys или credential
   exports;
5. content secret-scan выводит только rule/path/line number, не matched values;
6. соответствующие tests проходят;
7. после commit index снова пуст.

Запрещено заранее записывать в commit message старые числа `529/287 passed`.
Используется только фактический свежий результат.

### R1 — local recovery ignores

Файл: `.gitignore`.

Перед staging добавить `.venv_relocated_*/`. Текущий diff уже содержит
`.venv_flash_*/`, `.git_corrupt_*/`, `.git_mixed_*/` и
`recovery_snapshot_*.zip`, поэтому commit message обязан описывать весь diff:

```text
chore(repo): ignore local recovery environments and snapshots
```

### R2 — strategy/features foundation

```text
config/config.yaml
core/settings.py
core/indicators.py
core/feature_engineering.py
core/signal_generator.py
core/volume_profile.py
trading/features/pipeline.py
trading/signals/entry_gate.py
trading/signals/layered_strategy.py
trading/signals/models.py
tests/v2/test_entry_gate_v2.py
tests/v2/test_feature_pipeline_v2.py
tests/v2/test_volume_profile_v2.py
```

Commit:

```text
feat(strategy): strengthen MTF context and structural admission
```

### R3 — persistent lifecycle trackers

```text
trading/state/signal_observation_tracker.py
trading/state/signal_position_tracker.py
scripts/manual_signal_position.py
tests/v2/test_signal_observation_tracker_v2.py
tests/v2/test_signal_position_tracker_v2.py
```

Commit:

```text
feat(observation): add persistent signal lifecycle trackers
```

R3 идёт до `app/main.py`, который импортирует оба tracker-модуля.

### R4 — market data and transport

```text
core/market_data.py
trading/market_data/feed.py
trading/signals/runtime_source_adapter.py
trading/metrics/logging.py
alerts/telegram_client.py
tests/v2/test_market_data_feed_v2.py
tests/v2/test_metrics_logging_v2.py
tests/v2/test_telegram_client_v2.py
```

Commit:

```text
feat(market-data): add bulk ticker, native MTF, and trade-flow context
```

### R5 — runtime integration

```text
app/main.py
main.py
trading/alerts/signal_card_clean.py
config/secrets.env.example
scripts/start_bot_clean.ps1
tests/test_strategy_dry_run.py
tests/v2/test_alert_disable_v2.py
tests/v2/test_early_lifecycle_v2.py
tests/v2/test_market_parallelism_v2.py
tests/v2/test_ml_shadow_integration_v2.py
tests/v2/test_runtime_instance_lock_v2.py
```

Commit:

```text
feat(runtime): integrate signal lifecycle and single-instance control
```

К R5 уже существуют strategy, trackers, market feed и logging dependencies.

### R6 — observation quality tooling

```text
scripts/observation/analyze_recent_signal_quality.py
scripts/observation/collect_observation_window.ps1
tests/v2/test_observation_signal_quality_v2.py
```

Commit:

```text
feat(observation): consume local signal outcomes in quality analysis
```

После R6 выполнить предписанный triage на существующем comparison JSON. Exit
code 13 ожидаем и означает `window_size_not_comparable`; thresholds остаются
заморожены. Нельзя трактовать non-zero code 13 как поломку скрипта.

### R7 — environment validation

```text
requirements.txt
scripts/validate_install.ps1
```

Commit:

```text
chore(env): preserve dependency bounds and installation validation
```

Известный blocker остаётся: requirements и runtime lock расходятся, а clean
install воспроизводимость не доказана. Preservation фиксирует текущее состояние,
но не объявляет dependency problem решённой.

### R8 — optional bounded keep-awake helper

```text
scripts/keep_awake_night.ps1
```

Commit:

```text
chore(ops): add bounded Windows keep-awake helper
```

Если пользователь не хочет хранить этот opt-in helper, STOP и отдельно решить
его судьбу; нельзя молча удалить или оставить забытым untracked.

## 10. Exact-HEAD tests и первый root push

Из-за оставшихся unstaged будущих файлов тесты в основном worktree не доказывают
корректность отдельного commit. После R2–R7 предпочтительно создавать временный
detached worktree выбранного `HEAD`, запускать compile/targeted tests через
`$Python`, затем удалять только точно проверенный временный путь.

Перед первым root push обязательны:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File (Join-Path $Root 'scripts\validate_install.ps1') `
  -PythonPath $Python

& $Python -m pytest -q -p no:cacheprovider
```

Сохранить фактические stdout summaries и exit codes в backup execution logs.
Тесты не доказывают edge или live readiness.

Также повторить calibration triage:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File (Join-Path $Root 'scripts\observation\triage_calibration_result.ps1') `
  -ComparisonJson (Join-Path $Root `
    'logs\observation\comparison_pre_main_20260723_1954.json')
```

Ожидаемый смысл: `pause_calibration / window_size_not_comparable`.

Перед push:

- root final candidate tip не содержит `.env`;
- remote root tip повторно получен live;
- remote является ancestor local HEAD;
- secret gate пройден;
- `git push --dry-run` успешен.

Первый push отправляет R1–R8 без docs:

```powershell
git push --porcelain origin `
  HEAD:refs/heads/feat/phase2-layer1-pump-runtime-alignment
```

После push live remote hash обязан равняться local HEAD.

## 11. Этап F — документация последней

После подтверждённых MEXC и первого root push обновить:

```text
README.md
CLAUDE.md
docs/AI_HANDOFF.md
docs/PROJECT_HANDOFF_FOR_CLAUDE.md
docs/PRESERVATION_PLAN_2026-07-30.md
docs/PRESERVATION_PLAN_v3_2026-08-01.md
```

Требования:

- старый план помечен `SUPERSEDED — DO NOT EXECUTE`;
- актуальная дата и фактические base/final hashes;
- фактические test summaries;
- внешний backup path, manifest hash и `VERIFIED_OK`;
- MEXC remote sync и отсутствие `.env` в final tip;
- root final functional hash;
- bot stopped, paper-only;
- expectancy не доказана, calibration остаётся paused;
- `origin/main`/история `.env` остаются unresolved security item;
- собственный hash docs-коммита не вписывается внутрь него.

Docs commit:

```text
docs: record preserved runtime and research state
```

Второй root push выполняется тем же explicit refspec после повторных
staged/secret/fast-forward checks.

## 12. Финальная проверка и execution receipt

### 12.1 Remote

- root local HEAD == live root remote tip;
- MEXC local HEAD == live MEXC remote tip;
- root final tip не содержит `.env`;
- MEXC final tip не содержит `.env`;
- push range не содержит новых secret-bearing blobs;
- force push не применялся.

### 12.2 Worktrees

MEXC `status --porcelain` пуст; ignored local `.env` допустим.

Root index пуст, untracked non-ignored отсутствуют, а единственный допустимый
unstaged set равен:

```text
.idea/koteika_Ultra.iml
.idea/misc.xml
.idea/vcs.xml
logs/system.log
```

`.venv_relocated_20260723/` после R1 должен быть ignored и исчезнуть из status.
Удаление tracked noise — отдельный optional hygiene task.

### 12.3 Execution receipt

Во внешний backup записать `PRESERVATION_RESULT.json`:

- start/end timestamps;
- backup root и manifest SHA-256;
- root/MEXC before/after hashes;
- live remote before/after hashes;
- список commits и messages;
- фактические test summaries/exit codes;
- SQLite integrity/table-count report;
- rotation checklist без значений;
- MEXC/root final-tip secret checks;
- оставшиеся security/reproducibility blockers;
- `bot_state=stopped`, `runtime_mode=paper`.

Отдельно вычислить SHA-256 receipt и не изменять pre-mutation manifest.

## 13. Rollback и STOP-сценарии

- До commits: восстановление из full-file snapshot или binary patches.
- После local commits, до push: ничего не reset; при ошибке STOP и создать новый
  исправляющий commit либо запросить решение пользователя.
- После push: только revert/follow-up commit; никакого history rewrite без
  отдельного согласования.
- При remote divergence: не merge/rebase автоматически.
- При test failure: не ослаблять тест, threshold или safety gate ради push.
- При secret hit: не печатать совпадение, сообщить только rule/path/line.
- При backup mismatch: не продолжать даже если Git-коммиты выглядят безопасно.

## 14. Что сознательно остаётся после preservation

Не решается этим планом:

- history rewrite и очистка старых remote branches от `.env`;
- окончательный выбор MEXC/Bybit;
- causal implementation parity;
- доказательство edge;
- dependency lock/clean-install reconciliation;
- real per-order live cap и operator emergency stop;
- testnet/live execution;
- очистка IDE/log/recovery artifacts.

Следующий продуктовый этап начинается только после полного preservation и
явного подтверждения MEXC как целевой площадки: зафиксировать executable
single-position contract, реализовать недостающие causal features и выполнить
purged chronological evaluation против matched random baseline.

## 15. Реалистичная длительность

- preflight и создание каталогов: 5–10 минут;
- полный внешний backup + полная SHA-256 проверка: 30–90 минут, зависит от
  носителя;
- rotation: зависит от пользователя;
- MEXC commits/tests/push: 10–20 минут;
- root commits и exact staged checks: 30–60 минут;
- fresh suites: 5–15 минут;
- docs/final verification: 10–20 минут.

Оценка без rotation: **1.5–3 часа**, а не 45–60 минут.

## 16. Следующий этап: causal AI reversal advisory

Пользователь предложил добавить конкретную AI-модель, например Kimi K3, чтобы
лучше распознавать момент разворота. Правильная техническая формулировка — не
«модель чувствует рынок», а выдаёт проверяемую оценку:

```text
P(reversal before invalidation | information available at decision_ts)
```

вместе с горизонтом, uncertainty/abstention и неизменяемым provenance.

### 16.1 Выбор архитектуры

Нельзя помещать cloud LLM в hot decision path. Предлагаются два независимых
shadow-контура:

1. **Numeric reversal model** — основной кандидат для цены/объёма/MTF/flow:
   сначала существующий LightGBM/XGBoost/sklearn baseline, затем только при
   достаточных данных сравнение с causal TCN/LSTM/LOB-моделью.
2. **LLM context compiler** — асинхронно классифицирует только timestamped
   публичные новости/события. Kimi K3 допустим как research challenger, а более
   быстрый non-thinking model — как latency/cost baseline. LLM не получает
   позиции, balances, credentials, private strategy state и не возвращает
   action, entry, SL, TP, leverage или size.

```text
closed causal market inputs -> deterministic strategy -> immutable snapshot
                                                   |-> numeric ML shadow
public timestamped text -> separate worker -> structured LLM context shadow

all outputs -> append-only journals -> prospective outcomes -> honest ablation
```

Worker не импортирует `risk`, `execution`, exchange adapter или
`StrategyIntent`. Timeout, stale input, schema error и provider outage всегда
дают `unknown/abstain`; deterministic scanner продолжает работу независимо.

### 16.2 P0 до первого AI-вызова

1. Завершить preservation и rotation.
2. Подтвердить MEXC как целевой research venue.
3. Зафиксировать executable single-position contract.
4. В MEXC scanner исключать forming base/15m/1h/4h candles по правилу
   `bar_open_ts + bar_seconds <= decision_ts`.
5. Устранить shared mutable `LayeredPumpStrategy`/diagnostics между
   parallel-symbol workers.
6. Сохранять реальные `decision_ts`, `base_bar_close_ts`, source observation
   timestamps и deterministic `snapshot_id`.
7. Переместить/перестроить root ML snapshot так, чтобы он соответствовал
   финальному deterministic intent после всех late guards.
8. Свести `ML_INFERENCE_ENABLED`/`ML_SHADOW_ENABLED` к одному fail-closed
   source of truth; ungoverned artifact не может считаться enabled.

До выполнения этих пунктов AI-код не должен делать network calls.

### 16.3 Реализация по фазам

**AI-1 — context journal only:** immutable `CausalDecisionSnapshot`, строгий
feature/schema hash, append-only runtime-population JSONL; provider выключен.

**AI-2 — governed numeric shadow:** labels означают один вход и путь
`reversal-before-invalidation`, данные разделяются хронологически с purge/
embargo; prediction не меняет intent.

**AI-3 — isolated LLM worker:** strict structured schema, bounded timeout,
concurrency и budget; отдельный worker-only API key; frozen provider/model/
prompt/schema version; input/output journals и idempotency.

**AI-4 — prospective MEXC observation:** модель обрабатывает только события,
появившиеся после deployment timestamp. Нельзя запускать сегодняшнюю LLM на
старых новостях и называть это causal backtest: training data модели может уже
содержать будущий исход.

**AI-5 — evaluation:** deterministic-only против numeric-shadow против
numeric+context; matched random baseline, fees/spread/slippage/funding,
chronological purge, symbol-clustered confidence intervals, calibration и
abstention quality. Повторный no-edge принимается без подгонки.

Только устойчивый prospective incremental edge может разрешить отдельное
human-only сообщение `AI SHADOW — НЕ ТОРГОВЫЙ СИГНАЛ`. Ни одна фаза этого плана
не даёт AI право управлять admission, risk, sizing или execution.

---

Короткое правило исполнения:

> Сначала независимый проверенный backup, затем ротация и sanitised MEXC tip,
> затем топологически корректные commits с тестами, после этого explicit
> fast-forward pushes и только в самом конце handoff. Любое расхождение означает
> STOP, а не автоматическое «исправление».
