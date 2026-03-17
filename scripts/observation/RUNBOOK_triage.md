# Calibration Triage Quick Runbook

## 1) Build comparison JSON

Replace the `<...>` placeholders below with real log paths before running the command.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\observation\summarize_observation.ps1 `
  -AuditExtractFile "<after_extract.log>" `
  -CompareAuditExtractFile "<before_extract.log>" `
  -OutJson ".\logs\observation\comparison_latest.json"
```

Optional shortcut: add `-PrintTriage` to the summarize command to print the operator verdict immediately after the summary step.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\observation\summarize_observation.ps1 `
  -AuditExtractFile "<after_extract.log>" `
  -CompareAuditExtractFile "<before_extract.log>" `
  -OutJson ".\logs\observation\comparison_latest.json" `
  -PrintTriage
```

## 1b) End-to-end observation run with final triage

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\observation\collect_observation_window.ps1 `
  -DurationMinutes 15 `
  -PauseSeconds 10 `
  -Tag "post_change" `
  -PrintTriage
```

For a quick smoke check of the final step without waiting for a full window, add `-MaxRuns 1`.

## 2) Read the operator verdict through the PowerShell wrapper

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\observation\triage_calibration_result.ps1 `
  -ComparisonJson ".\logs\observation\comparison_latest.json"
```

## 3) Expected output

```text
VERDICT: <action_verdict>
STOP_REASON: <stop_reason>
TOP_COMBINATION: <top_regime_filter_blocker_combination>
ACTION 1: ...
ACTION 2: ...
ACTION 3: ...
ACTION 4: ...
```

## Exit codes

```text
0  = safe single-target continuation
10 = co-dominant overlap stop
11 = blocker semantics disagreement
12 = market context shift
13 = non-comparable window
```
