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

If you already have direct drill-down artifacts from the same window, attach them here as well:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\observation\summarize_observation.ps1 `
  -AuditExtractFile "<after_extract.log>" `
  -CompareAuditExtractFile "<before_extract.log>" `
  -SignalQualityJson ".\logs\observation\signal_quality_latest.json" `
  -ExitQualityJson ".\logs\observation\exit_quality_latest.json" `
  -ObservationTimeframe 1 `
  -OutJson ".\logs\observation\comparison_latest.json" `
  -PrintTriage
```

That lets the final comparison JSON include:

- `quality_guidance`
- `quality_overview`

so the triage step can speak not only about blocker calibration, but also about what to tighten or loosen in entries and exits.

## 1b) End-to-end observation run with final triage

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\observation\collect_observation_window.ps1 `
  -DurationMinutes 15 `
  -PauseSeconds 10 `
  -Tag "post_change" `
  -PrintTriage
```

For a quick smoke check of the final step without waiting for a full window, add `-MaxRuns 1`.

That end-to-end run now also tries to attach two extra drill-down artifacts automatically:

- `signal_quality_*.json`
- `exit_quality_*.json`

so one observation window can be reviewed for blocker calibration, entry timing, and exit timing together.

The returned manifest also includes a compact `quality_overview` block so you can immediately see:

- how many `main` / `early` signals were reviewed
- how many `main` / `early` exits were reviewed
- which verdict currently dominates for each reviewed profile

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
QUALITY_ENTRY: <entry_focus> priority=<entry_priority>
QUALITY_EXIT: <exit_focus> priority=<exit_priority>
QUALITY_ACTION 1: ...
QUALITY_ACTION 2: ...
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

## 4) Recent signal quality drill-down

After the calibration summary, run the direct signal-quality analyzer on the latest runtime window:

```powershell
powershell -ExecutionPolicy Bypass -Command "& '.\.venv\Scripts\python.exe' '.\scripts\observation\analyze_recent_signal_quality.py' --main-limit 12 --early-limit 12 --timeframe 1"
```

What this adds on top of the blocker summary:

- per-signal verdicts such as `worked`, `late_or_weak`, `continuation_trap`, `too_early`
- favorable/adverse excursion over `15m` and `60m`
- reaction latency: how quickly the first `0.35%` favorable or adverse move appeared
- short-horizon drill-down over `3/5/10/20` bars
- automatic enrichment of recent `early` alerts with real `entry/tp/sl` from the early runtime DB when a later execution exists
- top signal verdicts are also surfaced into observation manifests through `quality_overview`

## 5) Recent exit quality drill-down

To audit managed exits and see whether shorts were closed too early or on time:

```powershell
powershell -ExecutionPolicy Bypass -Command "& '.\.venv\Scripts\python.exe' '.\scripts\observation\analyze_recent_exit_quality.py' --main-limit 12 --early-limit 12 --timeframe 1"
```

What this adds:

- per-exit verdicts such as `timely`, `too_early`, `protective_exit`, `late_or_bad`
- post-exit move analysis: how much more downside remained vs how much rebound happened after the close
- reaction latency after exit: time to the first `0.35%` further down move or rebound
- short-horizon exit drill-down over `3/5/10/20` bars
- summary buckets by `exit_type` and `managed_exit_reason`, so managed-exit tuning can be driven by evidence instead of screenshots
- top exit verdicts are also surfaced into observation manifests through `quality_overview`
