<#
Usage:
  powershell -ExecutionPolicy Bypass -File .\scripts\observation\collect_observation_window.ps1 -DurationMinutes 15 -PauseSeconds 10 -Tag post_htf_auto -PrintTriage
  powershell -ExecutionPolicy Bypass -File .\scripts\observation\collect_observation_window.ps1 -DurationMinutes 15 -PauseSeconds 10 -Tag smoke_check -PrintTriage -MaxRuns 1

What this script does:
  1) runs app.main in a controlled loop for a fixed observation window
  2) writes a combined runtime log
  3) extracts health lines containing strategy_audit_* payloads
  4) optionally calls summarize_observation.ps1 to produce summary JSON
#>

[CmdletBinding()]
param(
    [Parameter()]
    [ValidateRange(1, 1440)]
    [int]$DurationMinutes = 15,

    [Parameter()]
    [ValidateRange(1, 3600)]
    [int]$PauseSeconds = 10,

    [Parameter()]
    [string]$Tag = "post_change",

    [Parameter()]
    [string]$PythonPath = ".\\.venv\\Scripts\\python.exe",

    [Parameter()]
    [string]$LogDir = "logs/observation",

    [Parameter()]
    [switch]$PrintTriage,

    [Parameter()]
    [ValidateRange(0, 100000)]
    [int]$MaxRuns = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "../..")
Push-Location $repoRoot
try {
    if (-not (Test-Path $PythonPath)) {
        throw "Python executable not found: $PythonPath"
    }

    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $safeTag = ($Tag -replace "[^a-zA-Z0-9_-]", "_").Trim("_")
    if ([string]::IsNullOrWhiteSpace($safeTag)) {
        $safeTag = "window"
    }

    $combinedLog = Join-Path $LogDir ("regime_observation_window_{0}_{1}.log" -f $safeTag, $ts)
    $extractLog = Join-Path $LogDir ("regime_audit_extract_{0}_{1}.log" -f $safeTag, $ts)
    $summaryJson = Join-Path $LogDir ("regime_observation_summary_{0}_{1}.json" -f $safeTag, $ts)

    $startedAt = Get-Date
    $stopAt = $startedAt.AddMinutes($DurationMinutes)
    $runs = 0
    $runFailures = 0

    Write-Host "[collect] start=$($startedAt.ToString('o')) stop=$($stopAt.ToString('o')) duration_min=$DurationMinutes pause_sec=$PauseSeconds"
    Write-Host "[collect] combined_log=$combinedLog"

    while ((Get-Date) -lt $stopAt) {
        if ($MaxRuns -gt 0 -and $runs -ge $MaxRuns) {
            break
        }

        $runs += 1

        $tmpStdout = Join-Path $LogDir ("tmp_obs_stdout_{0}_{1}.log" -f $ts, $runs)
        $tmpStderr = Join-Path $LogDir ("tmp_obs_stderr_{0}_{1}.log" -f $ts, $runs)

        $proc = Start-Process -FilePath $PythonPath -ArgumentList "-m", "app.main" -NoNewWindow -Wait -PassThru -RedirectStandardOutput $tmpStdout -RedirectStandardError $tmpStderr

        if (Test-Path $tmpStdout) {
            Get-Content $tmpStdout | Add-Content -Path $combinedLog
            Remove-Item -Force $tmpStdout -ErrorAction SilentlyContinue
        }
        if (Test-Path $tmpStderr) {
            Get-Content $tmpStderr | Add-Content -Path $combinedLog
            Remove-Item -Force $tmpStderr -ErrorAction SilentlyContinue
        }

        if ($proc.ExitCode -ne 0) {
            $runFailures += 1
            Add-Content -Path $combinedLog -Value ("[collector] run={0} exit_code={1} ts={2}" -f $runs, $proc.ExitCode, (Get-Date).ToString("o"))
        }

        if ((Get-Date) -ge $stopAt) {
            break
        }
        if ($MaxRuns -gt 0 -and $runs -ge $MaxRuns) {
            break
        }
        Start-Sleep -Seconds $PauseSeconds
    }

    $auditPattern = "strategy_audit_compact=|strategy_audit_regime_filter=|strategy_audit_source_quality=|strategy_audit="
    Get-Content $combinedLog | Where-Object { $_ -match $auditPattern } | Set-Content $extractLog

    $summaryScript = Join-Path $PSScriptRoot "summarize_observation.ps1"
    $summaryResult = $null
    if (Test-Path $summaryScript) {
        $summaryRaw = & $summaryScript -AuditExtractFile $extractLog -OutJson $summaryJson -PythonPath $PythonPath -PrintTriage:$PrintTriage
        try {
            $summaryResult = $summaryRaw | ConvertFrom-Json
        }
        catch {
            $summaryResult = $summaryRaw
        }
    }

    $result = [ordered]@{
        duration_minutes = $DurationMinutes
        pause_seconds = $PauseSeconds
        max_runs = $MaxRuns
        runs = $runs
        run_failures = $runFailures
        started_at = $startedAt.ToString("o")
        ended_at = (Get-Date).ToString("o")
        combined_log = $combinedLog
        extract_log = $extractLog
        summary_json = $summaryJson
        summary = $summaryResult
    }

    $result | ConvertTo-Json -Depth 8
}
finally {
    Pop-Location
}


