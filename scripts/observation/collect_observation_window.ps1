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
    [string]$ObservationTimeframe = "1",

    [Parameter()]
    [string]$MainRuntimeDb = "",

    [Parameter()]
    [string]$EarlyRuntimeDb = "",

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

function Get-QualityProfileSummary {
    param(
        [Parameter()]
        [object]$SummaryRoot,
        [Parameter(Mandatory = $true)]
        [string]$Profile
    )

    if ($null -eq $SummaryRoot) {
        return $null
    }
    try {
        return $SummaryRoot.$Profile
    }
    catch {
        return $null
    }
}

function Get-QualityProfileCount {
    param(
        [Parameter()]
        [object]$ProfileSummary
    )

    if ($null -eq $ProfileSummary) {
        return 0
    }
    try {
        return [int]($ProfileSummary.count)
    }
    catch {
        return 0
    }
}

function Get-TopVerdictLabel {
    param(
        [Parameter()]
        [object]$ProfileSummary
    )

    if ($null -eq $ProfileSummary) {
        return ""
    }

    $verdictCounts = $null
    try {
        $verdictCounts = $ProfileSummary.verdict_counts
    }
    catch {
        $verdictCounts = $null
    }
    if ($null -eq $verdictCounts) {
        return ""
    }

    $topName = ""
    $topCount = -1
    foreach ($prop in $verdictCounts.PSObject.Properties) {
        $name = [string]$prop.Name
        $count = 0
        try {
            $count = [int]$prop.Value
        }
        catch {
            $count = 0
        }
        if ($count -gt $topCount -or ($count -eq $topCount -and ($topName -eq "" -or $name -lt $topName))) {
            $topCount = $count
            $topName = $name
        }
    }
    return $topName
}

function Build-QualityOverview {
    param(
        [Parameter()]
        [object]$SignalQuality,
        [Parameter()]
        [object]$ExitQuality,
        [Parameter()]
        [string]$ObservationTimeframe = "1"
    )

    $signalSummaryRoot = if ($null -ne $SignalQuality) { $SignalQuality.summary } else { $null }
    $exitSummaryRoot = if ($null -ne $ExitQuality) { $ExitQuality.summary } else { $null }

    $signalMain = Get-QualityProfileSummary -SummaryRoot $signalSummaryRoot -Profile "main"
    $signalEarly = Get-QualityProfileSummary -SummaryRoot $signalSummaryRoot -Profile "early"
    $exitMain = Get-QualityProfileSummary -SummaryRoot $exitSummaryRoot -Profile "main"
    $exitEarly = Get-QualityProfileSummary -SummaryRoot $exitSummaryRoot -Profile "early"

    $signalMainCount = Get-QualityProfileCount -ProfileSummary $signalMain
    $signalEarlyCount = Get-QualityProfileCount -ProfileSummary $signalEarly
    $exitMainCount = Get-QualityProfileCount -ProfileSummary $exitMain
    $exitEarlyCount = Get-QualityProfileCount -ProfileSummary $exitEarly

    return [ordered]@{
        observation_timeframe = [string]$ObservationTimeframe
        signal_status = if ($null -ne $SignalQuality) { [string]$SignalQuality.status } else { "" }
        exit_status = if ($null -ne $ExitQuality) { [string]$ExitQuality.status } else { "" }
        signal_main_count = $signalMainCount
        signal_early_count = $signalEarlyCount
        signal_total_count = ($signalMainCount + $signalEarlyCount)
        exit_main_count = $exitMainCount
        exit_early_count = $exitEarlyCount
        exit_total_count = ($exitMainCount + $exitEarlyCount)
        signal_main_top_verdict = Get-TopVerdictLabel -ProfileSummary $signalMain
        signal_early_top_verdict = Get-TopVerdictLabel -ProfileSummary $signalEarly
        exit_main_top_verdict = Get-TopVerdictLabel -ProfileSummary $exitMain
        exit_early_top_verdict = Get-TopVerdictLabel -ProfileSummary $exitEarly
    }
}

function Invoke-PythonJsonCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Exe,
        [Parameter(Mandatory = $true)]
        [string[]]$Args,
        [Parameter()]
        [string]$OutJson = ""
    )

    $stdoutFile = Join-Path ([System.IO.Path]::GetTempPath()) ("obs_collect_stdout_" + [System.Guid]::NewGuid().ToString("N") + ".log")
    $stderrFile = Join-Path ([System.IO.Path]::GetTempPath()) ("obs_collect_stderr_" + [System.Guid]::NewGuid().ToString("N") + ".log")
    try {
        $previousErr = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $Exe @Args 1> $stdoutFile 2> $stderrFile
            $exitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $previousErr
        }

        $stdoutText = if (Test-Path $stdoutFile) { Get-Content -Raw -Path $stdoutFile } else { "" }
        $stderrText = if (Test-Path $stderrFile) { Get-Content -Raw -Path $stderrFile } else { "" }

        $result = [ordered]@{
            status = "error"
            exit_code = [int]$exitCode
            json_path = ""
            summary = $null
            payload = $null
            stderr = [string]$stderrText
        }

        if ($exitCode -ne 0 -or [string]::IsNullOrWhiteSpace($stdoutText)) {
            return $result
        }

        try {
            $payload = $stdoutText | ConvertFrom-Json -ErrorAction Stop
        }
        catch {
            $result.stderr = (([string]$stderrText).TrimEnd() + "`njson_parse_error=" + $_.Exception.Message).Trim()
            return $result
        }

        if (-not [string]::IsNullOrWhiteSpace($OutJson)) {
            $outDir = Split-Path -Parent $OutJson
            if (-not [string]::IsNullOrWhiteSpace($outDir)) {
                New-Item -ItemType Directory -Path $outDir -Force | Out-Null
            }
            ($payload | ConvertTo-Json -Depth 20) | Set-Content -Path $OutJson -Encoding UTF8
            $result.json_path = (Resolve-Path $OutJson).Path
        }
        $summaryPayload = $null
        $summaryProperty = $payload.PSObject.Properties["summary"]
        if ($null -ne $summaryProperty) {
            $summaryPayload = $summaryProperty.Value
        }
        $result.status = "ok"
        $result.summary = $summaryPayload
        $result.payload = $payload
        return $result
    }
    finally {
        Remove-Item -Force $stdoutFile, $stderrFile -ErrorAction SilentlyContinue
    }
}

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
    $signalQualityJson = Join-Path $LogDir ("signal_quality_{0}_{1}.json" -f $safeTag, $ts)
    $exitQualityJson = Join-Path $LogDir ("exit_quality_{0}_{1}.json" -f $safeTag, $ts)
    $qualityGuidanceJson = Join-Path $LogDir ("quality_guidance_{0}_{1}.json" -f $safeTag, $ts)

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

    $signalQualityResult = $null
    $signalQualityScript = Join-Path $PSScriptRoot "analyze_recent_signal_quality.py"
    if (Test-Path $signalQualityScript) {
        $signalArgs = @($signalQualityScript, "--timeframe", $ObservationTimeframe)
        if (-not [string]::IsNullOrWhiteSpace($MainRuntimeDb)) {
            $signalArgs += @("--main-db", $MainRuntimeDb)
        }
        if (-not [string]::IsNullOrWhiteSpace($EarlyRuntimeDb)) {
            $signalArgs += @("--early-db", $EarlyRuntimeDb)
        }
        $signalQualityResult = Invoke-PythonJsonCommand -Exe $PythonPath -Args $signalArgs -OutJson $signalQualityJson
    }

    $exitQualityResult = $null
    $exitQualityScript = Join-Path $PSScriptRoot "analyze_recent_exit_quality.py"
    if (Test-Path $exitQualityScript) {
        $exitArgs = @($exitQualityScript, "--timeframe", $ObservationTimeframe)
        if (-not [string]::IsNullOrWhiteSpace($MainRuntimeDb)) {
            $exitArgs += @("--main-db", $MainRuntimeDb)
        }
        if (-not [string]::IsNullOrWhiteSpace($EarlyRuntimeDb)) {
            $exitArgs += @("--early-db", $EarlyRuntimeDb)
        }
        $exitQualityResult = Invoke-PythonJsonCommand -Exe $PythonPath -Args $exitArgs -OutJson $exitQualityJson
    }

    $qualityOverview = Build-QualityOverview `
        -SignalQuality $signalQualityResult `
        -ExitQuality $exitQualityResult `
        -ObservationTimeframe $ObservationTimeframe

    $qualityGuidanceResult = $null
    $qualityGuidanceScript = Join-Path $PSScriptRoot "build_quality_guidance.py"
    if (Test-Path $qualityGuidanceScript) {
        $guidanceArgs = @($qualityGuidanceScript, "--timeframe", $ObservationTimeframe)
        if ($signalQualityResult -and $signalQualityResult.status -eq "ok" -and -not [string]::IsNullOrWhiteSpace($signalQualityResult.json_path)) {
            $guidanceArgs += @("--signal-json", $signalQualityResult.json_path)
        }
        if ($exitQualityResult -and $exitQualityResult.status -eq "ok" -and -not [string]::IsNullOrWhiteSpace($exitQualityResult.json_path)) {
            $guidanceArgs += @("--exit-json", $exitQualityResult.json_path)
        }
        $qualityGuidanceResult = Invoke-PythonJsonCommand -Exe $PythonPath -Args $guidanceArgs -OutJson $qualityGuidanceJson
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
        signal_quality = $signalQualityResult
        exit_quality = $exitQualityResult
        quality_overview = $qualityOverview
        quality_guidance = if ($qualityGuidanceResult) { $qualityGuidanceResult.payload } else { $null }
    }

    $result | ConvertTo-Json -Depth 8
}
finally {
    Pop-Location
}


