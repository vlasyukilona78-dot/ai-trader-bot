param(
    [string]$RuntimeMetaJson = "",
    [string]$RuntimeLog = "",
    [string]$PythonPath = ".\\.venv\\Scripts\\python.exe",
    [string]$OutDir = "logs\\observation",
    [switch]$PrintTriage,
    [string]$ObservationTimeframe = "1",
    [string]$MainRuntimeDb = "",
    [string]$EarlyRuntimeDb = ""
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

    $stdoutFile = Join-Path ([System.IO.Path]::GetTempPath()) ("obs_runtime_stdout_" + [System.Guid]::NewGuid().ToString("N") + ".log")
    $stderrFile = Join-Path ([System.IO.Path]::GetTempPath()) ("obs_runtime_stderr_" + [System.Guid]::NewGuid().ToString("N") + ".log")
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

function Resolve-RuntimeLog {
    param(
        [string]$MetaJson,
        [string]$DirectLog
    )
    if ($DirectLog -and (Test-Path $DirectLog)) {
        return (Resolve-Path $DirectLog).Path
    }
    if (-not $MetaJson -or -not (Test-Path $MetaJson)) {
        throw "Runtime log not found and RuntimeMetaJson missing."
    }
    $meta = Get-Content $MetaJson -Raw | ConvertFrom-Json
    if (-not $meta.stderr_log) {
        throw "stderr_log missing in runtime meta JSON."
    }
    return (Resolve-Path $meta.stderr_log).Path
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "../..")
Push-Location $repoRoot
try {
    $runtimeLogPath = Resolve-RuntimeLog -MetaJson $RuntimeMetaJson -DirectLog $RuntimeLog
    if (-not (Test-Path $runtimeLogPath)) {
        throw "Runtime log not found: $runtimeLogPath"
    }

    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

    $stem = [IO.Path]::GetFileNameWithoutExtension($runtimeLogPath)
    $extractLog = Join-Path $OutDir ("{0}.audit_extract.log" -f $stem)
    $summaryJson = Join-Path $OutDir ("{0}.summary.json" -f $stem)
    $manifestJson = Join-Path $OutDir ("{0}.analysis.json" -f $stem)
    $signalQualityJson = Join-Path $OutDir ("{0}.signal_quality.json" -f $stem)
    $exitQualityJson = Join-Path $OutDir ("{0}.exit_quality.json" -f $stem)
    $qualityGuidanceJson = Join-Path $OutDir ("{0}.quality_guidance.json" -f $stem)

    $auditPattern = "strategy_audit_compact=|strategy_audit_regime_filter=|strategy_audit_source_quality=|strategy_audit="
    Get-Content $runtimeLogPath | Where-Object { $_ -match $auditPattern } | Set-Content $extractLog

    $summaryScript = Join-Path $PSScriptRoot "summarize_observation.ps1"
    if (-not (Test-Path $summaryScript)) {
        throw "summarize_observation.ps1 not found: $summaryScript"
    }

    $summaryRaw = & $summaryScript -AuditExtractFile $extractLog -OutJson $summaryJson -PythonPath $PythonPath -PrintTriage:$PrintTriage
    $summaryObj = $null
    try {
        $summaryObj = $summaryRaw | ConvertFrom-Json
    }
    catch {
        $summaryObj = $summaryRaw
    }

    $signalQualityObj = $null
    $signalQualityScript = Join-Path $PSScriptRoot "analyze_recent_signal_quality.py"
    if (Test-Path $signalQualityScript) {
        $signalArgs = @($signalQualityScript, "--timeframe", $ObservationTimeframe)
        if (-not [string]::IsNullOrWhiteSpace($MainRuntimeDb)) {
            $signalArgs += @("--main-db", $MainRuntimeDb)
        }
        if (-not [string]::IsNullOrWhiteSpace($EarlyRuntimeDb)) {
            $signalArgs += @("--early-db", $EarlyRuntimeDb)
        }
        $signalQualityObj = Invoke-PythonJsonCommand -Exe $PythonPath -Args $signalArgs -OutJson $signalQualityJson
    }

    $exitQualityObj = $null
    $exitQualityScript = Join-Path $PSScriptRoot "analyze_recent_exit_quality.py"
    if (Test-Path $exitQualityScript) {
        $exitArgs = @($exitQualityScript, "--timeframe", $ObservationTimeframe)
        if (-not [string]::IsNullOrWhiteSpace($MainRuntimeDb)) {
            $exitArgs += @("--main-db", $MainRuntimeDb)
        }
        if (-not [string]::IsNullOrWhiteSpace($EarlyRuntimeDb)) {
            $exitArgs += @("--early-db", $EarlyRuntimeDb)
        }
        $exitQualityObj = Invoke-PythonJsonCommand -Exe $PythonPath -Args $exitArgs -OutJson $exitQualityJson
    }

    $qualityOverview = Build-QualityOverview `
        -SignalQuality $signalQualityObj `
        -ExitQuality $exitQualityObj `
        -ObservationTimeframe $ObservationTimeframe

    $qualityGuidanceObj = $null
    $qualityGuidanceScript = Join-Path $PSScriptRoot "build_quality_guidance.py"
    if (Test-Path $qualityGuidanceScript) {
        $guidanceArgs = @($qualityGuidanceScript, "--timeframe", $ObservationTimeframe)
        if ($signalQualityObj -and $signalQualityObj.status -eq "ok" -and -not [string]::IsNullOrWhiteSpace($signalQualityObj.json_path)) {
            $guidanceArgs += @("--signal-json", $signalQualityObj.json_path)
        }
        if ($exitQualityObj -and $exitQualityObj.status -eq "ok" -and -not [string]::IsNullOrWhiteSpace($exitQualityObj.json_path)) {
            $guidanceArgs += @("--exit-json", $exitQualityObj.json_path)
        }
        $qualityGuidanceObj = Invoke-PythonJsonCommand -Exe $PythonPath -Args $guidanceArgs -OutJson $qualityGuidanceJson
    }

    $manifest = [ordered]@{
        generated_at = (Get-Date).ToString("o")
        runtime_log = $runtimeLogPath
        extract_log = (Resolve-Path $extractLog).Path
        summary_json = (Resolve-Path $summaryJson).Path
        summary = $summaryObj
        signal_quality = $signalQualityObj
        exit_quality = $exitQualityObj
        quality_overview = $qualityOverview
        quality_guidance = if ($qualityGuidanceObj) { $qualityGuidanceObj.payload } else { $null }
    }
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $manifestJson -Encoding UTF8
    $manifest | ConvertTo-Json -Depth 8
}
finally {
    Pop-Location
}
