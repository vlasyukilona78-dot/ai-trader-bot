param(
    [string]$RuntimeMetaJson = "",
    [string]$RuntimeLog = "",
    [string]$PythonPath = ".\\.venv\\Scripts\\python.exe",
    [string]$OutDir = "logs\\observation",
    [switch]$PrintTriage
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

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

    $manifest = [ordered]@{
        generated_at = (Get-Date).ToString("o")
        runtime_log = $runtimeLogPath
        extract_log = (Resolve-Path $extractLog).Path
        summary_json = (Resolve-Path $summaryJson).Path
        summary = $summaryObj
    }
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $manifestJson -Encoding UTF8
    $manifest | ConvertTo-Json -Depth 8
}
finally {
    Pop-Location
}
