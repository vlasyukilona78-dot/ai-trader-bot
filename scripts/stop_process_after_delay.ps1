param(
    [Parameter(Mandatory = $true)]
    [int]$ProcessId,
    [Parameter(Mandatory = $true)]
    [int]$DelaySeconds,
    [string]$ControlLog = "",
    [string]$SummaryScript = "",
    [string]$RuntimeLog = "",
    [string]$RuntimeMetaJson = "",
    [string]$PythonPath = ".\\.venv\\Scripts\\python.exe",
    [string]$SummaryDir = "logs\\observation",
    [switch]$PrintTriage
)

Start-Sleep -Seconds $DelaySeconds

try {
    Stop-Process -Id $ProcessId -Force -ErrorAction Stop
    if ($ControlLog) {
        Add-Content -Path $ControlLog -Value "auto_stop ok pid=$ProcessId"
    }
}
catch {
    if ($ControlLog) {
        Add-Content -Path $ControlLog -Value "auto_stop failed pid=$ProcessId err=$($_.Exception.Message)"
    }
}

if ($SummaryScript -and (Test-Path $SummaryScript)) {
    try {
        & $SummaryScript -RuntimeLog $RuntimeLog -RuntimeMetaJson $RuntimeMetaJson -PythonPath $PythonPath -OutDir $SummaryDir -PrintTriage:$PrintTriage | Out-Null
        if ($ControlLog) {
            Add-Content -Path $ControlLog -Value "runtime_summary ok"
        }
    }
    catch {
        if ($ControlLog) {
            Add-Content -Path $ControlLog -Value "runtime_summary failed err=$($_.Exception.Message)"
        }
    }
}
