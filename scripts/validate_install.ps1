param(
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not $PythonPath) {
    $PythonPath = Join-Path $root ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Python runtime not found: $PythonPath"
}

Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$env:MPLCONFIGDIR = Join-Path $root "data\runtime\matplotlib"
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null

& $PythonPath -c "import numpy, pandas, requests, sklearn, xgboost, lightgbm; print('dependencies=ok')"
if ($LASTEXITCODE -ne 0) {
    throw "Dependency import check failed."
}

& $PythonPath -m compileall -q app core trading ai alerts
if ($LASTEXITCODE -ne 0) {
    throw "Python compilation check failed."
}

& $PythonPath -m app.main --help *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Application entry-point check failed."
}

& $PythonPath -m pytest `
    tests\v2\test_feature_pipeline_v2.py `
    tests\v2\test_entry_gate_v2.py `
    tests\v2\test_signal_position_tracker_v2.py `
    tests\v2\test_ml_shadow_integration_v2.py `
    -q
if ($LASTEXITCODE -ne 0) {
    throw "Critical smoke tests failed."
}

[pscustomobject]@{
    status = "ok"
    python = (& $PythonPath -c "import sys; print(sys.version.split()[0])")
    executable = $PythonPath
} | ConvertTo-Json -Compress
