param()

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$runner = Join-Path $PSScriptRoot "run_bot_infinite.ps1"
if (-not (Test-Path $runner)) {
    throw "run_bot_infinite.ps1 not found."
}

$mainResult = & $runner -SignalProfile main
$earlyResult = & $runner -SignalProfile early

Write-Output "main_profile"
$mainResult
Write-Output "early_profile"
$earlyResult
