param(
    [string]$PythonPath = "C:\Users\Zephyrus\AppData\Local\Programs\Python\Python314\python.exe",
    [string]$EntryPoint = ".\main.py"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$runtimeDir = Join-Path $root "logs\runtime"
New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdoutLog = Join-Path $runtimeDir ("bot_supervisor_{0}.stdout.log" -f $stamp)
$stderrLog = Join-Path $runtimeDir ("bot_supervisor_{0}.stderr.log" -f $stamp)

[Environment]::SetEnvironmentVariable("PYTHONHOME", $null, "Process")
[Environment]::SetEnvironmentVariable("PYTHONPATH", $null, "Process")
[Environment]::SetEnvironmentVariable("VIRTUAL_ENV", $null, "Process")

if (Test-Path Env:PATH) {
    Remove-Item Env:PATH -ErrorAction SilentlyContinue
}

$proc = Start-Process `
    -FilePath $PythonPath `
    -ArgumentList @("-u", $EntryPoint) `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

Start-Sleep -Seconds 3

[pscustomobject]@{
    pid = $proc.Id
    stdout = $stdoutLog
    stderr = $stderrLog
} | ConvertTo-Json -Compress
