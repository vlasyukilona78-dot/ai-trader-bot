param(
    [double]$DurationHours = 4
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$runtimeDir = Join-Path $root "logs\runtime"
New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdoutLog = Join-Path $runtimeDir ("bot_loop_{0}.stdout.log" -f $stamp)
$stderrLog = Join-Path $runtimeDir ("bot_loop_{0}.stderr.log" -f $stamp)
$controlLog = Join-Path $runtimeDir ("bot_loop_{0}.control.log" -f $stamp)
$metaPath = Join-Path $runtimeDir ("bot_loop_{0}.json" -f $stamp)
$summaryScript = Join-Path $root "scripts\\observation\\summarize_runtime_window_after_process.ps1"
$summaryDir = Join-Path $root "logs\\observation"

$pythonCandidates = @(
    (Join-Path $root ".runtime_env\Scripts\python.exe"),
    (Join-Path $root ".venv\Scripts\python.exe")
)
$python = $null
foreach ($candidate in $pythonCandidates) {
    if (-not (Test-Path $candidate)) {
        continue
    }
    try {
        & $candidate -V *> $null
        if ($LASTEXITCODE -eq 0) {
            $python = $candidate
            break
        }
    } catch {
        continue
    }
}
if (-not $python) {
    throw "No Python runtime found in .runtime_env or .venv."
}
$mainPy = Join-Path $root "main.py"

$pythonCmd = '"' + $python + '" "' + $mainPy + '" --loop 1>>"' + $stdoutLog + '" 2>>"' + $stderrLog + '"'
cmd.exe /c ('start "" /b ' + $pythonCmd) | Out-Null
Start-Sleep -Seconds 2
$proc = Get-Process python -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -eq $python } |
    Sort-Object StartTime -Descending |
    Select-Object -First 1
if (-not $proc) {
    throw "Failed to start bot process."
}
$stopAt = (Get-Date).AddHours($DurationHours)

$meta = [ordered]@{
    pid = $proc.Id
    started_at = (Get-Date).ToString("o")
    stop_at = $stopAt.ToString("o")
    stdout_log = $stdoutLog
    stderr_log = $stderrLog
    control_log = $controlLog
    summary_script = $summaryScript
    summary_dir = $summaryDir
}
$meta | ConvertTo-Json | Set-Content -Path $metaPath -Encoding UTF8

Add-Content -Path $controlLog -Value ("started pid={0}" -f $proc.Id)
Add-Content -Path $controlLog -Value ("stop_at={0}" -f $stopAt.ToString("o"))

$delaySeconds = [Math]::Max(1, [int]([TimeSpan]::FromHours($DurationHours).TotalSeconds))
$stopScript = Join-Path $PSScriptRoot "stop_process_after_delay.ps1"
$stopCmd = '"C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe" -ExecutionPolicy Bypass -File "' + $stopScript + '" ' +
    '-ProcessId ' + $proc.Id + ' ' +
    '-DelaySeconds ' + $delaySeconds + ' ' +
    '-ControlLog "' + $controlLog + '" ' +
    '-SummaryScript "' + $summaryScript + '" ' +
    '-RuntimeLog "' + $stderrLog + '" ' +
    '-RuntimeMetaJson "' + $metaPath + '" ' +
    '-PythonPath "' + $python + '" ' +
    '-SummaryDir "' + $summaryDir + '"'
cmd.exe /c ('start "" /b ' + $stopCmd) | Out-Null
