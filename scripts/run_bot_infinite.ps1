param(
    [ValidateSet("both", "main", "early")]
    [string]$SignalProfile = "both"
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Get-PyVenvExecutable {
    param([string]$CfgPath)

    if (-not (Test-Path $CfgPath)) {
        return $null
    }

    foreach ($line in Get-Content $CfgPath -ErrorAction SilentlyContinue) {
        if ($line -match '^(?:executable|base-executable)\s*=\s*(.+)$') {
            $candidate = $Matches[1].Trim()
            if ($candidate -and (Test-Path $candidate)) {
                return $candidate
            }
        }
    }

    return $null
}

function Get-VenvSitePackages {
    param([string]$VenvRoot)

    if (-not $VenvRoot) {
        return $null
    }

    $candidate = Join-Path $VenvRoot "Lib\site-packages"
    if (Test-Path $candidate) {
        return $candidate
    }

    return $null
}

function Get-BotProcesses {
    param(
        [string]$MainPy,
        [string]$SignalProfile
    )

    Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
        Where-Object {
            if (-not $_.CommandLine) {
                return $false
            }
            $matchesMain = $_.CommandLine -like ("*{0}*" -f $MainPy)
            $matchesLoop = $_.CommandLine -like "*--loop*"
            $matchesProfile = (
                $SignalProfile -eq "both" -or
                $_.CommandLine -like ("*--signal-profile {0}*" -f $SignalProfile) -or
                $_.CommandLine -notlike "*--signal-profile*"
            )
            return $matchesMain -and $matchesLoop -and $matchesProfile
        }
}

$runtimeDir = Join-Path $root "logs\runtime"
New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$profileSuffix = if ($SignalProfile -eq "both") { "" } else { "_$SignalProfile" }
$stdoutLog = Join-Path $runtimeDir ("bot_loop_infinite{0}_{1}.stdout.log" -f $profileSuffix, $stamp)
$stderrLog = Join-Path $runtimeDir ("bot_loop_infinite{0}_{1}.stderr.log" -f $profileSuffix, $stamp)
$controlLog = Join-Path $runtimeDir ("bot_loop_infinite{0}_{1}.control.log" -f $profileSuffix, $stamp)
$metaPath = Join-Path $runtimeDir ("bot_loop_infinite{0}_{1}.json" -f $profileSuffix, $stamp)
$mainPy = Join-Path $root "main.py"

$runtimeVenvRoot = Join-Path $root ".runtime_env"
$projectVenvRoot = Join-Path $root ".venv"
$runtimeBasePython = Get-PyVenvExecutable (Join-Path $runtimeVenvRoot "pyvenv.cfg")
$projectBasePython = Get-PyVenvExecutable (Join-Path $projectVenvRoot "pyvenv.cfg")
$projectVenvPython = Join-Path $projectVenvRoot "Scripts\python.exe"
$runtimeVenvPython = Join-Path $runtimeVenvRoot "Scripts\python.exe"

Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue

$pythonCandidates = @(
    $projectVenvPython,
    $runtimeVenvPython,
    $projectBasePython,
    $runtimeBasePython
) | Where-Object { $_ } | Select-Object -Unique

$python = $null
foreach ($candidate in @($projectVenvPython, $runtimeVenvPython, $projectBasePython, $runtimeBasePython) | Where-Object { $_ }) {
    if (Test-Path $candidate) {
        $python = $candidate
        break
    }
}

if (-not $python) {
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
}
if (-not $python) {
    throw "No Python runtime found in .runtime_env or .venv."
}

$venvRoot = $null
if ($python -eq $runtimeBasePython -or $python -eq $runtimeVenvPython -or $python -like "$runtimeVenvRoot*") {
    $venvRoot = $runtimeVenvRoot
} elseif ($python -eq $projectBasePython -or $python -eq $projectVenvPython -or $python -like "$projectVenvRoot*") {
    $venvRoot = $projectVenvRoot
}
if (-not $venvRoot -and (Test-Path $projectVenvRoot)) {
    $venvRoot = $projectVenvRoot
}
$sitePackages = Get-VenvSitePackages $venvRoot

$existing = @(Get-BotProcesses -MainPy $mainPy -SignalProfile $SignalProfile)
if ($existing.Count -gt 0) {
    foreach ($proc in $existing) {
        try {
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
            Add-Content -Path $controlLog -Value ("stopped_existing pid={0} profile={1}" -f $proc.ProcessId, $SignalProfile)
        } catch {
            Add-Content -Path $controlLog -Value ("failed_to_stop_existing pid={0} profile={1} err={2}" -f $proc.ProcessId, $SignalProfile, $_.Exception.Message)
        }
    }
    Start-Sleep -Seconds 1
}

$launchTs = Get-Date
$cmdLine = 'set "PYTHONHOME=" && set "PYTHONPATH=" && set "VIRTUAL_ENV=" && start "" /b "{0}" -u "{1}" --loop --signal-profile {2} 1>>"{3}" 2>>"{4}"' -f $python, $mainPy, $SignalProfile, $stdoutLog, $stderrLog
& $env:ComSpec /d /c $cmdLine | Out-Null
Start-Sleep -Seconds 1

$proc = Get-Process python -ErrorAction SilentlyContinue |
    Where-Object {
        $_.StartTime -ge $launchTs.AddSeconds(-2) -and ($_.Path -eq $python -or $_.Path -like "*koteika_Ultra*")
    } |
    Sort-Object StartTime -Descending |
    Select-Object -First 1

if (-not $proc) {
    throw "Failed to start bot process."
}

$meta = [ordered]@{
    pid = $proc.Id
    started_at = (Get-Date).ToString("o")
    mode = "infinite"
    signal_profile = $SignalProfile
    stdout_log = $stdoutLog
    stderr_log = $stderrLog
    control_log = $controlLog
}
$meta | ConvertTo-Json | Set-Content -Path $metaPath -Encoding UTF8

Add-Content -Path $controlLog -Value ("started pid={0} profile={1} python={2}" -f $proc.Id, $SignalProfile, $python)
Write-Output ("started pid={0} profile={1}" -f $proc.Id, $SignalProfile)
Write-Output ("stderr={0}" -f $stderrLog)
