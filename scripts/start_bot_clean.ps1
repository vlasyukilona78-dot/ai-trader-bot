param(
    [ValidateSet("both", "main", "early")]
    [string]$SignalProfile = "main",
    [string]$PythonPath = "",
    [switch]$Once
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not $PythonPath) {
    $PythonPath = Join-Path $root ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Python runtime not found: $PythonPath. Run scripts\validate_install.ps1 after creating .venv."
}

$runtimeDir = Join-Path $root "logs\runtime"
New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdoutLog = Join-Path $runtimeDir ("bot_{0}_{1}.stdout.log" -f $SignalProfile, $stamp)
$stderrLog = Join-Path $runtimeDir ("bot_{0}_{1}.stderr.log" -f $SignalProfile, $stamp)

Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$env:VIRTUAL_ENV = Join-Path $root ".venv"
$env:MPLCONFIGDIR = Join-Path $root "data\runtime\matplotlib"
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null

# Some Windows launcher environments preserve both `Path` and `PATH`.
# Start-Process copies variables into a case-insensitive dictionary and aborts
# on that duplicate before Python starts. Normalize only the current launcher
# process while preserving every distinct search-path entry.
$processEnvironment = [Environment]::GetEnvironmentVariables([EnvironmentVariableTarget]::Process)
$pathEntries = @(
    $processEnvironment.GetEnumerator() |
        Where-Object { [string]$_.Key -ieq "PATH" }
)
if ($pathEntries.Count -gt 1) {
    $normalizedPath = @(
        $pathEntries |
            ForEach-Object { ([string]$_.Value) -split ";" } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
            Select-Object -Unique
    ) -join ";"
    foreach ($entry in $pathEntries) {
        [Environment]::SetEnvironmentVariable(
            [string]$entry.Key,
            $null,
            [EnvironmentVariableTarget]::Process
        )
    }
    [Environment]::SetEnvironmentVariable(
        "Path",
        $normalizedPath,
        [EnvironmentVariableTarget]::Process
    )
}

$arguments = @("-u", "-m", "app.main", "--signal-profile", $SignalProfile)
if (-not $Once) {
    $arguments += "--loop"
}

if ($Once) {
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $PythonPath @arguments 1>> $stdoutLog 2>> $stderrLog
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($exitCode -ne 0) {
        $stderrTail = ""
        if (Test-Path -LiteralPath $stderrLog) {
            $stderrTail = (Get-Content -LiteralPath $stderrLog -Tail 20) -join [Environment]::NewLine
        }
        throw "Bot cycle failed with code $exitCode.`n$stderrTail"
    }
    [pscustomobject]@{
        pid = $null
        signal_profile = $SignalProfile
        loop = $false
        exit_code = $exitCode
        python = $PythonPath
        stdout = $stdoutLog
        stderr = $stderrLog
    } | ConvertTo-Json -Compress
    return
}

$proc = Start-Process `
    -FilePath $PythonPath `
    -ArgumentList $arguments `
    -WorkingDirectory $root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

Start-Sleep -Seconds 3
if ($proc.HasExited) {
    $stderrTail = ""
    if (Test-Path -LiteralPath $stderrLog) {
        $stderrTail = (Get-Content -LiteralPath $stderrLog -Tail 20) -join [Environment]::NewLine
    }
    throw "Bot exited during startup with code $($proc.ExitCode).`n$stderrTail"
}

[pscustomobject]@{
    pid = $proc.Id
    signal_profile = $SignalProfile
    loop = $true
    exit_code = $null
    python = $PythonPath
    stdout = $stdoutLog
    stderr = $stderrLog
} | ConvertTo-Json -Compress
