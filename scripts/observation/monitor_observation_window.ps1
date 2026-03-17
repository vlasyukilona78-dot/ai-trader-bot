param(
    [Parameter(Mandatory = $true)]
    [string]$CombinedLog,

    [string]$DriverStdErrLog = "",

    [string]$OutLog = "",

    [string]$ReportPath = "",

    [Parameter(Mandatory = $true)]
    [datetimeoffset]$StopAt,

    [int]$PollSeconds = 60,

    [int]$StaleWarnSeconds = 180,

    [int]$GraceMinutes = 20
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-MonitorLine {
    param(
        [string]$Path,
        [string]$Message
    )

    Add-Content -Path $Path -Value $Message -Encoding UTF8
}

if (-not (Test-Path $CombinedLog)) {
    throw "Combined log not found: $CombinedLog"
}

if ([string]::IsNullOrWhiteSpace($OutLog)) {
    $combinedDir = Split-Path -Parent $CombinedLog
    $combinedBase = [System.IO.Path]::GetFileNameWithoutExtension($CombinedLog)
    $OutLog = Join-Path $combinedDir ($combinedBase + "_monitor.log")
}

$outDir = Split-Path -Parent $OutLog
if (-not [string]::IsNullOrWhiteSpace($outDir) -and -not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

Write-MonitorLine -Path $OutLog -Message ("[{0}] monitor started combined_log={1} stop_at={2}" -f [datetimeoffset]::Now.ToString("o"), $CombinedLog, $StopAt.ToString("o"))

while ($true) {
    $now = [datetimeoffset]::Now
    $combinedItem = Get-Item $CombinedLog -ErrorAction SilentlyContinue
    $stderrItem = if ([string]::IsNullOrWhiteSpace($DriverStdErrLog)) {
        $null
    }
    else {
        Get-Item $DriverStdErrLog -ErrorAction SilentlyContinue
    }
    $reportExists = if ([string]::IsNullOrWhiteSpace($ReportPath)) {
        $false
    }
    else {
        Test-Path $ReportPath
    }
    $combinedLength = if ($combinedItem) { $combinedItem.Length } else { 0 }
    $combinedWrite = if ($combinedItem) { $combinedItem.LastWriteTime.ToString("o") } else { "missing" }
    $stderrLength = if ($stderrItem) { $stderrItem.Length } else { 0 }
    $staleSeconds = if ($combinedItem) { [int]((Get-Date) - $combinedItem.LastWriteTime).TotalSeconds } else { -1 }
    $level = "INFO"

    if ($stderrLength -gt 0) {
        $level = "WARN"
    }
    elseif ($now -lt $StopAt -and $staleSeconds -gt $StaleWarnSeconds) {
        $level = "WARN"
    }

    Write-MonitorLine -Path $OutLog -Message ("[{0}] {1} combined_len={2} combined_last_write={3} stale_sec={4} stderr_len={5} report_exists={6}" -f $now.ToString("o"), $level, $combinedLength, $combinedWrite, $staleSeconds, $stderrLength, $reportExists)

    if ($reportExists) {
        Write-MonitorLine -Path $OutLog -Message ("[{0}] INFO report detected; monitor stopping" -f [datetimeoffset]::Now.ToString("o"))
        break
    }

    if ($now -gt $StopAt.AddMinutes($GraceMinutes)) {
        Write-MonitorLine -Path $OutLog -Message ("[{0}] INFO stop deadline exceeded; monitor stopping" -f [datetimeoffset]::Now.ToString("o"))
        break
    }

    Start-Sleep -Seconds $PollSeconds
}
