param(
    [ValidateRange(1, 24)]
    [int]$Hours = 12
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type @"
using System;
using System.Runtime.InteropServices;

public static class KoteikaNightPower
{
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@

$continuous = [Convert]::ToUInt32("80000000", 16)
$systemRequired = [uint32]0x00000001
$deadline = (Get-Date).AddHours($Hours)

try {
    while ((Get-Date) -lt $deadline) {
        [void][KoteikaNightPower]::SetThreadExecutionState($continuous -bor $systemRequired)
        Start-Sleep -Seconds 30
    }
}
finally {
    [void][KoteikaNightPower]::SetThreadExecutionState($continuous)
}
