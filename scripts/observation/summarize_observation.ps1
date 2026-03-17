<#
Usage (single window summary):
  powershell -ExecutionPolicy Bypass -File .\scripts\observation\summarize_observation.ps1 -AuditExtractFile .\logs\observation\regime_audit_extract_post_htf_20260311_225357.log

Usage (comparison BEFORE vs AFTER):
  powershell -ExecutionPolicy Bypass -File .\scripts\observation\summarize_observation.ps1 -AuditExtractFile .\logs\observation\after.log -CompareAuditExtractFile .\logs\observation\before.log

Usage (comparison + operator triage):
  powershell -ExecutionPolicy Bypass -File .\scripts\observation\summarize_observation.ps1 -AuditExtractFile .\logs\observation\after.log -CompareAuditExtractFile .\logs\observation\before.log -OutJson .\logs\observation\comparison_latest.json -PrintTriage

Output includes:
  - normalized rates and blocker dominance metrics
  - comparison BEFORE / AFTER / DELTA
  - calibration guardrail recommendation payload
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$AuditExtractFile,

    [Parameter()]
    [string]$CompareAuditExtractFile = "",

    [Parameter()]
    [string]$OutJson = "",

    [Parameter()]
    [string]$PythonPath = ".\\.venv\\Scripts\\python.exe",

    [Parameter()]
    [switch]$PrintTriage,

    [Parameter()]
    [ValidateRange(0, 20)]
    [int]$TriageMaxActions = 4
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-ConcretePathValue {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ParameterName,
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string]$Value
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return
    }
    if ($Value.Contains("<") -or $Value.Contains(">")) {
        throw ("Replace the placeholder in -{0} with a real path before running summarize_observation.ps1." -f $ParameterName)
    }
}

function Invoke-PythonCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Exe,
        [Parameter(Mandatory = $true)]
        [string[]]$Args
    )

    $stdoutFile = Join-Path ([System.IO.Path]::GetTempPath()) ("obs_summary_stdout_" + [System.Guid]::NewGuid().ToString("N") + ".log")
    $stderrFile = Join-Path ([System.IO.Path]::GetTempPath()) ("obs_summary_stderr_" + [System.Guid]::NewGuid().ToString("N") + ".log")
    try {
        $previousErr = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $Exe @Args 1> $stdoutFile 2> $stderrFile
            $exitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $previousErr
        }

        $stdoutText = if (Test-Path $stdoutFile) { Get-Content -Raw -Path $stdoutFile } else { "" }
        $stderrText = if (Test-Path $stderrFile) { Get-Content -Raw -Path $stderrFile } else { "" }
        return @{
            exit_code = [int]$exitCode
            stdout = [string]$stdoutText
            stderr = [string]$stderrText
        }
    }
    finally {
        Remove-Item -Force $stdoutFile, $stderrFile -ErrorAction SilentlyContinue
    }
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "../..")
Push-Location $repoRoot
try {
    Assert-ConcretePathValue -ParameterName "AuditExtractFile" -Value $AuditExtractFile
    Assert-ConcretePathValue -ParameterName "CompareAuditExtractFile" -Value $CompareAuditExtractFile
    Assert-ConcretePathValue -ParameterName "OutJson" -Value $OutJson
    Assert-ConcretePathValue -ParameterName "PythonPath" -Value $PythonPath

    if (-not (Test-Path $PythonPath)) {
        throw "Python executable not found: $PythonPath"
    }
    if (-not (Test-Path $AuditExtractFile)) {
        throw "Audit extract not found: $AuditExtractFile"
    }
    if (-not [string]::IsNullOrWhiteSpace($CompareAuditExtractFile) -and -not (Test-Path $CompareAuditExtractFile)) {
        throw "Compare audit extract not found: $CompareAuditExtractFile"
    }

    $py = @"
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from trading.signals.calibration_control import build_observation_report


def main() -> int:
    after_file = Path(sys.argv[1])
    before_raw = sys.argv[2] if len(sys.argv) > 2 else ""
    before_file = Path(before_raw) if before_raw else None

    report = build_observation_report(after_file=after_file, before_file=before_file)
    report["generated_at"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"@

    $tmpPy = Join-Path ([System.IO.Path]::GetTempPath()) ("obs_calibration_summary_" + [System.Guid]::NewGuid().ToString("N") + ".py")
    Set-Content -Path $tmpPy -Value $py -Encoding UTF8
    try {
        $beforeArg = if ([string]::IsNullOrWhiteSpace($CompareAuditExtractFile)) { "" } else { $CompareAuditExtractFile }
        $args = @($tmpPy, $AuditExtractFile)
        if (-not [string]::IsNullOrWhiteSpace($beforeArg)) {
            $args += $beforeArg
        }

        $result = Invoke-PythonCommand -Exe $PythonPath -Args $args

        $combinedText = ($result.stdout + "`n" + $result.stderr)
        if ($result.exit_code -ne 0 -and $combinedText -match "Failed to import encodings module") {
            $wrapperExe = "C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe"
            $escapedPy = $PythonPath.Replace("'", "''")
            $escapedTmp = $tmpPy.Replace("'", "''")
            $escapedAfter = $AuditExtractFile.Replace("'", "''")
            $cmd = if ([string]::IsNullOrWhiteSpace($beforeArg)) {
                "& '$escapedPy' '$escapedTmp' '$escapedAfter'"
            }
            else {
                $escapedBefore = $beforeArg.Replace("'", "''")
                "& '$escapedPy' '$escapedTmp' '$escapedAfter' '$escapedBefore'"
            }
            $result = Invoke-PythonCommand -Exe $wrapperExe -Args @("-Command", $cmd)
        }

        if ($result.exit_code -ne 0) {
            if (-not [string]::IsNullOrWhiteSpace($result.stdout)) {
                Write-Host "[summarize_observation] python stdout (non-zero exit):"
                Write-Host $result.stdout
            }
            if (-not [string]::IsNullOrWhiteSpace($result.stderr)) {
                Write-Host "[summarize_observation] python stderr (non-zero exit):"
                Write-Host $result.stderr
            }
            throw ("Python observation summary failed (exit_code={0})" -f $result.exit_code)
        }

        $stdoutJson = [string]$result.stdout
        if ([string]::IsNullOrWhiteSpace($stdoutJson)) {
            if (-not [string]::IsNullOrWhiteSpace($result.stderr)) {
                Write-Host "[summarize_observation] python stderr (empty stdout):"
                Write-Host $result.stderr
            }
            throw "Python observation summary produced empty stdout JSON payload"
        }

        try {
            $report = $stdoutJson | ConvertFrom-Json -ErrorAction Stop
        }
        catch {
            Write-Host "[summarize_observation] failed to parse JSON stdout:"
            Write-Host $stdoutJson
            if (-not [string]::IsNullOrWhiteSpace($result.stderr)) {
                Write-Host "[summarize_observation] python stderr while parsing JSON:"
                Write-Host $result.stderr
            }
            throw
        }

        if (-not [string]::IsNullOrWhiteSpace($OutJson)) {
            $outDir = Split-Path -Parent $OutJson
            if (-not [string]::IsNullOrWhiteSpace($outDir)) {
                New-Item -ItemType Directory -Path $outDir -Force | Out-Null
            }
            ($report | ConvertTo-Json -Depth 20) | Set-Content -Path $OutJson
        }

        if ($PrintTriage) {
            $triageScript = Join-Path $PSScriptRoot "triage_calibration_result.ps1"
            if (-not (Test-Path $triageScript)) {
                Write-Host "[summarize_observation] triage script not found: $triageScript"
            }
            else {
                $triageJsonPath = $OutJson
                $deleteTriageJson = $false
                if ([string]::IsNullOrWhiteSpace($triageJsonPath)) {
                    $triageJsonPath = Join-Path ([System.IO.Path]::GetTempPath()) ("obs_triage_" + [System.Guid]::NewGuid().ToString("N") + ".json")
                    ($report | ConvertTo-Json -Depth 20) | Set-Content -Path $triageJsonPath
                    $deleteTriageJson = $true
                }

                try {
                    $triageResult = Invoke-PythonCommand -Exe "C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Args @(
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        $triageScript,
                        "-ComparisonJson",
                        $triageJsonPath,
                        "-MaxActions",
                        [string]$TriageMaxActions
                    )

                    if (-not [string]::IsNullOrWhiteSpace($triageResult.stdout)) {
                        Write-Host ($triageResult.stdout.TrimEnd())
                    }
                    if (-not [string]::IsNullOrWhiteSpace($triageResult.stderr)) {
                        Write-Host "[summarize_observation] triage stderr:"
                        Write-Host $triageResult.stderr
                    }

                    if (@(0, 10, 11, 12, 13, 14) -notcontains [int]$triageResult.exit_code) {
                        throw ("Triage command failed unexpectedly (exit_code={0})" -f $triageResult.exit_code)
                    }
                }
                finally {
                    if ($deleteTriageJson) {
                        Remove-Item -Force $triageJsonPath -ErrorAction SilentlyContinue
                    }
                }
            }
        }

        $report | ConvertTo-Json -Depth 20
    }
    finally {
        Remove-Item -Force $tmpPy -ErrorAction SilentlyContinue
    }
}
finally {
    Pop-Location
}
