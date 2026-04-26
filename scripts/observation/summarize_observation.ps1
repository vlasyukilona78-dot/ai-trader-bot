<#
Usage (single window summary):
  powershell -ExecutionPolicy Bypass -File .\scripts\observation\summarize_observation.ps1 -AuditExtractFile .\logs\observation\regime_audit_extract_post_htf_20260311_225357.log

Usage (comparison BEFORE vs AFTER):
  powershell -ExecutionPolicy Bypass -File .\scripts\observation\summarize_observation.ps1 -AuditExtractFile .\logs\observation\after.log -CompareAuditExtractFile .\logs\observation\before.log

Usage (comparison + operator triage):
  powershell -ExecutionPolicy Bypass -File .\scripts\observation\summarize_observation.ps1 -AuditExtractFile .\logs\observation\after.log -CompareAuditExtractFile .\logs\observation\before.log -OutJson .\logs\observation\comparison_latest.json -PrintTriage

Usage (comparison + quality drill-down attached):
  powershell -ExecutionPolicy Bypass -File .\scripts\observation\summarize_observation.ps1 -AuditExtractFile .\logs\observation\after.log -CompareAuditExtractFile .\logs\observation\before.log -SignalQualityJson .\logs\observation\signal_quality_latest.json -ExitQualityJson .\logs\observation\exit_quality_latest.json -OutJson .\logs\observation\comparison_latest.json -PrintTriage

Output includes:
  - normalized rates and blocker dominance metrics
  - comparison BEFORE / AFTER / DELTA
  - calibration guardrail recommendation payload
  - optional quality guidance / quality overview when signal/exit quality JSON is supplied
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
    [int]$TriageMaxActions = 4,

    [Parameter()]
    [string]$SignalQualityJson = "",

    [Parameter()]
    [string]$ExitQualityJson = "",

    [Parameter()]
    [string]$ObservationTimeframe = "1"
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

function Build-QualityOverviewFromGuidance {
    param(
        [Parameter(Mandatory = $true)]
        [AllowNull()]
        [object]$GuidancePayload
    )

    if ($null -eq $GuidancePayload) {
        return $null
    }

    $metrics = $GuidancePayload.PSObject.Properties["metrics"]
    if ($null -eq $metrics) {
        return $null
    }

    $metricPayload = $metrics.Value
    if ($null -eq $metricPayload) {
        return $null
    }

    $signalTotal = 0
    $signalMainVerdict = ""
    $signalEarlyVerdict = ""
    $exitTotal = 0
    $exitMainVerdict = ""
    $exitEarlyVerdict = ""

    $prop = $metricPayload.PSObject.Properties["signal_total_count"]
    if ($null -ne $prop) { $signalTotal = [int]$prop.Value }
    $prop = $metricPayload.PSObject.Properties["signal_main_top_verdict"]
    if ($null -ne $prop) { $signalMainVerdict = [string]$prop.Value }
    $prop = $metricPayload.PSObject.Properties["signal_early_top_verdict"]
    if ($null -ne $prop) { $signalEarlyVerdict = [string]$prop.Value }
    $prop = $metricPayload.PSObject.Properties["exit_total_count"]
    if ($null -ne $prop) { $exitTotal = [int]$prop.Value }
    $prop = $metricPayload.PSObject.Properties["exit_main_top_verdict"]
    if ($null -ne $prop) { $exitMainVerdict = [string]$prop.Value }
    $prop = $metricPayload.PSObject.Properties["exit_early_top_verdict"]
    if ($null -ne $prop) { $exitEarlyVerdict = [string]$prop.Value }

    return [pscustomobject]@{
        signal_total_count         = $signalTotal
        signal_main_top_verdict    = $signalMainVerdict
        signal_early_top_verdict   = $signalEarlyVerdict
        exit_total_count           = $exitTotal
        exit_main_top_verdict      = $exitMainVerdict
        exit_early_top_verdict     = $exitEarlyVerdict
    }
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "../..")
Push-Location $repoRoot
try {
    Assert-ConcretePathValue -ParameterName "AuditExtractFile" -Value $AuditExtractFile
    Assert-ConcretePathValue -ParameterName "CompareAuditExtractFile" -Value $CompareAuditExtractFile
    Assert-ConcretePathValue -ParameterName "OutJson" -Value $OutJson
    Assert-ConcretePathValue -ParameterName "PythonPath" -Value $PythonPath
    Assert-ConcretePathValue -ParameterName "SignalQualityJson" -Value $SignalQualityJson
    Assert-ConcretePathValue -ParameterName "ExitQualityJson" -Value $ExitQualityJson
    Assert-ConcretePathValue -ParameterName "ObservationTimeframe" -Value $ObservationTimeframe

    if (-not (Test-Path $PythonPath)) {
        throw "Python executable not found: $PythonPath"
    }
    if (-not (Test-Path $AuditExtractFile)) {
        throw "Audit extract not found: $AuditExtractFile"
    }
    if (-not [string]::IsNullOrWhiteSpace($CompareAuditExtractFile) -and -not (Test-Path $CompareAuditExtractFile)) {
        throw "Compare audit extract not found: $CompareAuditExtractFile"
    }
    if (-not [string]::IsNullOrWhiteSpace($SignalQualityJson) -and -not (Test-Path $SignalQualityJson)) {
        throw "Signal quality JSON not found: $SignalQualityJson"
    }
    if (-not [string]::IsNullOrWhiteSpace($ExitQualityJson) -and -not (Test-Path $ExitQualityJson)) {
        throw "Exit quality JSON not found: $ExitQualityJson"
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

        if ((-not [string]::IsNullOrWhiteSpace($SignalQualityJson)) -or (-not [string]::IsNullOrWhiteSpace($ExitQualityJson))) {
            $guidanceScript = Join-Path $PSScriptRoot "build_quality_guidance.py"
            if (-not (Test-Path $guidanceScript)) {
                throw "Quality guidance script not found: $guidanceScript"
            }

            $guidanceArgs = @($guidanceScript, "--timeframe", [string]$ObservationTimeframe)
            if (-not [string]::IsNullOrWhiteSpace($SignalQualityJson)) {
                $guidanceArgs += @("--signal-json", $SignalQualityJson)
            }
            if (-not [string]::IsNullOrWhiteSpace($ExitQualityJson)) {
                $guidanceArgs += @("--exit-json", $ExitQualityJson)
            }

            $guidanceResult = Invoke-PythonCommand -Exe $PythonPath -Args $guidanceArgs
            if ($guidanceResult.exit_code -ne 0) {
                if (-not [string]::IsNullOrWhiteSpace($guidanceResult.stdout)) {
                    Write-Host "[summarize_observation] quality guidance stdout (non-zero exit):"
                    Write-Host $guidanceResult.stdout
                }
                if (-not [string]::IsNullOrWhiteSpace($guidanceResult.stderr)) {
                    Write-Host "[summarize_observation] quality guidance stderr (non-zero exit):"
                    Write-Host $guidanceResult.stderr
                }
                throw ("Python quality guidance failed (exit_code={0})" -f $guidanceResult.exit_code)
            }

            if (-not [string]::IsNullOrWhiteSpace($guidanceResult.stdout)) {
                try {
                $guidancePayload = $guidanceResult.stdout | ConvertFrom-Json -ErrorAction Stop
                }
                catch {
                    Write-Host "[summarize_observation] failed to parse quality guidance JSON stdout:"
                    Write-Host $guidanceResult.stdout
                    if (-not [string]::IsNullOrWhiteSpace($guidanceResult.stderr)) {
                        Write-Host "[summarize_observation] quality guidance stderr while parsing JSON:"
                        Write-Host $guidanceResult.stderr
                    }
                    throw
                }

                $report | Add-Member -Force -NotePropertyName "quality_guidance" -NotePropertyValue $guidancePayload
                $qualityOverview = Build-QualityOverviewFromGuidance -GuidancePayload $guidancePayload
                if ($null -ne $qualityOverview) {
                    $report | Add-Member -Force -NotePropertyName "quality_overview" -NotePropertyValue $qualityOverview
                }

                $recommendationProp = $report.PSObject.Properties["calibration_recommendation"]
                if ($null -ne $recommendationProp -and $null -ne $recommendationProp.Value) {
                    $recommendationPayload = $recommendationProp.Value
                    $entryFocusProp = $guidancePayload.PSObject.Properties["entry_focus"]
                    $entryPriorityProp = $guidancePayload.PSObject.Properties["entry_priority"]
                    $exitFocusProp = $guidancePayload.PSObject.Properties["exit_focus"]
                    $exitPriorityProp = $guidancePayload.PSObject.Properties["exit_priority"]

                    if ($null -ne $entryFocusProp) {
                        $recommendationPayload | Add-Member -Force -NotePropertyName "QUALITY_ENTRY_FOCUS" -NotePropertyValue ([string]$entryFocusProp.Value)
                    }
                    if ($null -ne $entryPriorityProp) {
                        $recommendationPayload | Add-Member -Force -NotePropertyName "QUALITY_ENTRY_PRIORITY" -NotePropertyValue ([string]$entryPriorityProp.Value)
                    }
                    if ($null -ne $exitFocusProp) {
                        $recommendationPayload | Add-Member -Force -NotePropertyName "QUALITY_EXIT_FOCUS" -NotePropertyValue ([string]$exitFocusProp.Value)
                    }
                    if ($null -ne $exitPriorityProp) {
                        $recommendationPayload | Add-Member -Force -NotePropertyName "QUALITY_EXIT_PRIORITY" -NotePropertyValue ([string]$exitPriorityProp.Value)
                    }

                    $existingActions = New-Object System.Collections.Generic.List[string]
                    $existingActionsProp = $recommendationPayload.PSObject.Properties["RUNBOOK_ACTIONS"]
                    if ($null -ne $existingActionsProp -and $null -ne $existingActionsProp.Value) {
                        foreach ($item in @($existingActionsProp.Value)) {
                            $text = [string]$item
                            if (-not [string]::IsNullOrWhiteSpace($text) -and -not $existingActions.Contains($text)) {
                                $existingActions.Add($text)
                            }
                        }
                    }

                    $guidanceActionsProp = $guidancePayload.PSObject.Properties["runbook_actions"]
                    if ($null -ne $guidanceActionsProp -and $null -ne $guidanceActionsProp.Value) {
                        foreach ($item in @($guidanceActionsProp.Value)) {
                            $text = [string]$item
                            if (-not [string]::IsNullOrWhiteSpace($text) -and -not $existingActions.Contains($text)) {
                                $existingActions.Add($text)
                            }
                        }
                    }

                    $recommendationPayload | Add-Member -Force -NotePropertyName "RUNBOOK_ACTIONS" -NotePropertyValue @($existingActions.ToArray())
                }
            }
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
