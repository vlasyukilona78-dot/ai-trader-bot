<#
Usage:
  powershell -ExecutionPolicy Bypass -File .\scripts\observation\triage_calibration_result.ps1 -ComparisonJson .\logs\observation\comparison.json

Output:
  - VERDICT
  - STOP_REASON
  - TOP_COMBINATION
  - ACTION 1..N

Exit codes:
  - 0 safe single-target continuation
  - 10 overlap stop
  - 11 semantics disagreement
  - 12 market context shift
  - 13 non-comparable window
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [Alias("JsonFile")]
    [string]$ComparisonJson,

    [Parameter()]
    [int]$MaxActions = 4,

    [Parameter()]
    [string]$PythonPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ExitCodes = @{
    "co_dominant_regime_blockers" = 10
    "blocker_semantics_disagreement" = 11
    "market_context_shift_detected" = 12
    "window_size_not_comparable" = 13
}

function Get-TextValue {
    param(
        [Parameter()]
        $Value,
        [Parameter()]
        [AllowEmptyString()]
        [string]$Default
    )

    if ($null -eq $Value) {
        return $Default
    }
    $text = [string]$Value
    if ([string]::IsNullOrWhiteSpace($text)) {
        return $Default
    }
    return $text.Trim()
}

function Get-RecommendationExitCode {
    param(
        [Parameter(Mandatory = $true)]
        $Recommendation
    )

    if ([bool]$Recommendation.SAFE_TO_CONTINUE) {
        return 0
    }

    $actionVerdict = Get-TextValue -Value $Recommendation.ACTION_VERDICT -Default ""
    if ($actionVerdict -eq "single_blocker_ready") {
        return 0
    }

    $stopReason = Get-TextValue -Value $Recommendation.STOP_REASON -Default ""
    if ($ExitCodes.ContainsKey($stopReason)) {
        return [int]$ExitCodes[$stopReason]
    }
    return 14
}

if (-not (Test-Path $ComparisonJson)) {
    throw "Calibration result JSON not found: $ComparisonJson"
}

$payload = Get-Content -Raw -Path $ComparisonJson | ConvertFrom-Json
$recommendation = $payload.calibration_recommendation
if ($null -eq $recommendation) {
    throw "calibration_recommendation is missing from the JSON payload."
}

$afterSummary = $null
if ($null -ne $payload.after) {
    $afterSummary = $payload.after.summary
}

$verdict = Get-TextValue -Value $recommendation.ACTION_VERDICT -Default "unknown"
$stopReason = Get-TextValue -Value $recommendation.STOP_REASON -Default "none"
$topCombination = if ($null -eq $afterSummary) {
    "none"
}
else {
    Get-TextValue -Value $afterSummary.top_regime_filter_blocker_combination -Default "none"
}

$lines = @(
    "VERDICT: $verdict",
    "STOP_REASON: $stopReason",
    "TOP_COMBINATION: $topCombination"
)

$actions = @()
if ($null -ne $recommendation.RUNBOOK_ACTIONS -and $recommendation.RUNBOOK_ACTIONS -isnot [string]) {
    foreach ($item in $recommendation.RUNBOOK_ACTIONS) {
        $text = Get-TextValue -Value $item -Default ""
        if (-not [string]::IsNullOrWhiteSpace($text)) {
            $actions += $text
        }
    }
}

$actionLimit = [Math]::Min([Math]::Max($MaxActions, 0), $actions.Count)
for ($index = 0; $index -lt $actionLimit; $index++) {
    $lines += "ACTION $($index + 1): $($actions[$index])"
}

$lines
exit (Get-RecommendationExitCode -Recommendation $recommendation)
