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

function Get-IntegerValue {
    param(
        [Parameter()]
        $Value
    )

    try {
        return [int]$Value
    }
    catch {
        return 0
    }
}

function Get-QualityOverviewLine {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter()]
        $TotalCount,
        [Parameter()]
        [AllowEmptyString()]
        [string]$MainVerdict,
        [Parameter()]
        [AllowEmptyString()]
        [string]$EarlyVerdict
    )

    $total = Get-IntegerValue -Value $TotalCount
    $mainText = Get-TextValue -Value $MainVerdict -Default "n/a"
    $earlyText = Get-TextValue -Value $EarlyVerdict -Default "n/a"
    if ($total -le 0 -and $mainText -eq "n/a" -and $earlyText -eq "n/a") {
        return ""
    }
    return ("{0}: total={1} main={2} early={3}" -f $Label, $total, $mainText, $earlyText)
}

function Get-ObjectPropertyValue {
    param(
        [Parameter()]
        $Object,
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    if ($null -eq $Object) {
        return $null
    }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return $null
    }
    return $property.Value
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
$qualityOverview = Get-ObjectPropertyValue -Object $payload -Name "quality_overview"
$qualityGuidance = Get-ObjectPropertyValue -Object $payload -Name "quality_guidance"

$lines = @(
    "VERDICT: $verdict",
    "STOP_REASON: $stopReason",
    "TOP_COMBINATION: $topCombination"
)

$signalOverviewLine = Get-QualityOverviewLine `
    -Label "SIGNAL_OVERVIEW" `
    -TotalCount (Get-ObjectPropertyValue -Object $qualityOverview -Name "signal_total_count") `
    -MainVerdict ([string](Get-ObjectPropertyValue -Object $qualityOverview -Name "signal_main_top_verdict")) `
    -EarlyVerdict ([string](Get-ObjectPropertyValue -Object $qualityOverview -Name "signal_early_top_verdict"))
if (-not [string]::IsNullOrWhiteSpace($signalOverviewLine)) {
    $lines += $signalOverviewLine
}

$exitOverviewLine = Get-QualityOverviewLine `
    -Label "EXIT_OVERVIEW" `
    -TotalCount (Get-ObjectPropertyValue -Object $qualityOverview -Name "exit_total_count") `
    -MainVerdict ([string](Get-ObjectPropertyValue -Object $qualityOverview -Name "exit_main_top_verdict")) `
    -EarlyVerdict ([string](Get-ObjectPropertyValue -Object $qualityOverview -Name "exit_early_top_verdict"))
if (-not [string]::IsNullOrWhiteSpace($exitOverviewLine)) {
    $lines += $exitOverviewLine
}

$entryFocus = Get-TextValue -Value (Get-ObjectPropertyValue -Object $qualityGuidance -Name "entry_focus") -Default ""
$entryPriority = Get-TextValue -Value (Get-ObjectPropertyValue -Object $qualityGuidance -Name "entry_priority") -Default "n/a"
if (-not [string]::IsNullOrWhiteSpace($entryFocus)) {
    $lines += ("QUALITY_ENTRY: {0} priority={1}" -f $entryFocus, $entryPriority)
}

$exitFocus = Get-TextValue -Value (Get-ObjectPropertyValue -Object $qualityGuidance -Name "exit_focus") -Default ""
$exitPriority = Get-TextValue -Value (Get-ObjectPropertyValue -Object $qualityGuidance -Name "exit_priority") -Default "n/a"
if (-not [string]::IsNullOrWhiteSpace($exitFocus)) {
    $lines += ("QUALITY_EXIT: {0} priority={1}" -f $exitFocus, $exitPriority)
}

$guidanceActions = @()
$rawGuidanceActions = Get-ObjectPropertyValue -Object $qualityGuidance -Name "runbook_actions"
if ($null -ne $rawGuidanceActions) {
    foreach ($item in @($rawGuidanceActions)) {
        $text = Get-TextValue -Value $item -Default ""
        if (-not [string]::IsNullOrWhiteSpace($text)) {
            $guidanceActions += $text
        }
    }
}
for ($index = 0; $index -lt [Math]::Min(2, $guidanceActions.Count); $index++) {
    $lines += "QUALITY_ACTION $($index + 1): $($guidanceActions[$index])"
}

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
