<#
.SYNOPSIS
    Run simulated hardware compatibility scenarios against setup\compatibility.json.

.DESCRIPTION
    Exercises the same shared compatibility logic used by the real hardware
    preflight, but with synthetic snapshots for common supported and unsupported
    systems. This helps catch regressions without needing physical hardware for
    every platform.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\setup\test_compatibility_matrix.ps1
#>

[CmdletBinding()]
param(
    [string]$CompatibilityFile = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot 'preflight_common.ps1')

if ([string]::IsNullOrWhiteSpace($CompatibilityFile)) {
    $CompatibilityFile = Join-Path $PSScriptRoot 'compatibility.json'
}

function Write-Step { param([string]$Msg) Write-Host "  -> $Msg" -ForegroundColor Cyan }
function Write-OK   { param([string]$Msg) Write-Host "  [OK] $Msg" -ForegroundColor Green }
function Write-Fail { param([string]$Msg) Write-Host "  [X]  $Msg" -ForegroundColor Red }

function Copy-CompatibilityObject {
    param([pscustomobject]$Compatibility)
    return $Compatibility | ConvertTo-Json -Depth 20 | ConvertFrom-Json
}

function New-TestDriver {
    param(
        [string]$Version,
        [string]$DeviceName = 'Intel(R) AI Boost',
        [string]$Manufacturer = 'Intel Corporation'
    )

    return [pscustomobject]@{
        DeviceName = $DeviceName
        DriverVersion = $Version
        Manufacturer = $Manufacturer
    }
}

function Test-Scenario {
    param([pscustomobject]$Scenario)

    $skipDriverCheck = $false
    if ($Scenario.PSObject.Properties.Name -contains 'SkipDriverCheck') {
        $skipDriverCheck = [bool]$Scenario.SkipDriverCheck
    }

    $report = Test-HardwareCompatibility -Snapshot $Scenario.Snapshot -Compatibility $Scenario.Compatibility -SkipDriverCheck:$skipDriverCheck
    $issues = New-Object System.Collections.Generic.List[string]

    if ($report.Passed -ne [bool]$Scenario.ExpectPassed) {
        $issues.Add("Expected Passed=$($Scenario.ExpectPassed), actual=$($report.Passed).")
    }

    if ($Scenario.PSObject.Properties.Name -contains 'ExpectProfileId' -and -not [string]::IsNullOrWhiteSpace([string]$Scenario.ExpectProfileId)) {
        $actualProfileId = if ($report.Profile) { [string]$report.Profile.id } else { '' }
        if ($actualProfileId -ne [string]$Scenario.ExpectProfileId) {
            $issues.Add("Expected profile '$($Scenario.ExpectProfileId)', actual '$actualProfileId'.")
        }
    }

    if ($Scenario.PSObject.Properties.Name -contains 'ExpectEnvKey' -and -not [string]::IsNullOrWhiteSpace([string]$Scenario.ExpectEnvKey)) {
        if (-not $report.RequiredEnv.ContainsKey([string]$Scenario.ExpectEnvKey)) {
            $issues.Add("Expected required env key '$($Scenario.ExpectEnvKey)' was not set.")
        }
    }

    if ($Scenario.PSObject.Properties.Name -contains 'ExpectWarningRegex' -and -not [string]::IsNullOrWhiteSpace([string]$Scenario.ExpectWarningRegex)) {
        $warningText = $report.Warnings -join ' '
        if ($warningText -notmatch [string]$Scenario.ExpectWarningRegex) {
            $issues.Add("Expected warning matching '$($Scenario.ExpectWarningRegex)' not found.")
        }
    }

    if ($Scenario.PSObject.Properties.Name -contains 'ExpectErrorRegex' -and -not [string]::IsNullOrWhiteSpace([string]$Scenario.ExpectErrorRegex)) {
        $errorText = $report.Errors -join ' '
        if ($errorText -notmatch [string]$Scenario.ExpectErrorRegex) {
            $issues.Add("Expected error matching '$($Scenario.ExpectErrorRegex)' not found.")
        }
    }

    if ($Scenario.PSObject.Properties.Name -contains 'ExpectNoteRegex' -and -not [string]::IsNullOrWhiteSpace([string]$Scenario.ExpectNoteRegex)) {
        $noteText = $report.Notes -join ' '
        if ($noteText -notmatch [string]$Scenario.ExpectNoteRegex) {
            $issues.Add("Expected note matching '$($Scenario.ExpectNoteRegex)' not found.")
        }
    }

    return [pscustomobject]@{
        Name = [string]$Scenario.Name
        Passed = ($issues.Count -eq 0)
        Issues = @($issues)
        Report = $report
    }
}

$baseCompatibility = Get-CompatibilityConfig -CompatibilityFile $CompatibilityFile

$validatedCompatibility = Copy-CompatibilityObject -Compatibility $baseCompatibility
$validatedCompatibility.knownValidatedCombos = @(
    [pscustomobject]@{
        label = 'Synthetic validated combo for test coverage'
        cpuRegex = 'Ultra.*265U'
        biosVendorRegex = '^LENOVO$'
        biosVersionRegex = '^TEST-BIOS-VALID$'
        driverVersion = '32.0.100.4512'
    }
)

$problemCompatibility = Copy-CompatibilityObject -Compatibility $baseCompatibility
$problemCompatibility.knownProblemCombos = @(
    [pscustomobject]@{
        label = 'Synthetic blocked combo'
        cpuRegex = 'Ultra.*265U'
        biosVersionRegex = '^TEST-BIOS-BAD$'
        driverVersion = '32.0.100.4512'
        message = 'Known problematic BIOS/driver combination encountered.'
        severity = 'error'
    }
)

$scenarios = @(
    [pscustomobject]@{
        Name = 'meteor-lake-supported'
        Compatibility = $baseCompatibility
        Snapshot = New-HardwareSnapshot -CpuName 'Intel(R) Core(TM) Ultra 7 155H' -WindowsProduct 'Windows 11 Pro' -WindowsBuild 22631 -BiosVendor 'LENOVO' -BiosVersion 'TEST-MTL-BIOS' -BiosDate '2026-01-01' -Driver (New-TestDriver -Version '32.0.100.3104')
        ExpectPassed = $true
        ExpectProfileId = 'meteor-lake'
        ExpectEnvKey = 'IPEX_LLM_NPU_MTL'
    },
    [pscustomobject]@{
        Name = 'lunar-lake-supported'
        Compatibility = $baseCompatibility
        Snapshot = New-HardwareSnapshot -CpuName 'Intel(R) Core(TM) Ultra 7 258V' -WindowsProduct 'Windows 11 Pro' -WindowsBuild 26100 -BiosVendor 'ASUS' -BiosVersion 'TEST-LL-BIOS' -BiosDate '2026-01-01' -Driver (New-TestDriver -Version '32.0.100.4512')
        ExpectPassed = $true
        ExpectProfileId = 'lunar-lake'
    },
    [pscustomobject]@{
        Name = 'generic-core-ultra-warning'
        Compatibility = $baseCompatibility
        Snapshot = New-HardwareSnapshot -CpuName 'Intel(R) Core(TM) Ultra 7 265U' -WindowsProduct 'Windows 11 Enterprise' -WindowsBuild 26100 -BiosVendor 'LENOVO' -BiosVersion 'TEST-GENERIC-BIOS' -BiosDate '2026-01-01' -Driver (New-TestDriver -Version '32.0.100.4512')
        ExpectPassed = $true
        ExpectProfileId = 'core-ultra-generic'
        ExpectWarningRegex = 'not mapped'
    },
    [pscustomobject]@{
        Name = 'unsupported-cpu-fails'
        Compatibility = $baseCompatibility
        Snapshot = New-HardwareSnapshot -CpuName 'Intel(R) Core(TM) i7-13700H' -WindowsProduct 'Windows 11 Pro' -WindowsBuild 26100 -BiosVendor 'Dell' -BiosVersion 'TEST-CPU-BIOS' -BiosDate '2026-01-01' -Driver (New-TestDriver -Version '32.0.100.4512')
        ExpectPassed = $false
        ExpectErrorRegex = 'Unsupported CPU'
    },
    [pscustomobject]@{
        Name = 'old-driver-fails'
        Compatibility = $baseCompatibility
        Snapshot = New-HardwareSnapshot -CpuName 'Intel(R) Core(TM) Ultra 7 258V' -WindowsProduct 'Windows 11 Pro' -WindowsBuild 26100 -BiosVendor 'ASUS' -BiosVersion 'TEST-OLD-DRV' -BiosDate '2026-01-01' -Driver (New-TestDriver -Version '32.0.100.3000')
        ExpectPassed = $false
        ExpectErrorRegex = 'Version 32.0.100.3104 or newer is required'
    },
    [pscustomobject]@{
        Name = 'blocked-bios-fails'
        Compatibility = $problemCompatibility
        Snapshot = New-HardwareSnapshot -CpuName 'Intel(R) Core(TM) Ultra 7 265U' -WindowsProduct 'Windows 11 Pro' -WindowsBuild 26100 -BiosVendor 'LENOVO' -BiosVersion 'TEST-BIOS-BAD' -BiosDate '2026-01-01' -Driver (New-TestDriver -Version '32.0.100.4512')
        ExpectPassed = $false
        ExpectErrorRegex = 'Known problematic BIOS/driver combination encountered'
    },
    [pscustomobject]@{
        Name = 'validated-combo-note'
        Compatibility = $validatedCompatibility
        Snapshot = New-HardwareSnapshot -CpuName 'Intel(R) Core(TM) Ultra 7 265U' -WindowsProduct 'Windows 11 Enterprise' -WindowsBuild 26100 -BiosVendor 'LENOVO' -BiosVersion 'TEST-BIOS-VALID' -BiosDate '2026-01-01' -Driver (New-TestDriver -Version '32.0.100.4512')
        ExpectPassed = $true
        ExpectProfileId = 'core-ultra-generic'
        ExpectNoteRegex = 'Synthetic validated combo'
    }
)

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  Compatibility matrix test runner" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

$failedCount = 0
foreach ($scenario in $scenarios) {
    Write-Step "Running $($scenario.Name)..."
    $result = Test-Scenario -Scenario $scenario
    if ($result.Passed) {
        Write-OK $result.Name
    } else {
        $failedCount += 1
        Write-Fail $result.Name
        foreach ($issue in $result.Issues) {
            Write-Fail "  $issue"
        }
    }
}

Write-Host ""
if ($failedCount -eq 0) {
    Write-OK "All $($scenarios.Count) compatibility scenarios passed."
    exit 0
}

Write-Fail "$failedCount compatibility scenario(s) failed."
exit 1