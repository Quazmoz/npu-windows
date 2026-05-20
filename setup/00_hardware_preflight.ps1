<#
.SYNOPSIS
    Preflight hardware and driver compatibility checks for Intel NPU setup.

.DESCRIPTION
    Verifies the current machine is a plausible target for this repo before setup
    or server startup proceeds. Checks:
      - Windows 11 build requirement
      - Intel Core Ultra CPU family / platform profile
      - Intel NPU device and driver version
      - Optional BIOS blocklist from compatibility.json

    Use -AllowUnsupportedHardware to continue with warnings instead of stopping.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\setup\00_hardware_preflight.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\setup\00_hardware_preflight.ps1 -AllowUnsupportedHardware
#>

[CmdletBinding()]
param(
    [switch]$AllowUnsupportedHardware,
    [switch]$SkipDriverCheck,
    [string]$CompatibilityFile = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($CompatibilityFile)) {
    $CompatibilityFile = Join-Path $PSScriptRoot 'compatibility.json'
}

function Write-Step { param([string]$Msg) Write-Host "  -> $Msg" -ForegroundColor Cyan }
function Write-OK   { param([string]$Msg) Write-Host "  [OK] $Msg" -ForegroundColor Green }
function Write-Warn { param([string]$Msg) Write-Host "  [!]  $Msg" -ForegroundColor Yellow }
function Write-Fail { param([string]$Msg) Write-Host "  [X]  $Msg" -ForegroundColor Red }

. (Join-Path $PSScriptRoot 'preflight_common.ps1')

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  Preflight: Hardware compatibility" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

$compat = Get-CompatibilityConfig -CompatibilityFile $CompatibilityFile

Write-Step "Reading local hardware and OS information..."
$snapshot = Get-LocalHardwareSnapshot -Compatibility $compat -SkipDriverCheck:$SkipDriverCheck
$report = Test-HardwareCompatibility -Snapshot $snapshot -Compatibility $compat -SkipDriverCheck:$SkipDriverCheck
Apply-CompatibilityEnvironment -Report $report

Write-OK "Detected CPU: $($report.CpuName)"
Write-OK "Windows: $($report.WindowsProduct) (build $($report.WindowsBuild))"
if ($report.Profile) {
    Write-OK "Processor profile: $($report.Profile.label)"
}

if ($report.Driver) {
    $driverName = [string]$report.Driver.DeviceName
    if ([string]::IsNullOrWhiteSpace($driverName)) {
        $driverName = 'Intel NPU device'
    }

    if ([string]::IsNullOrWhiteSpace([string]$report.Driver.DriverVersion)) {
        Write-Warn "$driverName detected, but the driver version could not be read."
    } else {
        Write-OK "NPU device: $driverName"
        Write-OK "NPU driver: $($report.Driver.DriverVersion)"
    }
} elseif ($SkipDriverCheck) {
    Write-Warn 'Driver version check skipped.'
}

Write-OK "BIOS/firmware: $($report.BiosVendor) $($report.BiosVersion) ($($report.BiosDate))"

if ($report.Notes.Count -gt 0) {
    foreach ($noteText in $report.Notes) {
        Write-OK $noteText
    }
}

if ($report.Errors.Count -gt 0) {
    Write-Host ""
    foreach ($errorText in $report.Errors) {
        Write-Fail $errorText
    }

    if ($AllowUnsupportedHardware) {
        Write-Host ""
        Write-Warn 'Continuing because -AllowUnsupportedHardware was set.'
        exit 0
    }

    Write-Host ""
    Write-Warn 'Stop here and fix the compatibility problems above, or re-run with -AllowUnsupportedHardware if you intentionally want to continue.'
    exit 1
}

if ($report.Warnings.Count -gt 0) {
    Write-Host ""
    foreach ($warningText in $report.Warnings) {
        Write-Warn $warningText
    }
}

Write-Host ""
Write-OK 'Hardware preflight passed.'