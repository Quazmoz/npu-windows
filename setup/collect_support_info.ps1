<#
.SYNOPSIS
    Collect support information for Intel NPU setup and runtime issues.

.DESCRIPTION
    Writes a JSON report with local hardware, compatibility evaluation, conda
    environment details, and key package versions. Secrets are redacted: HF
    token values are never written, only presence and length.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\setup\collect_support_info.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\setup\collect_support_info.ps1 -NoFileOutput
#>

[CmdletBinding()]
param(
    [string]$CompatibilityFile = '',
    [string]$OutputPath = '',
    [string]$EnvName = '',
    [switch]$SkipDriverCheck,
    [switch]$NoFileOutput
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot 'preflight_common.ps1')

if ([string]::IsNullOrWhiteSpace($CompatibilityFile)) {
    $CompatibilityFile = Join-Path $PSScriptRoot 'compatibility.json'
}

if ([string]::IsNullOrWhiteSpace($EnvName)) {
    if ([string]::IsNullOrWhiteSpace($env:NPU_CONDA_ENV)) {
        $EnvName = 'ipex-npu'
    } else {
        $EnvName = $env:NPU_CONDA_ENV
    }
}

$RepoRoot = Split-Path -Parent $PSScriptRoot

function Get-CondaBasePath {
    $condaCmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($condaCmd) {
        $source = [string]$condaCmd.Source
        if ($source -match '\\condabin\\conda\.(bat|exe)$') {
            return Split-Path (Split-Path $source -Parent) -Parent
        }
        if ($source -match '\\Scripts\\conda\.exe$') {
            return Split-Path (Split-Path $source -Parent) -Parent
        }
    }

    $userHome = $env:USERPROFILE
    $candidates = @(
        "$userHome\miniconda3",
        "$userHome\Miniconda3",
        "$userHome\anaconda3",
        "$userHome\Anaconda3",
        "$env:LOCALAPPDATA\miniconda3",
        "$env:LOCALAPPDATA\Miniconda3",
        "$env:LOCALAPPDATA\anaconda3",
        "$env:LOCALAPPDATA\Anaconda3",
        'C:\ProgramData\miniconda3',
        'C:\ProgramData\Miniconda3',
        'C:\ProgramData\anaconda3',
        'C:\ProgramData\Anaconda3'
    )

    foreach ($candidate in $candidates) {
        if (Test-Path (Join-Path $candidate 'Scripts\conda.exe')) {
            return $candidate
        }
    }

    return $null
}

function Get-EnvFileInfo {
    param([string]$RepoRootPath)

    $envFile = Join-Path $RepoRootPath 'intel-npu-llm\.env'
    $tokenPresent = $false
    $tokenLength = 0

    if (Test-Path $envFile) {
        $content = Get-Content -Path $envFile -Raw
        $match = [regex]::Match($content, 'HF_TOKEN=(.+)')
        if ($match.Success) {
            $tokenValue = $match.Groups[1].Value.Trim()
            if (-not [string]::IsNullOrWhiteSpace($tokenValue)) {
                $tokenPresent = $true
                $tokenLength = $tokenValue.Length
            }
        }
    }

    return [ordered]@{
        path = $envFile
        exists = (Test-Path $envFile)
        hfTokenPresent = $tokenPresent
        hfTokenLength = $tokenLength
    }
}

function Get-PackageDiagnostics {
    param([string]$PythonPath)

    if ([string]::IsNullOrWhiteSpace($PythonPath) -or -not (Test-Path $PythonPath)) {
        return [ordered]@{
            pythonPath = $PythonPath
            available = $false
            packageVersions = @{}
            moduleChecks = @{}
        }
    }

    $pythonCode = @'
import importlib.util
import json
from importlib.metadata import PackageNotFoundError, version

packages = [
    "ipex-llm",
    "bigdl-core-npu",
    "neural-compressor",
    "torch",
    "transformers",
    "setuptools",
    "fastapi",
    "uvicorn",
    "pydantic",
    "python-certifi-win32",
]

modules = [
    "ipex_llm",
    "neural_compressor",
    "neural_compressor.adaptor",
    "torch",
    "transformers",
    "certifi_win32",
]

package_versions = {}
for package in packages:
    try:
        package_versions[package] = version(package)
    except PackageNotFoundError:
        package_versions[package] = None

module_checks = {}
for module_name in modules:
    module_checks[module_name] = importlib.util.find_spec(module_name) is not None

https_trust = {
    "certifiBundle": None,
    "certifiWin32Active": False,
}

try:
    import certifi

    if module_checks["certifi_win32"]:
        import certifi_win32  # noqa: F401
        https_trust["certifiWin32Active"] = True

    https_trust["certifiBundle"] = certifi.where()
except Exception as exc:
    https_trust["error"] = str(exc)

print(json.dumps({
    "packageVersions": package_versions,
    "moduleChecks": module_checks,
    "httpsTrust": https_trust,
}))
'@

    $tempScriptPath = Join-Path ([System.IO.Path]::GetTempPath()) ([System.Guid]::NewGuid().ToString() + '.py')
    [System.IO.File]::WriteAllText($tempScriptPath, $pythonCode, [System.Text.UTF8Encoding]::new($false))

    try {
        $rawJson = & $PythonPath $tempScriptPath
    } finally {
        if (Test-Path $tempScriptPath) {
            Remove-Item $tempScriptPath -Force -ErrorAction SilentlyContinue
        }
    }

    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($rawJson)) {
        return [ordered]@{
            pythonPath = $PythonPath
            available = $true
            packageVersions = @{}
            moduleChecks = @{}
        }
    }

    $parsed = $rawJson | ConvertFrom-Json
    return [ordered]@{
        pythonPath = $PythonPath
        available = $true
        packageVersions = $parsed.packageVersions
        moduleChecks = $parsed.moduleChecks
        httpsTrust = $parsed.httpsTrust
    }
}

$compatibility = Get-CompatibilityConfig -CompatibilityFile $CompatibilityFile
$snapshot = Get-LocalHardwareSnapshot -Compatibility $compatibility -SkipDriverCheck:$SkipDriverCheck
$report = Test-HardwareCompatibility -Snapshot $snapshot -Compatibility $compatibility -SkipDriverCheck:$SkipDriverCheck

$condaBase = Get-CondaBasePath
$envRoot = $null
$envPython = $null
$envExists = $false
if ($condaBase) {
    $envRoot = Join-Path $condaBase (Join-Path 'envs' $EnvName)
    $envPython = Join-Path $envRoot 'python.exe'
    $envExists = Test-Path $envPython
}

$hfToken = $env:HF_TOKEN
if ([string]::IsNullOrWhiteSpace($hfToken)) {
    $hfToken = $env:HUGGING_FACE_HUB_TOKEN
}

$payload = [ordered]@{
    generatedAtUtc = [DateTime]::UtcNow.ToString('o')
    repoRoot = $RepoRoot
    compatibility = [ordered]@{
        compatibilityFile = $CompatibilityFile
        minimumWindowsBuild = [int]$compatibility.minimumWindowsBuild
        minimumNpuDriverVersion = [string]$compatibility.minimumNpuDriverVersion
        passed = [bool]$report.Passed
        errors = @($report.Errors)
        warnings = @($report.Warnings)
        notes = @($report.Notes)
        matchedValidatedCombos = @($report.MatchedValidatedCombos)
        matchedProblemCombos = @($report.MatchedProblemCombos)
    }
    hardware = [ordered]@{
        cpuName = $report.CpuName
        windowsProduct = $report.WindowsProduct
        windowsBuild = $report.WindowsBuild
        biosVendor = $report.BiosVendor
        biosVersion = $report.BiosVersion
        biosDate = $report.BiosDate
        profileId = if ($report.Profile) { [string]$report.Profile.id } else { $null }
        profileLabel = if ($report.Profile) { [string]$report.Profile.label } else { $null }
        requiredEnv = $report.RequiredEnv
        npuDriver = if ($report.Driver) {
            [ordered]@{
                deviceName = [string]$report.Driver.DeviceName
                driverVersion = [string]$report.Driver.DriverVersion
                manufacturer = [string]$report.Driver.Manufacturer
            }
        } else {
            $null
        }
    }
    runtime = [ordered]@{
        condaBasePath = $condaBase
        envName = $EnvName
        envRoot = $envRoot
        envExists = $envExists
        packageDiagnostics = Get-PackageDiagnostics -PythonPath $envPython
    }
    environment = [ordered]@{
        NPU_CONDA_ENV = $env:NPU_CONDA_ENV
        IPEX_LLM_NPU_MTL = $env:IPEX_LLM_NPU_MTL
        NPU_ALLOW_UNSUPPORTED = $env:NPU_ALLOW_UNSUPPORTED
        NPU_SKIP_DRIVER_CHECK = $env:NPU_SKIP_DRIVER_CHECK
        NPU_SKIP_PREFLIGHT = $env:NPU_SKIP_PREFLIGHT
        HF_HOME = $env:HF_HOME
        REQUESTS_CA_BUNDLE = $env:REQUESTS_CA_BUNDLE
        SSL_CERT_FILE = $env:SSL_CERT_FILE
        hfTokenPresent = (-not [string]::IsNullOrWhiteSpace($hfToken))
        hfTokenLength = if ([string]::IsNullOrWhiteSpace($hfToken)) { 0 } else { $hfToken.Length }
        envFile = Get-EnvFileInfo -RepoRootPath $RepoRoot
    }
}

$json = $payload | ConvertTo-Json -Depth 10

if (-not $NoFileOutput) {
    if ([string]::IsNullOrWhiteSpace($OutputPath)) {
        $fileName = 'support-info-{0}.json' -f (Get-Date -Format 'yyyyMMdd-HHmmss')
        $OutputPath = Join-Path $RepoRoot $fileName
    }

    [System.IO.File]::WriteAllText($OutputPath, $json, [System.Text.UTF8Encoding]::new($false))
    Write-Host "Support info written to: $OutputPath" -ForegroundColor Green
}

Write-Output $json