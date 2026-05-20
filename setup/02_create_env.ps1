<#
.SYNOPSIS
    Step 2 of 4  -  Create the 'ipex-npu' conda environment (Python 3.11).

.DESCRIPTION
    Creates the conda environment required by the Intel NPU backend.
    Safe to re-run  -  skips creation if the environment already exists.

    Run AFTER 01_install_miniconda.ps1.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\setup\02_create_env.ps1
#>

[CmdletBinding()]
param(
    [string]$EnvName = 'ipex-npu',
    [string]$PythonVersion = '3.11'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Step { param([string]$Msg) Write-Host "  -> $Msg" -ForegroundColor Cyan }
function Write-OK   { param([string]$Msg) Write-Host "  [OK] $Msg" -ForegroundColor Green }
function Write-Fail { param([string]$Msg) Write-Host "  [X]  $Msg" -ForegroundColor Red }

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  Step 2: Create conda environment" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

$PyVersion = $PythonVersion

# -- Verify conda is available -------------------------------------------------
$condaCmd = Get-Command conda -ErrorAction SilentlyContinue
if (-not $condaCmd) {
    Write-Fail "conda not found on PATH."
    Write-Fail "Run 01_install_miniconda.ps1 first, then reopen this terminal."
    exit 1
}

# -- Check if environment already exists ---------------------------------------
Write-Step "Checking for existing '$EnvName' environment..."
$envList = & conda env list 2>&1 | Where-Object { $_ -match "^\s*$EnvName\s" }
if ($envList) {
    Write-OK "Environment '$EnvName' already exists  -  skipping creation."
    Write-OK "To recreate it, run: conda env remove -n $EnvName -y"
    exit 0
}

# -- Create environment --------------------------------------------------------
Write-Step "Creating conda environment '$EnvName' with Python $PyVersion..."
Write-Step "This may take a couple of minutes..."

& conda create -n $EnvName python=$PyVersion -y
if ($LASTEXITCODE -ne 0) {
    Write-Fail "Failed to create conda environment (exit $LASTEXITCODE)."
    exit 1
}

Write-Host ""
Write-OK "Environment '$EnvName' created with Python $PyVersion."
Write-OK "Proceed to step 3: install Python dependencies."
