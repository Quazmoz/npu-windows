<#
.SYNOPSIS
    Step 1 of 4  -  Ensure Miniconda is installed and conda is on PATH.

.DESCRIPTION
    Checks whether conda is already available. If not, searches common install
    locations and adds them to the user PATH. If truly absent, installs
    Miniconda3 via winget and then patches the current session's PATH so the
    remaining setup scripts can proceed without reopening the terminal.

    Safe to re-run  -  skips installation if conda is already working.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\setup\01_install_miniconda.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Step  { param([string]$Msg) Write-Host "  -> $Msg" -ForegroundColor Cyan }
function Write-OK    { param([string]$Msg) Write-Host "  [OK] $Msg" -ForegroundColor Green }
function Write-Warn  { param([string]$Msg) Write-Host "  [!]  $Msg" -ForegroundColor Yellow }
function Write-Fail  { param([string]$Msg) Write-Host "  [X]  $Msg" -ForegroundColor Red }

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  Step 1: Miniconda / conda check" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# -- Helper: add paths to both the persistent user PATH and the live session --
function Add-ToUserPath {
    param([string[]]$Dirs)
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User') -split ';' |
                Where-Object { $_ -ne '' }
    $added = @()
    foreach ($d in $Dirs) {
        if ($d -and (Test-Path $d) -and ($userPath -notcontains $d)) {
            $userPath += $d
            $added   += $d
        }
    }
    if ($added) {
        [Environment]::SetEnvironmentVariable('Path', ($userPath -join ';'), 'User')
        # Also update the live session immediately
        $env:Path = ($env:Path.TrimEnd(';') + ';' + ($added -join ';'))
        foreach ($d in $added) { Write-OK "Added to PATH: $d" }
    }
}

# -- Helper: given a conda base dir, return the dirs that should be on PATH --
function Get-CondaPathDirs {
    param([string]$Base)
    $dirs = @()
    if (Test-Path "$Base\condabin")      { $dirs += "$Base\condabin" }
    if (Test-Path "$Base\Scripts")       { $dirs += "$Base\Scripts" }
    if (Test-Path "$Base\Library\bin")   { $dirs += "$Base\Library\bin" }
    return $dirs
}

# -- 1. Quick win: already on PATH? -------------------------------------------
$condaCmd = Get-Command conda -ErrorAction SilentlyContinue
if ($condaCmd) {
    Write-OK "conda is already available: $($condaCmd.Source)"
    Write-OK "Version: $(& conda --version 2>&1)"
    exit 0
}

# -- 2. Search well-known install locations -----------------------------------
Write-Step "conda not found on PATH; searching common install locations..."

$userHome = $env:USERPROFILE
$searchRoots = @(
    "$userHome\Miniconda3", "$userHome\miniconda3",
    "$userHome\Anaconda3",  "$userHome\anaconda3",
    "$userHome\AppData\Local\miniconda3",
    "$userHome\AppData\Local\anaconda3",
    'C:\ProgramData\Miniconda3', 'C:\ProgramData\Anaconda3',
    'C:\Miniconda3', 'C:\Anaconda3'
)

$condaBase = $null
foreach ($root in $searchRoots) {
    if (Test-Path "$root\Scripts\conda.exe") {
        $condaBase = $root
        break
    }
}

if ($condaBase) {
    Write-OK "Found conda installation: $condaBase"
    Add-ToUserPath (Get-CondaPathDirs $condaBase)

    # Verify it works now
    $v = & conda --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-OK "conda is now available: $v"
        Write-Host ""
        Write-Warn "Tip: run 'conda init powershell' once in a NEW terminal to enable 'conda activate' for all future sessions."
        exit 0
    }
}

# -- 3. Not found  -  install via winget ----------------------------------------
Write-Step "No existing conda installation found."
Write-Step "Installing Miniconda3 via winget (this may take a few minutes)..."

$winget = Get-Command winget -ErrorAction SilentlyContinue
if (-not $winget) {
    Write-Fail "winget is not available."
    Write-Fail "Please install Miniconda manually: https://docs.conda.io/en/latest/miniconda.html"
    Write-Fail "Then re-run this script."
    exit 1
}

& winget install Anaconda.Miniconda3 --silent --accept-package-agreements --accept-source-agreements
if ($LASTEXITCODE -ne 0) {
    Write-Fail "winget installation failed (exit $LASTEXITCODE)."
    Write-Fail "Try installing manually: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
}

Write-OK "Miniconda3 installed."

# Refresh search after install
$condaBase = $null
foreach ($root in $searchRoots) {
    if (Test-Path "$root\Scripts\conda.exe") {
        $condaBase = $root
        break
    }
}

if (-not $condaBase) {
    Write-Fail "Installation succeeded but conda.exe not found in expected locations."
    Write-Fail "Check the install location manually and re-run this script."
    exit 1
}

Add-ToUserPath (Get-CondaPathDirs $condaBase)

Write-Host ""
Write-OK "Miniconda3 ready at: $condaBase"
Write-Warn "Close and reopen this terminal once, then continue with step 2."
Write-Warn "Or, if you are running setup_all.ps1, it handles this automatically."

# Verify for the current session
$v = & conda --version 2>&1
if ($LASTEXITCODE -eq 0) { Write-OK "conda version: $v" }
