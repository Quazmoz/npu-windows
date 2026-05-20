<#
.SYNOPSIS
    Full environment setup  -  run this once after cloning the repo.

.DESCRIPTION
    Walks through all setup steps in order:
      1. Install / locate Miniconda and add conda to PATH
      2. Create the 'ipex-npu' conda environment (Python 3.11)
      3. Install ipex-llm[npu] and server dependencies
      4. (Optional) Set a HuggingFace token for gated models

    Each step is idempotent  -  safe to re-run if something fails partway through.

.EXAMPLE
    # Run from the repo root (npu-windows\)
    powershell -ExecutionPolicy Bypass -File .\setup\setup_all.ps1

    # Skip the HF token prompt
    powershell -ExecutionPolicy Bypass -File .\setup\setup_all.ps1 -SkipHfToken

.EXAMPLE
     # Continue on unknown or unsupported hardware after showing warnings
     powershell -ExecutionPolicy Bypass -File .\setup\setup_all.ps1 -AllowUnsupportedHardware

.EXAMPLE
     # Use a custom conda environment name
     powershell -ExecutionPolicy Bypass -File .\setup\setup_all.ps1 -EnvName my-ipex-npu
#>

param(
    [switch]$SkipHfToken,
    [switch]$SkipHardwareCheck,
    [switch]$AllowUnsupportedHardware,
    [switch]$SkipDriverCheck,
    [string]$EnvName = 'ipex-npu',
    [string]$PythonVersion = '3.11'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$SetupDir = $PSScriptRoot

function Write-Banner {
    param([string]$Msg)
    $line = '=' * 42
    Write-Host ""
    Write-Host $line -ForegroundColor Magenta
    Write-Host "  $Msg" -ForegroundColor Magenta
    Write-Host $line -ForegroundColor Magenta
}

function Invoke-Step {
    param(
        [string]$Script,
        [string]$Label,
        [hashtable]$Arguments = @{}
    )

    Write-Banner $Label
    # Dot-source so that any $env:Path changes made by step 1 persist for
    # subsequent steps in this same session.
    . (Join-Path $SetupDir $Script) @Arguments
    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "  [X] '$Script' exited with code $LASTEXITCODE. Fix the error above and re-run." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Magenta
Write-Host "  Intel NPU LLM  -  Environment Setup" -ForegroundColor Magenta
Write-Host "==========================================" -ForegroundColor Magenta
Write-Host ""
Write-Host "  This will set up everything needed to run start_server.bat."
Write-Host "  Each step is skipped automatically if already complete."
Write-Host "  Conda env: $EnvName | Python: $PythonVersion"
Write-Host ""

if (-not $SkipHardwareCheck) {
    Invoke-Step '00_hardware_preflight.ps1' 'Preflight  -  Hardware compatibility' @{
        AllowUnsupportedHardware = $AllowUnsupportedHardware
        SkipDriverCheck = $SkipDriverCheck
    }
}

Invoke-Step '01_install_miniconda.ps1' 'Step 1/4  -  Miniconda'

# After step 1, conda.exe may now be on PATH for this session.
# Verify before continuing to step 2.
$condaOk = Get-Command conda -ErrorAction SilentlyContinue
if (-not $condaOk) {
    Write-Host ""
    Write-Host "  [!] conda is not yet available in this session." -ForegroundColor Yellow
    Write-Host "  [!] This usually means Miniconda was just installed for the first time." -ForegroundColor Yellow
    Write-Host "  [!] Please CLOSE this terminal, open a NEW PowerShell window, then run:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "        powershell -ExecutionPolicy Bypass -File .\setup\setup_all.ps1" -ForegroundColor White
    Write-Host ""
    exit 0
}

Invoke-Step '02_create_env.ps1'  'Step 2/4  -  Conda environment' @{
    EnvName = $EnvName
    PythonVersion = $PythonVersion
}
Invoke-Step '03_install_deps.ps1' 'Step 3/4  -  Python dependencies' @{
    EnvName = $EnvName
}

if (-not $SkipHfToken) {
    Write-Banner 'Step 4/4  -  HuggingFace token (optional)'
    Write-Host ""
    Write-Host "  Models like Qwen, DeepSeek, and MiniCPM work without a token." -ForegroundColor Gray
    Write-Host "  Only needed for Llama 2 / 3 / 3.2." -ForegroundColor Gray
    Write-Host ""
    $answer = Read-Host "  Set up a HuggingFace token now? [y/N]"
    if ($answer -match '^[Yy]$') {
        Invoke-Step '04_hf_token.ps1' 'Step 4/4  -  HuggingFace token'
    } else {
        Write-Host "  Skipping. Run .\setup\04_hf_token.ps1 any time to add a token later." -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  To start the backend:" -ForegroundColor White
Write-Host "    .\start_server.bat" -ForegroundColor Cyan
if ($EnvName -ne 'ipex-npu') {
    Write-Host "  Before starting, set NPU_CONDA_ENV=$EnvName in that terminal." -ForegroundColor Yellow
}
Write-Host ""
Write-Host "  To run conda init for future terminal sessions:" -ForegroundColor White
Write-Host "    conda init powershell" -ForegroundColor Cyan
Write-Host ""
