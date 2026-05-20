<#
.SYNOPSIS
    Step 3 of 4  -  Install ipex-llm[npu] and all server dependencies.

.DESCRIPTION
    Uses 'conda run' to install packages into the 'ipex-npu' environment
    without needing to activate it first. Order matters:
      1. ipex-llm[npu]   -  pins torch and transformers versions
      2. requirements.txt   -  must come AFTER ipex-llm to avoid overwriting pins

    Safe to re-run  -  pip's resolver will skip already-satisfied packages.

    Run AFTER 02_create_env.ps1.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\setup\03_install_deps.ps1
#>

[CmdletBinding()]
param(
    [string]$EnvName = 'ipex-npu'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Step { param([string]$Msg) Write-Host "  -> $Msg" -ForegroundColor Cyan }
function Write-OK   { param([string]$Msg) Write-Host "  [OK] $Msg" -ForegroundColor Green }
function Write-Fail { param([string]$Msg) Write-Host "  [X]  $Msg" -ForegroundColor Red }
function Write-Warn { param([string]$Msg) Write-Host "  [!]  $Msg" -ForegroundColor Yellow }

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  Step 3: Install Python dependencies" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Resolve the repo root (one level up from this script's directory)
$RepoRoot   = Split-Path -Parent $PSScriptRoot
$ReqFile    = Join-Path $RepoRoot 'intel-npu-llm\requirements.txt'

# -- Verify conda --------------------------------------------------------------
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Fail "conda not found. Run 01_install_miniconda.ps1 first, then reopen terminal."
    exit 1
}

# -- Verify the env exists -----------------------------------------------------
$envList = & conda env list 2>&1 | Where-Object { $_ -match "^\s*$EnvName\s" }
if (-not $envList) {
    Write-Fail "Conda environment '$EnvName' not found."
    Write-Fail "Run 02_create_env.ps1 first."
    exit 1
}

# -- Verify requirements file --------------------------------------------------
if (-not (Test-Path $ReqFile)) {
    Write-Fail "requirements.txt not found at: $ReqFile"
    Write-Fail "Make sure you are running this script from inside the npu-windows repo."
    exit 1
}

# -- Install ipex-llm[npu] -----------------------------------------------------
Write-Step "Installing ipex-llm[npu] (this is large  -  can take 5-15 minutes)..."
Write-Warn "torch and transformers versions are pinned by ipex-llm. Do not upgrade them separately."

& conda run -n $EnvName --no-capture-output pip install --pre --upgrade "ipex-llm[npu]"
if ($LASTEXITCODE -ne 0) {
    Write-Fail "ipex-llm[npu] installation failed (exit $LASTEXITCODE)."
    exit 1
}
Write-OK "ipex-llm[npu] installed."

# -- Pin neural-compressor + setuptools BEFORE requirements.txt ----------------
# ipex-llm pulls in neural-compressor 3.x which dropped the adaptor module.
# setuptools 71+ removed pkg_resources which neural-compressor 2.x requires.
Write-Step "Pinning neural-compressor==2.6 and setuptools<71 (required by ipex-llm NPU)..."
& conda run -n $EnvName --no-capture-output pip install --force-reinstall --quiet "neural-compressor==2.6" "setuptools<71"
if ($LASTEXITCODE -ne 0) {
    Write-Fail "Failed to pin neural-compressor/setuptools (exit $LASTEXITCODE)."
    exit 1
}
Write-OK "neural-compressor==2.6 and setuptools pinned."

# -- Install server dependencies -----------------------------------------------
Write-Step "Installing server dependencies from requirements.txt..."

& conda run -n $EnvName --no-capture-output pip install -r $ReqFile
if ($LASTEXITCODE -ne 0) {
    Write-Fail "requirements.txt installation failed (exit $LASTEXITCODE)."
    exit 1
}
Write-OK "Server dependencies installed."

# -- Install optional Hugging Face Xet transport ------------------------------
# Some model repos use Xet-backed storage. Without hf_xet, downloads still work
# but fall back to regular HTTP and emit a warning on startup.
Write-Step "Installing optional hf_xet package for faster Hugging Face downloads..."
& conda run -n $EnvName --no-capture-output pip install --quiet hf_xet
if ($LASTEXITCODE -ne 0) {
    Write-Warn "hf_xet installation failed (exit $LASTEXITCODE). Downloads will use regular HTTP."
} else {
    Write-OK "hf_xet installed."
}

# -- Mark deps as installed (start_server.bat checks this file) ---------------
$depsFlag = Join-Path $RepoRoot '.deps_installed'
if (-not (Test-Path $depsFlag)) {
    "" | Out-File -FilePath $depsFlag -Encoding ascii
    Write-OK "Created .deps_installed flag so start_server.bat skips re-install on first launch."
}

Write-Host ""
Write-OK "All dependencies installed into the '$EnvName' environment."
Write-OK "You are ready to run:  .\start_server.bat"
