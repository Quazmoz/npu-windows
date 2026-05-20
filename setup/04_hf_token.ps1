<#
.SYNOPSIS
    Step 4 of 4 (Optional)  -  Set a HuggingFace token for gated models.

.DESCRIPTION
    Prompts for your HuggingFace access token and writes it to
    intel-npu-llm/.env so the backend can download gated models
    (Llama 2, Llama 3, Llama 3.2).

    Models that do NOT need a token: Qwen, DeepSeek, MiniCPM, GLM-Edge,
    Baichuan2. Skip this step if you're only using those.

    To get a token:
      1. Create an account at https://huggingface.co
      2. Accept the model license on the model page
      3. Generate a token at https://huggingface.co/settings/tokens

    Safe to re-run  -  overwrites the existing .env file.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\setup\04_hf_token.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Step { param([string]$Msg) Write-Host "  -> $Msg" -ForegroundColor Cyan }
function Write-OK   { param([string]$Msg) Write-Host "  [OK] $Msg" -ForegroundColor Green }
function Write-Fail { param([string]$Msg) Write-Host "  [X]  $Msg" -ForegroundColor Red }
function Write-Warn { param([string]$Msg) Write-Host "  [!]  $Msg" -ForegroundColor Yellow }

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "  Step 4 (Optional): HuggingFace token" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

$RepoRoot  = Split-Path -Parent $PSScriptRoot
$EnvFile   = Join-Path $RepoRoot 'intel-npu-llm\.env'

Write-Warn "Only needed for Llama 2 / Llama 3 / Llama 3.2."
Write-Warn "Skip this step (press Enter) if you only want to use Qwen, DeepSeek, MiniCPM, etc."
Write-Host ""

# Check for an existing token
if (Test-Path $EnvFile) {
    $existing = Get-Content $EnvFile -Raw
    if ($existing -match 'HF_TOKEN=hf_[A-Za-z0-9]+') {
        Write-OK "An HF_TOKEN already exists in $EnvFile"
        $overwrite = Read-Host "  Overwrite it? [y/N]"
        if ($overwrite -notmatch '^[Yy]$') {
            Write-OK "Keeping existing token. Skipping."
            exit 0
        }
    }
}

# NOTE: We intentionally use Read-Host (plain text) here rather than
#       Read-Host -AsSecureString so the token is immediately usable as a
#       string. The token only goes into a local .env file  -  it is never
#       transmitted by this script.
$token = Read-Host "  Enter your HuggingFace token (starts with hf_), or press Enter to skip"

if ([string]::IsNullOrWhiteSpace($token)) {
    Write-Warn "No token entered. Skipping. Gated models will fail to download."
    exit 0
}

if ($token -notmatch '^hf_[A-Za-z0-9]+$') {
    Write-Fail "That doesn't look like a valid HuggingFace token (expected 'hf_...')."
    Write-Fail "Double-check at https://huggingface.co/settings/tokens and re-run."
    exit 1
}

# Write .env with UTF-8 (no BOM)  -  important for the Python dotenv loader
$envDir = Split-Path $EnvFile
if (-not (Test-Path $envDir)) { New-Item -ItemType Directory -Path $envDir | Out-Null }

[System.IO.File]::WriteAllText($EnvFile, "HF_TOKEN=$token`n", [System.Text.UTF8Encoding]::new($false))

Write-OK ".env written to: $EnvFile"
Write-Warn "Keep this token private  -  do not commit .env to source control."
Write-Host ""
Write-OK "All done! Run: .\start_server.bat"
