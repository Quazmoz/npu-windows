@echo off
setlocal enabledelayedexpansion
REM ============================================
REM   Intel NPU LLM Backend Server
REM ============================================
REM 
REM Auto-detects Intel Core Ultra processor and configures NPU appropriately.
REM
REM Supported:
REM   - Core Ultra Series 1 (Meteor Lake): 1xxH/U - sets IPEX_LLM_NPU_MTL=1
REM   - Core Ultra Series 2 (Arrow Lake): 2xxK/H - no special config
REM   - Core Ultra (Lunar Lake): 2xxV - no special config
REM
REM Usage:
REM   start_backend.bat              - Load default models
REM   start_backend.bat --list       - Show all available models
REM   start_backend.bat --models X   - Load specific models

echo ========================================
echo   Intel NPU LLM Backend Server
echo ========================================
echo.

REM ---- Auto-detect Intel Core Ultra Series ----
REM Get CPU name and write to temp file to avoid parentheses issues
powershell -NoProfile -Command "(Get-CimInstance -ClassName Win32_Processor).Name" > "%TEMP%\cpu_name.txt"
set /p CPU_NAME=<"%TEMP%\cpu_name.txt"
del "%TEMP%\cpu_name.txt" 2>nul

echo Detected CPU: !CPU_NAME!

REM Check for Meteor Lake (Series 1) by looking for "1" followed by two digits and H/U
REM Examples: 185H, 165H, 155H, 125U
echo !CPU_NAME! | findstr /r "1[0-9][0-9][HU]" >nul
if !errorlevel!==0 (
    echo Processor: Intel Core Ultra Series 1 - Meteor Lake
    set IPEX_LLM_NPU_MTL=1
) else (
    echo !CPU_NAME! | findstr /i "Ultra" >nul
    if !errorlevel!==0 (
        echo Processor: Intel Core Ultra Series 2 / Lunar Lake
    ) else (
        echo WARNING: Intel Core Ultra processor not detected
        echo This software requires an Intel Core Ultra with NPU
    )
)

if defined IPEX_LLM_NPU_MTL (
    echo NPU Config: IPEX_LLM_NPU_MTL=1 - required for Meteor Lake
) else (
    echo NPU Config: Native mode
)
echo.

REM ---- Activate conda environment ----
set "CONDA_PATH="
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" set "CONDA_PATH=%USERPROFILE%\miniconda3"
if not defined CONDA_PATH if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" set "CONDA_PATH=%USERPROFILE%\anaconda3"

if not defined CONDA_PATH (
    echo ERROR: Conda installation not found
    echo.
    echo Please install Miniconda: https://docs.conda.io/en/latest/miniconda.html
    echo Then run:
    echo   conda create -n ipex-npu python=3.11 -y
    echo   conda activate ipex-npu
    echo   pip install --pre --upgrade ipex-llm[npu]
    echo   pip install fastapi uvicorn pydantic
    pause
    exit /b 1
)

call "!CONDA_PATH!\Scripts\activate.bat" ipex-npu
if errorlevel 1 (
    echo ERROR: Could not activate 'ipex-npu' environment
    echo Run: conda create -n ipex-npu python=3.11 -y
    pause
    exit /b 1
)

echo Conda: !CONDA_PATH! [ipex-npu]
echo.

cd /d "%~dp0intel-npu-llm"

REM ---- Start server ----
if "%~1"=="" (
    echo Loading default model: qwen1.5-1.8b ^(verified working^)
    echo.
    python npu_server.py --models "qwen1.5-1.8b"
) else (
    python npu_server.py %*
)

endlocal
