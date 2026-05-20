@echo off
setlocal enabledelayedexpansion
set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not defined NPU_CONDA_ENV set "NPU_CONDA_ENV=ipex-npu"
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
REM   start_server.bat                - Load default models
REM   start_server.bat --diagnose     - Write a support JSON report and exit
REM   start_server.bat --list         - Show all available models
REM   start_server.bat --models X     - Load specific models
REM   start_server.bat --port 8001    - Change default port
REM   start_server.bat --models X --port Y  - Multi-argument support
REM
REM Optional environment overrides:
REM   NPU_CONDA_ENV=my-ipex-npu       - Activate a non-default conda environment
REM   NPU_ALLOW_UNSUPPORTED=1         - Continue past hardware preflight failures
REM   NPU_SKIP_DRIVER_CHECK=1         - Skip only the NPU driver version check
REM   NPU_SKIP_PREFLIGHT=1            - Skip all hardware compatibility checks
REM   IPEX_LLM_NPU_MTL=1              - Force Meteor Lake runtime mode manually

echo ========================================
echo   Intel NPU LLM Backend Server
echo ========================================
echo.

if /i "%~1"=="--diagnose" (
    "%POWERSHELL_EXE%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup\collect_support_info.ps1"
    exit /b %errorlevel%
)

REM ---- Shared hardware preflight ----
if /i not "%NPU_SKIP_PREFLIGHT%"=="1" (
    set "PREFLIGHT_ARGS="
    if /i "%NPU_ALLOW_UNSUPPORTED%"=="1" set "PREFLIGHT_ARGS=!PREFLIGHT_ARGS! -AllowUnsupportedHardware"
    if /i "%NPU_SKIP_DRIVER_CHECK%"=="1" set "PREFLIGHT_ARGS=!PREFLIGHT_ARGS! -SkipDriverCheck"

    "%POWERSHELL_EXE%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup\00_hardware_preflight.ps1"!PREFLIGHT_ARGS!
    if errorlevel 1 (
        echo.
        echo ERROR: Hardware preflight failed.
        echo Set NPU_ALLOW_UNSUPPORTED=1 to continue anyway, or NPU_SKIP_PREFLIGHT=1 to bypass the check.
        pause
        exit /b 1
    )
    echo.
)

REM ---- Auto-detect Intel Core Ultra Series ----
REM Get CPU name and write to temp file to avoid parentheses issues
"%POWERSHELL_EXE%" -NoProfile -Command "(Get-CimInstance -ClassName Win32_Processor).Name" > "%TEMP%\cpu_name.txt"
set /p CPU_NAME=<"%TEMP%\cpu_name.txt"
del "%TEMP%\cpu_name.txt" 2>nul

echo Detected CPU: !CPU_NAME!

set "CPU_PROFILE=Intel Core Ultra (generation not mapped yet)"
set "NPU_CONFIG_LABEL=Native mode"

if defined IPEX_LLM_NPU_MTL (
    set "NPU_CONFIG_LABEL=IPEX_LLM_NPU_MTL=!IPEX_LLM_NPU_MTL! - pre-set override"
) else (
    echo !CPU_NAME! | findstr /r "Ultra.*1[0-9][0-9][A-Z]" >nul
    if !errorlevel!==0 (
        set "CPU_PROFILE=Intel Core Ultra Series 1 - Meteor Lake"
        set IPEX_LLM_NPU_MTL=1
        set "NPU_CONFIG_LABEL=IPEX_LLM_NPU_MTL=1 - required for Meteor Lake"
    ) else (
        echo !CPU_NAME! | findstr /r "Ultra.*2[0-9][0-9]V" >nul
        if !errorlevel!==0 (
            set "CPU_PROFILE=Intel Core Ultra Series 2 - Lunar Lake"
        ) else (
            echo !CPU_NAME! | findstr /r "Ultra.*2[0-9][0-9][HKS]" >nul
            if !errorlevel!==0 (
                set "CPU_PROFILE=Intel Core Ultra Series 2 - Arrow Lake"
            ) else (
                echo !CPU_NAME! | findstr /i "Ultra" >nul
                if !errorlevel! neq 0 (
                    set "CPU_PROFILE=Unsupported / not Intel Core Ultra"
                    echo WARNING: Intel Core Ultra processor not detected
                    echo This software requires an Intel Core Ultra with NPU
                )
            )
        )
    )
)

echo Processor: !CPU_PROFILE!
echo NPU Config: !NPU_CONFIG_LABEL!
echo.

REM ---- Activate conda environment ----
set "CONDA_PATH="
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat"                     set "CONDA_PATH=%USERPROFILE%\miniconda3"
if not defined CONDA_PATH if exist "%USERPROFILE%\Miniconda3\Scripts\activate.bat"  set "CONDA_PATH=%USERPROFILE%\Miniconda3"
if not defined CONDA_PATH if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat"   set "CONDA_PATH=%USERPROFILE%\anaconda3"
if not defined CONDA_PATH if exist "%USERPROFILE%\Anaconda3\Scripts\activate.bat"   set "CONDA_PATH=%USERPROFILE%\Anaconda3"
if not defined CONDA_PATH if exist "%LOCALAPPDATA%\miniconda3\Scripts\activate.bat" set "CONDA_PATH=%LOCALAPPDATA%\miniconda3"
if not defined CONDA_PATH if exist "%LOCALAPPDATA%\Miniconda3\Scripts\activate.bat" set "CONDA_PATH=%LOCALAPPDATA%\Miniconda3"
if not defined CONDA_PATH if exist "%LOCALAPPDATA%\anaconda3\Scripts\activate.bat"  set "CONDA_PATH=%LOCALAPPDATA%\anaconda3"
if not defined CONDA_PATH if exist "%LOCALAPPDATA%\Anaconda3\Scripts\activate.bat"  set "CONDA_PATH=%LOCALAPPDATA%\Anaconda3"
if not defined CONDA_PATH if exist "C:\ProgramData\miniconda3\Scripts\activate.bat" set "CONDA_PATH=C:\ProgramData\miniconda3"
if not defined CONDA_PATH if exist "C:\ProgramData\Miniconda3\Scripts\activate.bat" set "CONDA_PATH=C:\ProgramData\Miniconda3"
if not defined CONDA_PATH if exist "C:\ProgramData\anaconda3\Scripts\activate.bat"  set "CONDA_PATH=C:\ProgramData\anaconda3"
if not defined CONDA_PATH if exist "C:\ProgramData\Anaconda3\Scripts\activate.bat"  set "CONDA_PATH=C:\ProgramData\Anaconda3"

if not defined CONDA_PATH (
    echo ERROR: Conda installation not found
    echo.
    echo Please install Miniconda: https://docs.conda.io/en/latest/miniconda.html
    echo Then run:
    echo   conda create -n %NPU_CONDA_ENV% python=3.11 -y
    echo   conda activate %NPU_CONDA_ENV%
    echo   pip install --pre --upgrade ipex-llm[npu]
    echo   pip install -r intel-npu-llm\requirements.txt
    pause
    exit /b 1
)

call "!CONDA_PATH!\Scripts\activate.bat" %NPU_CONDA_ENV%
if errorlevel 1 (
    echo ERROR: Could not activate '%NPU_CONDA_ENV%' environment
    echo Run: conda create -n %NPU_CONDA_ENV% python=3.11 -y
    pause
    exit /b 1
)

echo Conda: !CONDA_PATH! [%NPU_CONDA_ENV%]
echo.

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

if not exist "%~dp0.deps_installed" (
    echo Installing dependencies ^(first run only^)...
    pip install -r "%~dp0intel-npu-llm\requirements.txt"
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies.
        pause
        exit /b 1
    )
    echo. > "%~dp0.deps_installed"
    echo Dependencies installed.
) else (
    echo Dependencies: OK ^(cached^)
)

python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('hf_xet') else 1)" >nul 2>&1
if errorlevel 1 (
    echo Installing optional hf_xet package for Hugging Face Xet downloads...
    pip install --quiet hf_xet
    if errorlevel 1 (
        echo WARNING: Failed to install hf_xet. Downloads will continue over regular HTTP.
    ) else (
        echo hf_xet installed.
    )
)
echo.

cd /d "%~dp0intel-npu-llm"

set SKIP_ENV_CHECK=
for %%A in (%*) do (
    if /i "%%~A"=="--list" set SKIP_ENV_CHECK=1
)

if not defined SKIP_ENV_CHECK (
    echo Verifying Intel NPU Python environment...
    python npu_server.py --check-env
    if errorlevel 1 (
        echo.
        echo ERROR: Intel NPU runtime check failed.
        pause
        exit /b 1
    )
    echo.
)

REM ---- Check port availability ----
set PORT=8000
for %%A in (%*) do (
    if "%%A"=="--port" set NEXT_IS_PORT=1
    if defined NEXT_IS_PORT if not "%%A"=="--port" (
        set PORT=%%A
        set NEXT_IS_PORT=
    )
)
netstat -ano 2>nul | findstr /r /c:":%PORT% .*LISTENING" >nul
if !errorlevel!==0 (
    echo ERROR: Port !PORT! is already in use.
    echo Stop the process using that port or pass --port XXXX to use a different one.
    pause
    exit /b 1
)

REM ---- Start server ----
if "%~1"=="" (
    echo Loading default model: qwen1.5-1.8b ^(verified working^)
    echo.
    python npu_server.py --models "qwen1.5-1.8b"
) else (
    python npu_server.py %*
)

endlocal
