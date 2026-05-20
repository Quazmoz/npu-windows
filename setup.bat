@echo off
REM ============================================
REM  Intel NPU LLM — First-Time Setup
REM ============================================
REM Runs the PowerShell setup scripts with execution
REM policy bypass so no manual policy change is needed.
REM
REM Just double-click this file or run from any terminal:
REM   .\setup.bat

echo ==========================================
echo   Intel NPU LLM - Environment Setup
echo ==========================================
echo.

"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup\setup_all.ps1" %*

if errorlevel 1 (
    echo.
    echo Setup did not complete successfully.
    echo Review the messages above, fix any errors, then run setup.bat again.
    pause
    exit /b 1
)

pause
