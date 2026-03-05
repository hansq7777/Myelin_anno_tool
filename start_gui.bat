@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_CMD="
set "PYTHON_ARGS="

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_CMD=%~dp0.venv\Scripts\python.exe"
    goto resolve_done
)

if exist "%~dp0venv\Scripts\python.exe" (
    set "PYTHON_CMD=%~dp0venv\Scripts\python.exe"
    goto resolve_done
)

where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_CMD=py"
    set "PYTHON_ARGS=-3"
    goto resolve_done
)

set "PYTHON_CMD=python"

:resolve_done
if /I "%~1"=="--check" goto check_mode

echo [Myelin_anno_tool] Starting GUI...
"%PYTHON_CMD%" %PYTHON_ARGS% -m zstack_anno
if errorlevel 1 (
    echo.
    echo Failed to start GUI.
    echo Please run: pip install -r requirements.txt
    pause
)

endlocal
exit /b 0

:check_mode
echo [Myelin_anno_tool] Python command: %PYTHON_CMD% %PYTHON_ARGS%
"%PYTHON_CMD%" %PYTHON_ARGS% --version
exit /b %ERRORLEVEL%
