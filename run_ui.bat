@echo off
REM Launch the Ai-Bridge_Houdini Qt UI from a fresh checkout.
REM Requires the [gui] extra: `pip install -e .[gui]` (or `pip install PySide6`).
REM Usage: run_ui.bat [args...]
setlocal
set "DIR=%~dp0"
if "%AIBRIDGE_PYTHON%"=="" (set "PYTHON=python") else (set "PYTHON=%AIBRIDGE_PYTHON%")
set "PYTHONPATH=%DIR%src;%PYTHONPATH%"
"%PYTHON%" -m aibridge_houdini --ui qt %*
exit /b %ERRORLEVEL%
