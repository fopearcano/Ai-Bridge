@echo off
REM Launch the Ai-Bridge_Houdini CLI from a fresh checkout.
REM Bootstraps PYTHONPATH to .\src so `pip install -e .` is optional.
REM Usage: run_cli.bat [args...]
setlocal
set "DIR=%~dp0"
if "%AIBRIDGE_PYTHON%"=="" (set "PYTHON=python") else (set "PYTHON=%AIBRIDGE_PYTHON%")
set "PYTHONPATH=%DIR%src;%PYTHONPATH%"
"%PYTHON%" -m aibridge_houdini %*
exit /b %ERRORLEVEL%
