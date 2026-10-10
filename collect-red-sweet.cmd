@echo off
chcp 65001 >nul
setlocal
set "REPO=%~dp0"
set "PYTHON_EXE=%REPO%.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%REPO%..\rosmaster_m3_control\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo Python environment not found. See docs\RED_SWEET_RL.md.
    pause
    exit /b 1
)
pushd "%REPO%"
"%PYTHON_EXE%" -m pc_client.red_rl.run_session %*
set "RESULT=%ERRORLEVEL%"
popd
echo.
pause
exit /b %RESULT%
