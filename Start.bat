@echo off
setlocal
cd /d "%~dp0"

set VENV_DIR=.venv
set PYEXE=%VENV_DIR%\Scripts\python.exe

if exist "%PYEXE%" goto :launch

echo ============================================================
echo   Urdu OCR - first-time setup
echo   This only happens once and can take a few minutes.
echo   Please keep this window open.
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on this computer.
    echo Please install Python 3.11 from https://www.python.org/downloads/
    echo ^(tick "Add python.exe to PATH" during install^), then run this file again.
    echo.
    pause
    exit /b 1
)

echo Creating the app environment...
python -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo.
    echo Could not create the app environment. See the message above.
    pause
    exit /b 1
)

echo Installing required components ^(this is the slow part^)...
"%PYEXE%" -m pip install --upgrade pip --quiet
"%PYEXE%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Installation failed. Check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo Checking Tesseract Urdu language data...
powershell -NoProfile -ExecutionPolicy Bypass -File "tools\check_tesseract_urdu.ps1"

echo.
echo Setup complete.
echo.

:launch
"%PYEXE%" -m app.main
if errorlevel 1 (
    echo.
    echo The app closed with an error. See the message above.
    pause
)
