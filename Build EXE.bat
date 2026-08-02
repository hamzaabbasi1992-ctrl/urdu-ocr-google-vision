@echo off
setlocal
cd /d "%~dp0"

set VENV_DIR=.venv
set PYEXE=%VENV_DIR%\Scripts\python.exe

if not exist "%PYEXE%" (
    echo Could not find %PYEXE% - set up the venv first (python -m venv .venv, then pip install -r requirements.txt -r requirements-dev.txt).
    pause
    exit /b 1
)

"%VENV_DIR%\Scripts\pyinstaller.exe" --noconfirm --windowed --onefile --name "Urdu OCR (Google Vision)" run_app.py
if errorlevel 1 (
    echo.
    echo Build failed. See the message above.
    pause
    exit /b 1
)

echo.
echo Built: dist\Urdu OCR (Google Vision).exe
echo Re-point the desktop shortcut at this file if it's not already.
pause
