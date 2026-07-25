@echo off
setlocal
cd /d "%~dp0"

set VENV_DIR=.venv
set PYEXE=%VENV_DIR%\Scripts\python.exe

if not exist "%PYEXE%" (
    echo Could not find %PYEXE%
    echo Run Start.bat first to set up the app environment.
    pause
    exit /b 1
)

"%PYEXE%" -m app.simple_gui
if errorlevel 1 (
    echo.
    echo The app closed with an error. See the message above.
    pause
)
