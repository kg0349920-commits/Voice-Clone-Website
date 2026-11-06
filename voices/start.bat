@echo off
title Voxora AI Voice Studio

echo.
echo ==========================================
echo        VOXORA AI VOICE STUDIO
echo ==========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    uv venv --python 3.11 .venv
    if errorlevel 1 (
        echo Failed to create venv. Is uv installed?
        pause
        exit /b
    )
)

echo Installing requirements...
uv pip install -r requirements.txt --python ".venv\Scripts\python.exe"

echo.
echo Starting Voxora AI...
echo.

".venv\Scripts\python.exe" server.py

pause