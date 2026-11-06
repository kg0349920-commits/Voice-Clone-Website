```bat
@echo off

title VoxClone AI - Voice Studio

echo.
echo ==========================================
echo          VoxClone AI Voice Studio
echo ==========================================
echo.

if not exist ".venv\Scripts\python.exe" (

    echo Virtual environment not found.
    echo Creating virtual environment...
    echo.

    python -m venv .venv

    echo.
    echo Installing packages...
    echo This can take some time.
    echo.

    .venv\Scripts\python.exe -m pip install --upgrade pip

    .venv\Scripts\python.exe -m pip install -r requirements.txt

)

echo.
echo Starting VoxClone AI...
echo.

.venv\Scripts\python.exe server.py

pause
```