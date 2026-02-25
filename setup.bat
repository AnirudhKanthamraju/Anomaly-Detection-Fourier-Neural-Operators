@echo off
REM ============================================================
REM setup.bat - Avionic Anomaly Detection Pipeline
REM One-click virtual environment setup for Command Prompt
REM Usage: setup.bat
REM ============================================================

echo.
echo ============================================
echo   Avionic Anomaly Detection Pipeline Setup
echo ============================================
echo.

REM ---- Step 1: Check Python ----
echo [1/4] Checking Python installation...
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo ERROR: Python is not installed or not on PATH.
    echo        Please install Python 3.11+ from https://www.python.org/downloads/
    pause
    exit /b 1
)
FOR /F "tokens=*" %%i IN ('python --version') DO echo       Found: %%i

REM ---- Step 2: Create virtual environment ----
echo.
echo [2/4] Creating virtual environment (.venv)...
IF EXIST ".venv\" (
    echo       .venv already exists, skipping creation.
) ELSE (
    python -m venv .venv
    IF ERRORLEVEL 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo       Virtual environment created successfully.
)

REM ---- Step 3: Install dependencies ----
echo.
echo [3/4] Installing dependencies from requirements.txt...
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
.venv\Scripts\pip.exe install -r requirements.txt
IF ERRORLEVEL 1 (
    echo ERROR: Dependency installation failed.
    pause
    exit /b 1
)
echo       All dependencies installed successfully.

REM ---- Step 4: Done ----
echo.
echo [4/4] Setup complete!
echo.
echo ============================================
echo   To activate your environment, run:
echo.
echo   .venv\Scripts\activate
echo.
echo   Then run the data loader:
echo   python data_loaders.py
echo ============================================
echo.
pause
