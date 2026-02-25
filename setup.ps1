# ============================================================
# setup.ps1 - Avionic Anomaly Detection Pipeline
# One-click virtual environment setup for Windows PowerShell
# Usage: .\setup.ps1
# ============================================================

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Avionic Anomaly Detection Pipeline Setup " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# ---- Step 1: Check Python ----
Write-Host "[1/4] Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python is not installed or not on PATH." -ForegroundColor Red
    Write-Host "       Please install Python 3.11+ from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}
Write-Host "      Found: $pythonVersion" -ForegroundColor Green

# ---- Step 2: Create virtual environment ----
Write-Host ""
Write-Host "[2/4] Creating virtual environment (.venv)..." -ForegroundColor Yellow
if (Test-Path ".venv") {
    Write-Host "      .venv already exists, skipping creation." -ForegroundColor DarkGray
} else {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
    Write-Host "      Virtual environment created successfully." -ForegroundColor Green
}

# ---- Step 3: Install dependencies ----
Write-Host ""
Write-Host "[3/4] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
& ".\.venv\Scripts\pip.exe" install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Dependency installation failed." -ForegroundColor Red
    exit 1
}
Write-Host "      All dependencies installed successfully." -ForegroundColor Green

# ---- Step 4: Done ----
Write-Host ""
Write-Host "[4/4] Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  To activate your environment, run:" -ForegroundColor White
Write-Host ""
Write-Host "  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process" -ForegroundColor Yellow
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Then run the data loader:" -ForegroundColor White
Write-Host "  python data_loaders.py" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
