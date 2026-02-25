# ============================================================
# setup.ps1 - Avionic Anomaly Detection Pipeline (FIXED)
# ============================================================

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Avionic Anomaly Detection Pipeline Setup " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# ---- Step 1: Force 64-bit Python ----
Write-Host "[1/4] Checking for 64-bit Python..." -ForegroundColor Yellow
$python64 = "C:\Program Files\Python312\python.exe"

if (Test-Path $python64) {
    $pythonExe = $python64
    Write-Host "      Found 64-bit Python at $pythonExe" -ForegroundColor Green
} else {
    Write-Host "WARNING: 64-bit path not found, using default 'python' command." -ForegroundColor Gray
    $pythonExe = "python"
}

# ---- Step 2: Create virtual environment ----
Write-Host ""
Write-Host "[2/4] Creating virtual environment (.venv)..." -ForegroundColor Yellow
if (Test-Path ".venv") {
    Write-Host "      Removing old .venv to ensure architecture match..." -ForegroundColor DarkGray
    Remove-Item -Recurse -Force ".venv"
}
& $pythonExe -m venv .venv
Write-Host "      Virtual environment created successfully." -ForegroundColor Green

# ---- Step 3: Install dependencies (Binary Only) ----
Write-Host ""
Write-Host "[3/4] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
# We force --only-binary to skip the C++ compiler check
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
& ".\.venv\Scripts\pip.exe" install -r requirements.txt --only-binary=:all:

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Dependency installation failed. Try pinning pandas==2.2.3 in requirements.txt." -ForegroundColor Red
    exit 1
}
Write-Host "      All dependencies installed successfully." -ForegroundColor Green

# ---- Step 4: Done ----
Write-Host ""
Write-Host "[4/4] Setup complete!" -ForegroundColor Green
Write-Host "Activate with: .\.venv\Scripts\Activate.ps1"