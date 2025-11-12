# Lung Cancer Classification Setup Script
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "SETUP SCRIPT" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`n[1/7] Checking Python..." -ForegroundColor Green
$pyVer = python --version 2>&1
Write-Host "  $pyVer" -ForegroundColor Green

Write-Host "`n[2/7] Creating venv..." -ForegroundColor Green
if (!(Test-Path "lung_env")) {
    python -m venv lung_env
}
Write-Host "  Done" -ForegroundColor Green

Write-Host "`n[3/7] Activating..." -ForegroundColor Green
& ".\lung_env\Scripts\Activate.ps1"
Write-Host "  Done" -ForegroundColor Green

Write-Host "`n[4/7] Upgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip --quiet
Write-Host "  Done" -ForegroundColor Green

Write-Host "`n[5/7] Installing packages (5-10 min)..." -ForegroundColor Green
Write-Host "  Note: Using TensorFlow 2.20.0 (Python 3.13 compatible)" -ForegroundColor Yellow
pip install tensorflow==2.20.0 --quiet
pip install -r requirements.txt --quiet
Write-Host "  Done" -ForegroundColor Green

Write-Host "`n[6/7] Checking GPU..." -ForegroundColor Green
$gpuCode = "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print('Found:', len(gpus), 'GPU(s)')"
$gpuResult = python -c $gpuCode 2>&1
Write-Host "  $gpuResult" -ForegroundColor Cyan

Write-Host "`n[7/7] Creating dirs..." -ForegroundColor Green
"data\raw", "data\augmented", "data\processed", "saved_models", "results" | ForEach-Object {
    if (!(Test-Path $_)) { New-Item -ItemType Directory -Path $_ -Force | Out-Null }
}
Write-Host "  Done" -ForegroundColor Green

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "SETUP COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nNext: Download dataset (see DATASET_DOWNLOAD_GUIDE.md)" -ForegroundColor Yellow
