# ============================================================
#  mjai - one-click setup (Windows PowerShell 5)
#
#  1. Create .venv (Python 3.10+)
#  2. Install Python deps (torch / numpy / maturin ...)
#  3. Compile Rust engine (maturin develop --release)
#
#  Run:  powershell -ExecutionPolicy Bypass -File .\setup.ps1
#  Pre:  Python 3.10+ and Rust (rustup) installed
# ============================================================

$ErrorActionPreference = "Continue"   # sub-process stderr must NOT kill the script

$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $PROJECT_ROOT

# --- Critical fix for Chinese Windows username ---
# maturin / cargo call CreateDirectoryW on $env:TEMP, which fails with
# os error 3 "path not found" when the path contains non-ASCII chars
# like C:\Users\张三\AppData\Local\Temp\.tmpXXXXXX
# Workaround: redirect TMP/TEMP into a pure-ASCII dir inside the project
$localTmp = Join-Path $PROJECT_ROOT ".tmp"
New-Item -ItemType Directory -Force -Path $localTmp | Out-Null
$env:TMP  = $localTmp
$env:TEMP = $localTmp

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  mjai - one-click setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "project root: $PROJECT_ROOT"
Write-Host "TMP/TEMP    : $localTmp" -ForegroundColor DarkGray
Write-Host ""

function Check-LastExit($stepName) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "ERROR at [$stepName] (exit=$LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# --- Step 0: prereqs ---
Write-Host "[0/5] Checking prerequisites..." -ForegroundColor Yellow

$pythonCmd = $null
foreach ($c in @("python", "py -3.10", "py -3", "py")) {
    try {
        $verStr = Invoke-Expression "$c --version 2>&1"
        if ($LASTEXITCODE -eq 0) {
            $pythonCmd = $c
            Write-Host "  Python : $verStr" -ForegroundColor Green
            break
        }
    } catch {}
}
if (-not $pythonCmd) { Write-Host "ERROR: Python 3.10+ required" -ForegroundColor Red; exit 1 }

if (-not (Get-Command rustc -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: rustc not found, install via rustup" -ForegroundColor Red; exit 1
}
Write-Host "  Rust   : $(rustc --version)" -ForegroundColor Green
Write-Host "  Cargo  : $(cargo --version)" -ForegroundColor Green

# --- Step 1: .venv ---
Write-Host ""
Write-Host "[1/5] Creating .venv..." -ForegroundColor Yellow

$venvDir = Join-Path $PROJECT_ROOT ".venv"
$venvPy  = Join-Path $venvDir "Scripts\python.exe"
$venvPip = Join-Path $venvDir "Scripts\pip.exe"

if (Test-Path $venvPy) {
    Write-Host "  .venv exists, skip" -ForegroundColor DarkGray
} else {
    Invoke-Expression "$pythonCmd -m venv .venv"
    Check-LastExit "venv create"
    Write-Host "  .venv created" -ForegroundColor Green
}

Write-Host "  Upgrading pip + setuptools..."
& $venvPy -m pip install --upgrade pip setuptools wheel 2>$null | Out-Null

# --- Step 2: maturin ---
Write-Host ""
Write-Host "[2/5] Installing maturin..." -ForegroundColor Yellow
& $venvPip install 'maturin>=1.0,<2.0' 2>$null | Out-Host
Check-LastExit "maturin install"
Write-Host "  maturin installed" -ForegroundColor Green

# --- Step 3: Python deps ---
Write-Host ""
Write-Host "[3/5] Installing Python dependencies..." -ForegroundColor Yellow

$torchAlready = ($(& $venvPy -c 'import torch' 2>$null; $LASTEXITCODE) -eq 0)
if ($torchAlready) {
    $tv = & $venvPy -c 'import torch; print(torch.__version__)' 2>$null
    Write-Host "  torch already installed: $tv" -ForegroundColor DarkGray
} else {
    Write-Host "  Installing torch (CPU)..."
    & $venvPip install torch torchvision --index-url https://download.pytorch.org/whl/cpu 2>$null | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Fallback: default index..." -ForegroundColor DarkYellow
        & $venvPip install torch torchvision 2>$null | Out-Host
    }
    Check-LastExit "torch install"
}

foreach ($dep in @("numpy>=1.24", "tensorboard>=2.14", "tqdm>=4.65", "tomli>=2.0")) {
    & $venvPip install $dep 2>$null | Out-Null
}
Write-Host "  Python deps done" -ForegroundColor Green

# --- Step 4: Rust compile ---
Write-Host ""
Write-Host "[4/5] Compiling Rust engine (maturin develop --release)..." -ForegroundColor Yellow

Push-Location (Join-Path $PROJECT_ROOT "engine")
& $venvPip install maturin 2>$null | Out-Null
$maturinBin = Join-Path $venvDir "Scripts\maturin.exe"
if (-not (Test-Path $maturinBin)) { $maturinBin = "maturin" }
& $maturinBin develop --release
$maturinExit = $LASTEXITCODE
Pop-Location

if ($maturinExit -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Rust compile failed (exit=$maturinExit)" -ForegroundColor Red
    Write-Host "  If you see 'Failed to create temporary directory'," -ForegroundColor DarkYellow
    Write-Host "  the TMP/TEMP redirect at the top of this script should fix it." -ForegroundColor DarkYellow
    exit $maturinExit
}
Write-Host "  Rust engine compiled (.pyd in .venv)" -ForegroundColor Green

# --- Step 4.5: patch __init__.py (maturin 1.15 buggy template) ---
Write-Host ""
Write-Host "[4.5] Patching mjai_engine __init__.py..." -ForegroundColor Yellow
$initPy = Join-Path $venvDir "Lib\site-packages\mjai_engine\__init__.py"
@"
from . import mjai_engine as _native
from .mjai_engine import *
__doc__ = _native.__doc__
if hasattr(_native, "__all__"):
    __all__ = _native.__all__
"@ | Set-Content -Path $initPy -Encoding UTF8
$pycache = Join-Path $venvDir "Lib\site-packages\mjai_engine\__pycache__"
if (Test-Path $pycache) { Remove-Item -Recurse -Force $pycache }
Write-Host "  __init__.py patched" -ForegroundColor Green

# --- Step 5: install mjai + verify ---
Write-Host ""
Write-Host "[5/5] Installing mjai (editable) + verify..." -ForegroundColor Yellow

& $venvPip install -e $PROJECT_ROOT 2>$null | Out-Host
Check-LastExit "mjai editable install"

Write-Host ""
Write-Host "  --- Verify ---" -ForegroundColor Cyan

$verifyScript = Join-Path $PROJECT_ROOT "scripts\_verify_setup.py"
& $venvPy $verifyScript 2>&1 | ForEach-Object { Write-Host "  $_" }
$ok = ($LASTEXITCODE -eq 0)

Write-Host ""
if ($ok) {
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "  SETUP COMPLETE" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  .\.venv\Scripts\Activate.ps1"
    Write-Host "  .\.venv\Scripts\python.exe scripts\train_large_pool.py"
    Write-Host ""
} else {
    Write-Host "VERIFY FAILED - scroll up for errors" -ForegroundColor Red
    exit 1
}
