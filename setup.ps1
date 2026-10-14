$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Creating virtual environment in .venv ..."
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip

Write-Host "Installing PyTorch (CUDA 12.1 wheels) ..."
try {
    & .\.venv\Scripts\python.exe -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 `
        --index-url https://download.pytorch.org/whl/cu121
} catch {
    Write-Host "CUDA wheel install failed; falling back to CPU PyTorch from PyPI ..."
    & .\.venv\Scripts\python.exe -m pip install "torch>=2.0.0" "torchvision>=0.15.0"
}

Write-Host "Installing remaining dependencies ..."
& .\.venv\Scripts\python.exe -m pip install timm numpy pandas scikit-learn opencv-python matplotlib Pillow PyYAML tqdm tabulate

Write-Host ""
Write-Host "Setup complete. Activate with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
