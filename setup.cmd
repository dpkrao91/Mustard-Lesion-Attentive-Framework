@echo off
setlocal
cd /d "%~dp0"

echo Creating virtual environment in .venv ...
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 goto :fail
)

call .venv\Scripts\activate.bat

echo Upgrading pip ...
python -m pip install --upgrade pip

echo Installing PyTorch (CUDA 12.1 wheels) ...
python -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
  echo CUDA wheel install failed; falling back to CPU PyTorch from PyPI ...
  python -m pip install "torch>=2.0.0" "torchvision>=0.15.0"
)

echo Installing remaining dependencies ...
python -m pip install timm numpy pandas scikit-learn opencv-python matplotlib Pillow PyYAML tqdm tabulate
if errorlevel 1 goto :fail

echo.
echo Setup complete. Activate with:
echo   .venv\Scripts\activate.bat
echo.
exit /b 0

:fail
echo Setup failed.
exit /b 1
