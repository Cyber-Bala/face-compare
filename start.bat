@echo off
setlocal
cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
    echo Creating virtual environment...
    python -m venv venv || goto :fail
)

echo Checking dependencies...
venv\Scripts\python.exe -m pip install -q --upgrade pip
venv\Scripts\python.exe -m pip install -q -r requirements.txt || goto :fail

if not exist models mkdir models
if not exist models\face_detection_yunet_2023mar.onnx (
    echo Downloading YuNet...
    curl.exe -L -s -o models\face_detection_yunet_2023mar.onnx "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" || goto :fail
)
if not exist models\adaface_ir101.onnx (
    echo Downloading AdaFace IR-101 ^(about 260 MB^)...
    curl.exe -L -s -o models\adaface_ir101.onnx "https://huggingface.co/ludovitkramar/adaface-ir101-onnx/resolve/main/adaface_ir101.onnx" || goto :fail
    curl.exe -L -s -o models\adaface_ir101.onnx.data "https://huggingface.co/ludovitkramar/adaface-ir101-onnx/resolve/main/adaface_ir101.onnx.data" || goto :fail
)

echo Starting portal. Press Ctrl+C to stop.
venv\Scripts\python.exe run.py
pause
exit /b 0

:fail
echo.
echo Setup failed. See the message above.
pause
exit /b 1
