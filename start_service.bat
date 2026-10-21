@echo off
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
    echo Run start.bat once first to set up.
    pause
    exit /b 1
)
echo Face ML service for the gym backend: http://127.0.0.1:8001
venv\Scripts\python.exe run.py --http
pause
