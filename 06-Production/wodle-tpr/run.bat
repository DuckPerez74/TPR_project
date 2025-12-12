@echo off
echo Running Wodle TPR...
python main.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Wodle execution failed.
    pause
    exit /b 1
)
