@echo off
setlocal EnableDelayedExpansion

echo ===========================================
echo Wodle TPR - Windows Setup
echo ===========================================

REM 1. Check Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found! Please install Python 3.10+ and add to PATH.
    pause
    exit /b 1
)
echo [OK] Python found

REM 2. Create logs directory locally (to avoid /var/ossec issues)
if not exist "logs" (
    mkdir logs
    echo [OK] Created local 'logs' directory
)

REM 3. Install dependencies
if exist "requirements.txt" (
    echo [INFO] Installing dependencies...
    pip install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed
) else (
    echo [WARNING] requirements.txt not found. Skipping pip install.
)

REM 4. Setup .env file
if not exist ".env" (
    echo [INFO] Creating .env from .env.example...
    copy .env.example .env >nul
    
    REM Append Windows-specific log paths to .env
    echo. >> .env
    echo # Windows Local Overrides >> .env
    echo LOG_FILE=./logs/anomaly_detection.log >> .env
    echo ANOMALY_LOG_FILE=./logs/anomaly_alerts.log >> .env
    
    echo [OK] Created .env file with local log paths.
    echo [ACTION REQUIRED] Please edit .env with your OpenSearch credentials.
) else (
    echo [INFO] .env already exists. Skipping creation.
    echo [TIP] Ensure LOG_FILE and ANOMALY_LOG_FILE are set in .env to avoid errors.
)

echo.
echo ===========================================
echo Setup Complete!
echo ===========================================
echo To run the detection:
echo   call run.bat
echo.
pause
