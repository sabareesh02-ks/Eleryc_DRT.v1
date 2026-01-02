@echo off
echo ================================================================================
echo   ELERYC EXPERIMENT VIEWER - QUICK RESTART
echo ================================================================================
echo.

REM Change to the UI directory (this script should be in the UI folder)
cd /d "%~dp0"

echo Checking if port 5000 is in use...
netstat -ano | findstr :5000 >nul
if %errorlevel% == 0 (
    echo.
    echo WARNING: Port 5000 is already in use!
    echo Please stop any existing server first (Ctrl+C in that window)
    echo.
    pause
    exit /b
)

echo Port 5000 is available.
echo.
echo Starting Flask server...
echo.
echo Server will be available at: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo ================================================================================
echo.

python app.py

pause

