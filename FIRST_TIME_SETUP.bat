@echo off
echo ================================================================================
echo   ELERYC EXPERIMENT VIEWER - FIRST TIME SETUP
echo ================================================================================
echo.
echo This will:
echo   1. Install required Python packages (Flask, Pandas, Matplotlib)
echo   2. Start the automated comparison server
echo.
echo This only needs to be run ONCE (or if packages are missing)
echo ================================================================================
echo.

echo [1/2] Installing Python packages...
echo.
pip install flask flask-cors pandas matplotlib openpyxl
echo.

if %errorlevel% neq 0 (
    echo.
    echo ================================================================================
    echo   ERROR: Package installation failed!
    echo ================================================================================
    echo.
    echo Please check:
    echo   - Python is installed and in PATH
    echo   - pip is working correctly
    echo   - You have internet connection
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo   ✅ Packages installed successfully!
echo ================================================================================
echo.
echo [2/2] Starting Flask backend server...
echo.
echo Server will run on: http://localhost:5000
echo.
echo ================================================================================
echo   🚀 READY! Open your browser to http://localhost:5000
echo ================================================================================
echo.

python app.py

pause

