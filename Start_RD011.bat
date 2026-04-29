@echo off
TITLE RD.011 Generator Launcher
COLOR 0B

echo ===================================================
echo     RD.011 Future Process Model Generator
echo ===================================================
echo.
echo Starting application... Please wait.
echo.

:: Check if Python is available
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ and try again, or use the Portable Python version.
    pause
    exit /b 1
)

:: Run streamlit
echo Launching Streamlit interface...
streamlit run app.py --server.headless true

pause
