@echo off
echo Installing dependencies...
python -m pip install --upgrade -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo.
echo Starting Flask web app...
echo.
echo Open your browser and go to: http://localhost:5000
echo Press Ctrl+C to stop the app
echo.
python app.py
pause
