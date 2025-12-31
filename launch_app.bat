@echo off
REM Simple launcher for Windows

echo 🏀 Launching NBA Player Similarity App...
echo.

REM Check if virtual environment exists and activate it
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Launch the app
python launch_app.py

