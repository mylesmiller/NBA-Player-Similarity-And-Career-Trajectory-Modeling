#!/bin/bash
# Simple launcher for Linux/Mac

echo "🏀 Launching NBA Player Similarity App..."
echo ""

# Check if virtual environment exists and activate it
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Launch the app
python launch_app.py

