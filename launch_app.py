"""Simple launcher script for the Streamlit app."""

import subprocess
import sys
import webbrowser
import time
import threading
from pathlib import Path

def open_browser():
    """Open browser after server starts."""
    time.sleep(2)
    webbrowser.open("http://localhost:8501")

def main():
    """Launch the Streamlit app."""
    app_path = Path(__file__).parent / "app.py"
    
    if not app_path.exists():
        print(f"Error: Streamlit app not found at {app_path}")
        sys.exit(1)
    
    print("🏀 Launching NBA Player Similarity App...")
    print("The app will open automatically in your browser...")
    
    # Start browser opening in background
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Run Streamlit
    try:
        subprocess.run([
            sys.executable, 
            "-m", 
            "streamlit", 
            "run", 
            str(app_path),
            "--server.headless", 
            "true"
        ])
    except KeyboardInterrupt:
        print("\n\nApp stopped by user.")

if __name__ == "__main__":
    main()

