import uvicorn
import os
import sys
import webbrowser
from threading import Timer

def open_browser():
    """Open the browser after a short delay to ensure server is running."""
    webbrowser.open("http://127.0.0.1:8000")

def main():
    print("="*50)
    print("Starting Modern Multi-Agent Interface")
    print("="*50)
    print("\nServer starting at http://127.0.0.1:8000")
    print("Press Ctrl+C to stop the server.\n")

    # Schedule browser open
    Timer(1.5, open_browser).start()

    # Run Uvicorn
    # reload=True allows for auto-reload on code changes during development
    uvicorn.run("modern_ui.main:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()

