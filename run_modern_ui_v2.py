import uvicorn
import webbrowser
from threading import Timer

def open_browser():
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    print("Starting Modern UI v2 (Real-Time WebSocket)...")
    Timer(1.5, open_browser).start()
    uvicorn.run("modern_ui_v2.backend.main:app", host="127.0.0.1", port=8000, reload=True)
