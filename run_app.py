import subprocess
import os
import sys

def main():
    """Launch the Multi_Agent Streamlit app"""
    print("Launching Multi_Agent Streamlit App...")
    
    # Check if Streamlit is installed
    try:
        import streamlit
        print(f"Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("Streamlit is not installed. Installing required dependencies...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
            print("Dependencies installed successfully.")
        except subprocess.CalledProcessError:
            print("Failed to install dependencies. Please run 'pip install -r requirements.txt' manually.")
            return
    
    # Check if the OpenRouter API key is set
    if not os.environ.get("OPENROUTER_API_KEY") and not os.path.exists(".env"):
        print("WARNING: OpenRouter API key not found. The app may not function correctly.")
        print("Please create a .env file with your OpenRouter API key:")
        print("OPENROUTER_API_KEY=your_api_key_here")
        answer = input("Do you want to continue anyway? (y/n): ")
        if answer.lower() != "y":
            return
    
    # Launch the Streamlit app
    print("Starting Streamlit app...")
    print("\nTo access the web interface, open the URL displayed in your browser.")
    print("Press Ctrl+C to stop the app.")
    
    subprocess.run(["streamlit", "run", "app.py"])

if __name__ == "__main__":
    main()