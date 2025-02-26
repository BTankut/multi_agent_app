import json
import os
from dotenv import load_dotenv
import requests

# Load the environment variables
load_dotenv()

# Get the API key
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("ERROR: OPENROUTER_API_KEY not found in environment variables.")
    exit(1)

print(f"API key loaded: {api_key[:5]}...{api_key[-3:]}")

# Test API by getting available models
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

try:
    response = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers=headers,
        timeout=10
    )
    
    response.raise_for_status()
    models = response.json().get('data', [])
    
    print(f"Available models ({len(models)}):")
    print("---------------------------")
    
    # Print model IDs in alphabetical order
    for model in sorted(models, key=lambda x: x.get('id', '').lower()):
        model_id = model.get('id', 'Unknown')
        print(f"- {model_id}")
    
except requests.exceptions.RequestException as e:
    print(f"Error with API request: {str(e)}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"Response text: {e.response.text}")
except Exception as e:
    print(f"Unexpected error: {str(e)}")