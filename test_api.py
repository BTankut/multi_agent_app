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
    
    print(f"Successfully retrieved {len(models)} models from OpenRouter.")
    print("\nFirst 5 models:")
    for model in models[:5]:
        print(f"- {model.get('id', 'Unknown')}")
    
    # Test calling a model
    print("\nTesting model call with Claude...")
    model_id = "anthropic/claude-3.5-haiku"
    
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json={
            "model": model_id,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"}
            ]
        },
        timeout=30
    )
    
    response.raise_for_status()
    result = response.json()
    print("\nResponse from model:")
    print(result['choices'][0]['message']['content'])
    
    print("\nAPI test successful!")
    
except requests.exceptions.RequestException as e:
    print(f"Error with API request: {str(e)}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"Response text: {e.response.text}")
except Exception as e:
    print(f"Unexpected error: {str(e)}")