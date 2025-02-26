import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# API key
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("ERROR: OPENROUTER_API_KEY not found")
    exit(1)

def call_model(model_id, prompt):
    """
    Call a model via OpenRouter API
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data
    )
    
    if response.status_code != 200:
        return f"Error: {response.status_code} - {response.text}"
    
    result = response.json()
    return result['choices'][0]['message']['content']

# Test with a simple query
model_id = "mistralai/mistral-small-24b-instruct-2501:free"  # Free model for testing
query = "What's the difference between a function and a method in programming?"

print(f"Testing with model: {model_id}")
print("Query:", query)
print("\nResponse:")
print("---------")
print(call_model(model_id, query))