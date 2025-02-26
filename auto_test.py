import os
import sys
import json
from dotenv import load_dotenv
import requests
import random

# Load environment variables
load_dotenv()

# API key
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("ERROR: OPENROUTER_API_KEY not found. Please create a .env file with your API key.")
    sys.exit(1)

# Model categories
FREE_MODELS = [
    "mistralai/mistral-small-24b-instruct-2501:free",
    "google/gemini-2.0-pro-exp-02-05:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "microsoft/phi-3-medium-128k-instruct:free"
]

PAID_MODELS = [
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-4o",
    "mistralai/mistral-large",
    "google/gemini-pro-1.5"
]

# Constants
API_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

def call_model(model_id, system_prompt, user_prompt):
    """Call a model via OpenRouter API"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }
    
    try:
        response = requests.post(API_ENDPOINT, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"Error with model {model_id}: {str(e)}"

def analyze_query(query):
    """Analyze query to determine appropriate labels"""
    labels = ["general_assistant"]  # Default label
    
    # Simple keyword matching for labels
    if any(kw in query.lower() for kw in ["code", "programming", "function", "class", "algorithm"]):
        labels.append("code_expert")
    
    if any(kw in query.lower() for kw in ["math", "calculate", "equation", "formula", "number"]):
        labels.append("math_expert")
    
    if any(kw in query.lower() for kw in ["reason", "logic", "analyze", "think", "evaluate"]):
        labels.append("reasoning_expert")
    
    if any(kw in query.lower() for kw in ["creative", "story", "poem", "write", "fiction"]):
        labels.append("creative_writer")
    
    return labels

def select_models(labels, option="free", max_models=2):
    """Select models based on labels and user preference"""
    if option == "free":
        return random.sample(FREE_MODELS, min(max_models, len(FREE_MODELS)))
    elif option == "paid":
        return random.sample(PAID_MODELS, min(max_models, len(PAID_MODELS)))
    else:  # Mixed
        models = []
        models.extend(random.sample(FREE_MODELS, min(1, len(FREE_MODELS))))
        models.extend(random.sample(PAID_MODELS, min(1, len(PAID_MODELS))))
        return models[:max_models]

def get_system_prompt(label):
    """Get appropriate system prompt based on label"""
    prompts = {
        "general_assistant": "You are a helpful, accurate, and friendly assistant. Provide thorough and thoughtful responses to the user's queries.",
        "code_expert": "You are a software development expert. When writing code, include detailed comments, proper error handling, and follow best practices. For debugging, analyze the problem step by step and suggest specific fixes.",
        "math_expert": "You are a mathematics expert. Show your work step by step, explain mathematical concepts clearly, and verify your calculations. Use proper notation and be precise in your explanations.",
        "reasoning_expert": "You are a critical thinking and reasoning expert. Analyze problems from multiple perspectives, identify assumptions, evaluate evidence, and consider alternatives before reaching conclusions.",
        "creative_writer": "You are a creative writer. Your role is to generate original and imaginative content, including stories, poems, scripts, articles, and other forms of written expression."
    }
    
    return prompts.get(label, prompts["general_assistant"])

def process_query(query, option):
    """Process a user query using multiple models"""
    # Step 1: Analyze query
    print("Analyzing query...")
    labels = analyze_query(query)
    print(f"Identified labels: {', '.join(labels)}")
    
    # Step 2: Select models
    models = select_models(labels, option)
    print(f"Selected models: {', '.join(models)}")
    
    # Step 3: Call models with appropriate system prompts
    print("Calling models...")
    responses = {}
    
    # Use the most specific label for each model
    for model in models:
        for label in labels:
            if label != "general_assistant":  # Use more specific label if available
                system_prompt = get_system_prompt(label)
                break
        else:
            system_prompt = get_system_prompt("general_assistant")
        
        print(f"- Calling {model}...")
        response = call_model(model, system_prompt, query)
        responses[model] = response
    
    # Step 4: Combine responses (simplified version)
    if len(responses) == 1:
        # If only one model responded, return its response directly
        return list(responses.values())[0]
    else:
        # If multiple models responded, return their responses with attribution
        combined = "Responses from multiple models:\n\n"
        for model, response in responses.items():
            combined += f"--- {model} ---\n{response}\n\n"
        return combined

def run_auto_test():
    """Run a test with a predefined query"""
    print("Multi-Agent CLI Auto Test")
    print("========================")
    
    # Set test parameters
    option = "free"
    print(f"Using {option} models.")
    
    # Predefined test queries - you can change these
    test_queries = [
        "Write a Python function to calculate the Fibonacci sequence",
        "What are the key differences between machine learning and deep learning?",
        "Explain how quantum computing works"
    ]
    
    # Process each test query
    for i, query in enumerate(test_queries):
        print(f"\nTest Query #{i+1}: {query}")
        print("\nProcessing query...")
        response = process_query(query, option)
        
        print("\nResponse:")
        print("=========")
        print(response)
        print("\n" + "="*50)

if __name__ == "__main__":
    run_auto_test()