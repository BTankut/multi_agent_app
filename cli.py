import json
import sys
from coordinator import process_query
from utils import get_openrouter_models

def main():
    """
    Command-line interface for the Multi-Agent System
    """
    print("Multi-Agent System CLI")
    print("======================")
    
    # Load OpenRouter models
    print("Loading models from OpenRouter...")
    openrouter_models = get_openrouter_models()
    if not openrouter_models:
        print("No models found or API key missing. Please check your .env file.")
        sys.exit(1)
    
    print(f"Found {len(openrouter_models)} models.")
    
    # Set defaults
    coordinator_model = "anthropic/claude-3.5-haiku"
    option = "free"  # Can be "free", "paid", or "optimized"
    
    # Get user query
    print("\nEnter your query (type 'exit' to quit):")
    query = input("> ")
    
    while query.lower() != "exit":
        print("\nProcessing query...")
        print("Step 1: Analyzing query...")
        print("Step 2: Selecting models...")
        print("Step 3: Processing with models...")
        
        # Process the query
        result, labels = process_query(query, coordinator_model, option, openrouter_models)
        
        print("\nSelected labels:", ", ".join(labels))
        print("\nResponse:")
        print("=========")
        print(result)
        
        # Get next query
        print("\nEnter your query (type 'exit' to quit):")
        query = input("> ")

if __name__ == "__main__":
    main()