import sys
import logging
from utils import get_openrouter_models
from coordinator import process_query

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test the Multi_Agent system without Streamlit UI"""
    print("Testing Multi_Agent System...")
    print("Fetching models from OpenRouter...")
    
    # Get models from OpenRouter
    openrouter_models = get_openrouter_models()
    if not openrouter_models:
        print("Failed to fetch models from OpenRouter. Check your API key.")
        return
        
    print(f"Found {len(openrouter_models)} models")
    
    # Process a sample query
    sample_query = "Explain the difference between a binary tree and a binary search tree in programming."
    coordinator_model = "anthropic/claude-3.5-haiku"
    option = "free"  # Use free models
    
    print(f"\nProcessing query: {sample_query}")
    print(f"Coordinator model: {coordinator_model}")
    print(f"Option: {option}")
    
    # Process the query
    try:
        final_answer, labels = process_query(
            sample_query, 
            coordinator_model, 
            option, 
            openrouter_models
        )
        
        print(f"\nIdentified labels: {', '.join(labels)}")
        print("\nFinal answer:")
        print("-" * 50)
        print(final_answer)
        print("-" * 50)
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return

if __name__ == "__main__":
    main()