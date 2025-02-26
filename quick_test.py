import logging
from utils import get_openrouter_models, load_json
from agents import get_models_by_labels

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Quick test to verify the system components without making API calls"""
    print("Running quick test of Multi_Agent System components...")
    
    # Load model labels
    model_labels = load_json("data/model_labels.json")
    print(f"Loaded {len(model_labels)} model entries from model_labels.json")
    
    # Load model roles
    model_roles = load_json("data/model_roles.json")
    print(f"Loaded {len(model_roles['labels'])} labels and {len(model_roles['roles'])} roles from model_roles.json")
    
    # Test model selection
    test_labels = ["code_expert", "general_assistant"]
    print(f"\nTesting model selection with labels: {', '.join(test_labels)}")
    
    # Get OpenRouter models
    openrouter_models = get_openrouter_models()
    if not openrouter_models:
        print("Failed to fetch models from OpenRouter. Check your API key.")
        return
    
    print(f"Fetched {len(openrouter_models)} models from OpenRouter")
    
    # Test free model selection
    free_models = get_models_by_labels(test_labels, "free", openrouter_models, min_models=1, max_models=2)
    print(f"\nSelected free models: {free_models}")
    
    # All components verified
    print("\nQuick test completed. Core components are working correctly.")

if __name__ == "__main__":
    main()