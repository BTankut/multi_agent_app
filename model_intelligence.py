import json
import logging
import time
import os
from datetime import datetime
from pathlib import Path
from utils import get_openrouter_models, call_agent

# Configure logger
logger = logging.getLogger("model_intelligence")
logger.setLevel(logging.INFO)

# Constants
DATA_DIR = Path("data")
INTELLIGENCE_FILE = DATA_DIR / "model_intelligence.json"
LABELS_FILE = DATA_DIR / "model_labels.json"
ROLES_FILE = DATA_DIR / "model_roles.json"

# Default analyst model - fast, capable, and up-to-date
DEFAULT_ANALYST_MODEL = "x-ai/grok-4.1-fast:free"  # Using free tier if available, or paid if configured

def analyze_models(analyst_model=None, batch_size=20):
    """
    Analyzes all available OpenRouter models using an AI analyst to determine
    their capabilities, tiers, and optimal roles.
    Supports incremental saving and resuming.
    """
    if not analyst_model:
        analyst_model = DEFAULT_ANALYST_MODEL
        
    logger.info(f"Starting model analysis using {analyst_model}...")
    
    # 1. Fetch all models
    models = get_openrouter_models()
    if not models:
        logger.error("Failed to fetch models from OpenRouter.")
        return {"success": False, "error": "Could not fetch models"}
    
    total_models = len(models)
    logger.info(f"Fetched {total_models} total models available.")
    
    # 2. Load existing intelligence to resume progress
    analyzed_models_map = {}
    if INTELLIGENCE_FILE.exists():
        try:
            with open(INTELLIGENCE_FILE, 'r') as f:
                full_data = json.load(f)
                # Create a map of id -> model_data for easy lookup
                for m in full_data.get("models", []):
                    # Only verify if it has essential fields to consider it "analyzed"
                    if "tier" in m and "capabilities" in m:
                        analyzed_models_map[m["id"]] = m
            logger.info(f"Loaded {len(analyzed_models_map)} previously analyzed models.")
        except Exception as e:
            logger.warning(f"Could not load existing intelligence: {e}")

    # Filter out models that are already analyzed
    models_to_analyze = [m for m in models if m["id"] not in analyzed_models_map]
    logger.info(f"Models remaining to analyze: {len(models_to_analyze)}")
    
    if not models_to_analyze:
        logger.info("All models are already analyzed. Updating system files only.")
        update_system_files(list(analyzed_models_map.values()))
        return {"success": True, "count": 0, "message": "All models were already analyzed."}

    # 3. Process in batches
    # Create batches from the remaining models
    batches = [models_to_analyze[i:i + batch_size] for i in range(0, len(models_to_analyze), batch_size)]
    
    # We start with the already analyzed ones
    current_results = list(analyzed_models_map.values())
    
    for i, batch in enumerate(batches):
        logger.info(f"Processing batch {i+1}/{len(batches)} ({len(batch)} models)...")
        
        # Prepare simplified model list for the prompt to save tokens
        batch_input = []
        for m in batch:
            batch_input.append({
                "id": m["id"],
                "name": m["name"],
                "description": m.get("description", ""),
                "context_length": m.get("context_length", 0)
            })
            
        user_prompt = json.dumps(batch_input, indent=2)
        
        try:
            # Call the analyst agent
            response_tuple = call_agent(analyst_model, "You are an expert AI Model Analyst.", user_prompt, models, reasoning_mode="disabled")
            
            # Handle response tuple
            if isinstance(response_tuple, tuple):
                response_text = response_tuple[0]
            else:
                response_text = response_tuple
                
            # Clean response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
                
            # Parse JSON
            batch_results = json.loads(response_text)
            
            # Validate and merge
            if isinstance(batch_results, list):
                # Update our main list
                current_results.extend(batch_results)
                logger.info(f"Successfully analyzed {len(batch_results)} models in batch {i+1}")
                
                # INCREMENTAL SAVE: Save after every successful batch
                save_intelligence_data(current_results, analyst_model)
                # Also update system files incrementally so user sees progress immediately
                update_system_files(current_results)
                
            else:
                logger.error(f"Invalid JSON format in batch {i+1}: Expected list")
                
            # Rate limit pause
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Error analyzing batch {i+1}: {e}")
            # We don't stop the process, but we log the error. 
            # The unanalyzed models in this batch won't be in 'current_results', 
            # so they will be picked up again next time the script runs.

    return {"success": True, "count": len(current_results)}

def save_intelligence_data(models_data, analyst_model):
    """Helper function to save intelligence data to JSON."""
    result_data = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "analyst_model": analyst_model,
        "models": models_data
    }
    
    try:
        with open(INTELLIGENCE_FILE, 'w') as f:
            json.dump(result_data, f, indent=2)
        logger.info(f"Incremental save: {len(models_data)} models saved to {INTELLIGENCE_FILE}")
    except Exception as e:
        logger.error(f"Failed to save intelligence file: {e}")

def update_system_files(analyzed_models):
    """
    Updates model_labels.json and model_roles.json based on the new analysis.
    """
    logger.info("Updating system configuration files...")
    
    # 1. Update model_labels.json
    labels_data = []
    for m in analyzed_models:
        # Skip image generation models for text labels (or give them a specific tag)
        if m.get("is_image_generation", False):
            continue
            
        labels = []
        # Map capabilities to our system labels
        caps = m.get("capabilities", [])
        if "code" in caps: labels.append("code_expert")
        if "math" in caps: labels.append("math_expert")
        if "reasoning" in caps: labels.append("reasoning_expert")
        if "creative_writing" in caps: labels.append("creative_writer")
        if "vision" in caps: labels.append("vision_expert")
        
        # Always add general assistant if no specific experts, or as a base
        if not labels or "general" in caps:
            labels.append("general_assistant")
            
        # Add Tier label (optional but useful)
        # labels.append(f"tier_{m.get('tier', 3)}")
        
        labels_data.append({
            "model": m["id"],
            "labels": labels
        })
        
    try:
        with open(LABELS_FILE, 'w') as f:
            json.dump(labels_data, f, indent=2)
        logger.info(f"Updated {LABELS_FILE} with {len(labels_data)} models")
    except Exception as e:
        logger.error(f"Failed to update labels file: {e}")

    # 2. Update model_roles.json
    # We need to preserve the structure of roles file: {"labels": [...], "roles": [...]}
    # But actually, we might just want to update a new "model_specific_roles.json" or 
    # inject into the existing one if your system supports per-model roles.
    # Currently agents.py uses 'model_roles.json' which maps LABELS to ROLES, not MODELS to ROLES.
    # However, get_model_roles in agents.py assigns roles based on labels.
    
    # OPTION: We can create a new mapping file "model_specific_roles.json" 
    # and update agents.py to check that first.
    
    specific_roles = {}
    for m in analyzed_models:
        if not m.get("is_image_generation", False):
            specific_roles[m["id"]] = m.get("recommended_role", "You are a helpful assistant.")
            
    specific_roles_file = DATA_DIR / "model_specific_roles.json"
    try:
        with open(specific_roles_file, 'w') as f:
            json.dump(specific_roles, f, indent=2)
        logger.info(f"Created/Updated {specific_roles_file}")
    except Exception as e:
        logger.error(f"Failed to save specific roles: {e}")
        
    return True

if __name__ == "__main__":
    # For CLI testing
    analyze_models()
