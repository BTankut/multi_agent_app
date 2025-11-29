#!/usr/bin/env python3
"""
Model Labels Updater
Updates model_labels.json by fetching the latest model list from OpenRouter API.
Preserves existing labels and applies smart labeling for new models.
"""

import os
import json
import re
import requests
import logging
import time
from dotenv import load_dotenv

# Logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Constants
MODEL_LABELS_FILE = "data/model_labels.json"
MODEL_ROLES_FILE = "data/model_roles.json"
BACKUP_DIR = "data/backups"

def ensure_backup_dir():
    """Ensures backup directory exists"""
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
        logger.info(f"Backup directory created: {BACKUP_DIR}")

def load_json(file_path):
    """Loads JSON file"""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        return None

def save_json(file_path, data):
    """Saves JSON data to file"""
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=2)
        logger.info(f"File saved successfully: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        return False

def backup_file(file_path):
    """Creates a timestamped backup of a file and cleans old backups"""
    ensure_backup_dir()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.basename(file_path)
    backup_path = f"{BACKUP_DIR}/{filename}-{timestamp}.bak"
    
    try:
        data = load_json(file_path)
        if data:
            save_json(backup_path, data)
            logger.info(f"Backup created: {backup_path}")
            
            # Clean old backups - keep only last 5 for each file type
            clean_old_backups(filename)
            return True
    except Exception as e:
        logger.error(f"Error during backup: {e}")
    
    return False

def clean_old_backups(filename_prefix):
    """Cleans old backups of a specific type, keeping only the last 5"""
    try:
        # Find related backup files
        backup_files = []
        for file in os.listdir(BACKUP_DIR):
            if file.startswith(filename_prefix) and file.endswith(".bak"):
                file_path = os.path.join(BACKUP_DIR, file)
                backup_files.append((file_path, os.path.getmtime(file_path)))
        
        # Sort by modification time (newest last)
        backup_files.sort(key=lambda x: x[1])
        
        # Delete older than last 5
        if len(backup_files) > 5:
            for file_path, _ in backup_files[:-5]:
                os.remove(file_path)
                logger.info(f"Old backup cleaned: {file_path}")
            
            logger.info(f"Cleaned {len(backup_files)-5} old backups, kept {min(5, len(backup_files))} backups")
    except Exception as e:
        logger.error(f"Error cleaning old backups: {e}")

def get_openrouter_models(min_date=None, sort_by_newest=True):
    """
    Fetches available models from OpenRouter and optionally filters them.
    
    Args:
        min_date: Optional date string in format 'YYYY-MM-DD' to filter models created after this date
        sort_by_newest: Whether to sort models by newest first (default: True)
        
    Returns:
        List of model data dictionaries
    """
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY not found!")
        return []
    
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=15
        )
        
        response.raise_for_status()
        models_data = response.json().get('data', [])
        
        logger.info(f"Retrieved {len(models_data)} models from OpenRouter")
        
        # Filter by date if min_date is provided
        if min_date:
            try:
                import datetime
                min_datetime = datetime.datetime.strptime(min_date, '%Y-%m-%d')
                filtered_models = []
                
                for model in models_data:
                    # Check if created_at exists
                    if 'created_at' in model:
                        try:
                            # Convert to datetime object (assuming ISO format)
                            model_date = datetime.datetime.fromisoformat(model['created_at'].replace('Z', '+00:00'))
                            if model_date >= min_datetime:
                                filtered_models.append(model)
                        except (ValueError, TypeError):
                            # If date conversion fails, keep the model (be inclusive)
                            filtered_models.append(model)
                    else:
                        # If no date information, keep the model (be inclusive)
                        filtered_models.append(model)
                
                logger.info(f"{len(filtered_models)} models remaining after date filter (after {min_date})")
                models_data = filtered_models
            except Exception as filter_error:
                logger.warning(f"Date filtering error: {str(filter_error)}")
        
        # Sort by newest first if requested
        if sort_by_newest and models_data:
            # Check if models have created_at field
            has_created_at = False
            for model in models_data:
                if 'created_at' in model:
                    has_created_at = True
                    break
                
            if has_created_at:
                try:
                    # Sort by created_at field (newest first)
                    models_data.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                    logger.info("Models sorted by creation date (newest first)")
                except Exception as sort_error:
                    logger.warning(f"Error sorting models by date: {str(sort_error)}")
        
        return models_data
        
    except Exception as e:
        logger.error(f"Error fetching model list from OpenRouter: {e}")
        return []

def get_available_labels():
    """Loads available labels from model_roles.json"""
    roles_data = load_json(MODEL_ROLES_FILE)
    if not roles_data or "labels" not in roles_data:
        logger.error("Could not load model roles or 'labels' key not found")
        return []
    
    return [label_entry['label'] for label_entry in roles_data['labels']]

def get_existing_model_labels():
    """Loads model-label mappings from existing model_labels.json"""
    model_data = load_json(MODEL_LABELS_FILE)
    if not model_data:
        logger.error("Could not load model labels")
        return {}
    
    # Return as dictionary of model -> labels
    return {entry.get("model", ""): entry.get("labels", []) for entry in model_data if "model" in entry}

def determine_labels_for_model(model_info, available_labels, existing_labels):
    """
    Determines labels for a model. Uses existing labels first, 
    applies smart labeling for new models.
    
    IMPORTANT: Ensures all labels are defined in model_roles.json.
    New: Gathers extra info from model's OpenRouter page.
    """
    model_id = model_info.get('id', '')
    
    # If model is already labeled, use existing labels and validate
    if model_id in existing_labels:
        # Filter existing labels with available_labels - clean up labels not in model_roles.json
        valid_labels = [label for label in existing_labels[model_id] if label in available_labels]
        
        # If no valid labels remain, create new ones
        if not valid_labels:
            logger.warning(f"No valid labels found for model {model_id}, re-labeling.")
        else:
            # If at least one valid label exists, use it
            return valid_labels
    
    # Determine labels for new model
    labels = []
    
    # Check and add "general_assistant" label
    if "general_assistant" in available_labels:  # Every model is marked at least as general assistant
        labels.append("general_assistant")
    
    # Label based on pricing
    try:
        prompt_price = model_info.get('pricing', {}).get('prompt', '0')
        # Convert to float
        prompt_price = float(prompt_price) if prompt_price else 0
        
        if prompt_price > 0 and "paid" in available_labels:
            labels.append("paid") 
        elif "free" in available_labels:
            labels.append("free")
    except (ValueError, TypeError):
        # Add free label by default on error (if available)
        if "free" in available_labels:
            labels.append("free")
    
    # New: Fetch model info from OpenRouter page
    try:
        from utils import get_model_description
        model_details = get_model_description(model_id)
        
        if model_details and model_details.get("success", False):
            logger.info(f"Additional info found for model {model_id}: {model_details.get('description', '')}")
            
            # Extract labels from description
            description = model_details.get("description", "").lower()
            
            # Reasoning
            if "reasoning" in description:
                if "reasoning_expert" in available_labels:
                    labels.append("reasoning_expert")
                    
            # Math
            if "math" in description:
                if "math_expert" in available_labels:
                    labels.append("math_expert")
                    
            # Code
            if any(term in description for term in ["code", "coding", "programming"]):
                if "code_expert" in available_labels:
                    labels.append("code_expert")
                    
            # Vision
            if any(term in description for term in ["vision", "image", "visual"]):
                if "vision_expert" in available_labels:
                    labels.append("vision_expert")
    except Exception as e:
        logger.warning(f"Error fetching additional info for model {model_id}: {str(e)}")
    
    # Auto-labeling based on Model ID
    model_id_lower = model_id.lower()
    model_name = model_info.get('name', '').lower()
    
    # Code/Coding expert models
    if any(term in model_id_lower or term in model_name for term in ["code", "coding", "coder", "codestral", "phi-3", "o3"]) or "claude" in model_id_lower:
        if "code_expert" in available_labels:
            labels.append("code_expert")
    
    # Math expert models
    if any(term in model_id_lower or term in model_name for term in ["math", "numeric", "gemini", "gpt-4", "claude", "o1", "mixtral", "o3"]):
        if "math_expert" in available_labels:
            labels.append("math_expert")
    
    # Vision models
    if any(term in model_id_lower or term in model_name for term in ["vision", "vl", "image", "pixtral", "visual"]):
        if "vision_expert" in available_labels:
            labels.append("vision_expert")
    
    # Experimental models
    if any(term in model_id_lower for term in ["exp", "experimental", "beta", "preview"]):
        if "experimental" in available_labels:
            labels.append("experimental")
    
    # Reasoning experts
    if any(term in model_id_lower or term in model_name for term in ["reasoning", "gemini", "claude", "gpt-4", "o1", "mixtral"]):
        if "reasoning_expert" in available_labels:
            labels.append("reasoning_expert")
    
    # Fast response models
    if any(term in model_id_lower for term in ["flash", "haiku", "mini", "small", "fast"]):
        if "fast_response" in available_labels:
            labels.append("fast_response")
    
    # Instruction following
    if "instruct" in model_id_lower:
        if "instruction_following" in available_labels:
            labels.append("instruction_following")
    
    # Multilingual models
    if any(term in model_id_lower for term in ["multilingual", "multi-lingual"]):
        if "multilingual" in available_labels:
            labels.append("multilingual")
    
    # Conversational models
    if any(term in model_id_lower for term in ["chat", "convers", "talk"]):
        if "conversationalist" in available_labels:
            labels.append("conversationalist")
    
    # Creative writer models
    if any(term in model_id_lower for term in ["creative", "writer", "narrat"]):
        if "creative_writer" in available_labels:
            labels.append("creative_writer")
    
    # Label based on model size
    size_match = re.search(r'(\d+)[bB]', model_id)
    if size_match:
        size = int(size_match.group(1))
        if size >= 70:  # Large models are generally better at reasoning
            if "reasoning_expert" in available_labels and "reasoning_expert" not in labels:
                labels.append("reasoning_expert")
    
    # If no labels determined, add general_assistant by default
    if not labels and "general_assistant" in available_labels:
        labels.append("general_assistant")
    
    # Make labels unique and keep only valid ones
    valid_labels = list(set([label for label in labels if label in available_labels]))
    
    # If no valid labels remain, add "general_assistant" (if defined)
    if not valid_labels and "general_assistant" in available_labels:
        valid_labels.append("general_assistant")
    
    return valid_labels

def update_model_labels(min_date=None):
    """
    Fetches model list from OpenRouter API and updates model_labels.json.
    Preserves existing labels and auto-labels new models.
    IMPORTANT: Verifies all labels are defined in model_roles.json.
    
    Args:
        min_date: Optional date string in format 'YYYY-MM-DD' to filter models created after this date
    """
    # Backup existing files
    if not backup_file(MODEL_LABELS_FILE):
        # Continue even if backup fails (might happen in web interface)
        logger.warning("Backup failed, continuing...")
    
    # Load data sources
    openrouter_models = get_openrouter_models(min_date=min_date, sort_by_newest=True)
    available_labels = get_available_labels()
    existing_labels = get_existing_model_labels()
    
    if not openrouter_models:
        logger.error("Could not retrieve OpenRouter models, aborting")
        return False
    
    if not available_labels:
        logger.error("Could not load available labels, aborting")
        logger.error("Check model_roles.json! You might need to run update_model_roles.py first.")
        return False
    
    # Info: Show available labels from model_roles.json
    logger.info(f"{len(available_labels)} labels defined in model_roles.json: {', '.join(available_labels)}")
    
    # Check inconsistencies in existing labels
    invalid_labels_count = 0
    for model_id, labels in existing_labels.items():
        invalid_labels = [label for label in labels if label not in available_labels]
        if invalid_labels:
            invalid_labels_count += 1
            logger.warning(f"Model {model_id} contains undefined labels: {', '.join(invalid_labels)}")
    
    if invalid_labels_count > 0:
        logger.warning(f"Total {invalid_labels_count} models contain undefined labels. These labels will be cleaned.")
    
    # Create new model_labels.json data
    new_model_labels = []
    updated_count = 0
    new_count = 0
    cleaned_count = 0
    
    for model in openrouter_models:
        model_id = model.get('id', '')
        if not model_id:
            continue
        
        # Determine model labels - this function returns only valid labels
        old_labels = existing_labels.get(model_id, [])
        labels = determine_labels_for_model(model, available_labels, existing_labels)
        
        # Track new or updated models
        if model_id in existing_labels:
            old_valid_labels = [label for label in old_labels if label in available_labels]
            # If difference between old and new labels
            if set(labels) != set(old_valid_labels):
                # If old labels had undefined labels, they were cleaned
                if len(old_valid_labels) != len(old_labels):
                    cleaned_count += 1
                    logger.info(f"Undefined labels cleaned for model {model_id}: " +
                              f"Old: {old_labels}, New: {labels}")
                else:
                    updated_count += 1
        else:
            new_count += 1
            logger.info(f"New model added: {model_id} - Labels: {labels}")
        
        # Finally, ensure model has at least one label
        if not labels and "general_assistant" in available_labels:
            labels = ["general_assistant"]
            logger.warning(f"No labels could be determined for model {model_id}. 'general_assistant' added.")
        
        # Add model to new list
        new_model_labels.append({
            "model": model_id,
            "labels": labels
        })
    
    # Save changes
    if save_json(MODEL_LABELS_FILE, new_model_labels):
        logger.info(f"Model labels updated: {len(new_model_labels)} total models")
        logger.info(f"  - {new_count} new models added")
        logger.info(f"  - {updated_count} existing models updated")
        logger.info(f"  - {cleaned_count} models cleaned of undefined labels")
        
        # Final check - are all labels defined in model_roles.json?
        all_used_labels = set()
        for entry in new_model_labels:
            all_used_labels.update(entry.get("labels", []))
        
        missing_roles = [label for label in all_used_labels if label not in available_labels]
        if missing_roles:
            logger.error(f"ERROR: Some labels are still not defined in model_roles.json: {', '.join(missing_roles)}")
            logger.error("This should not be a logic error! Check your code.")
            return False
        else:
            logger.info("Consistency check passed: All labels are defined in model_roles.json.")
        
        return True
    else:
        logger.error("Error occurred while updating model labels")
        return False

if __name__ == "__main__":
    logger.info("Starting Model Labels Updater...")
    
    import argparse
    
    # Command line arguments
    parser = argparse.ArgumentParser(description="Update model labels from OpenRouter API")
    parser.add_argument("--min-date", 
                      help="Filter models created after this date (format: YYYY-MM-DD)",
                      default=None)
    
    args = parser.parse_args()
    
    if args.min_date:
        logger.info(f"Filtering models created after {args.min_date}")
    
    update_model_labels(min_date=args.min_date)