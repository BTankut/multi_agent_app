#!/usr/bin/env python3
"""
Model Roles Updater
Reorganizes model_roles.json and ensures consistency with model_labels.json.
"""

import os
import json
import logging
import time
from dotenv import load_dotenv

# Logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def get_all_labels_from_model_labels():
    """Extracts all unique labels used in model_labels.json"""
    model_labels_data = load_json(MODEL_LABELS_FILE)
    if not model_labels_data:
        logger.error("Could not load model_labels.json!")
        return set()
    
    all_labels = set()
    for model_entry in model_labels_data:
        labels = model_entry.get("labels", [])
        all_labels.update(labels)
    
    return all_labels

def update_model_roles():
    """
    Reorganizes model_roles.json and ensures consistency with model_labels.json.
    Performs consistency checks to ensure all labels are both defined and used.
    """
    # Backup existing files
    if not backup_file(MODEL_ROLES_FILE):
        # Continue even if backup fails (might happen in web interface)
        logger.warning("Backup failed, continuing...")
    
    # Load data sources
    model_roles_data = load_json(MODEL_ROLES_FILE)
    if not model_roles_data:
        logger.error("Could not load model_roles.json!")
        return False
    
    # Get all used labels from model_labels.json
    all_used_labels = get_all_labels_from_model_labels()
    logger.info(f"{len(all_used_labels)} unique labels used in model_labels.json")
    
    # Extract existing label and role definitions
    existing_label_descriptions = {}
    existing_role_prompts = {}
    
    # Get label descriptions from "labels" section
    if "labels" in model_roles_data:
        for label_entry in model_roles_data["labels"]:
            label = label_entry.get("label", "")
            description = label_entry.get("description", "")
            if label:
                existing_label_descriptions[label] = description
    
    # Get role prompts from "roles" section
    if "roles" in model_roles_data:
        for role_entry in model_roles_data["roles"]:
            label = role_entry.get("label", "")
            prompt = role_entry.get("prompt", "")
            if label:
                existing_role_prompts[label] = prompt
    
    # Check for inconsistencies
    # 1. Labels with definitions but no role prompts
    labels_without_roles = [label for label in existing_label_descriptions if label not in existing_role_prompts]
    if labels_without_roles:
        logger.warning(f"These labels have definitions but no role prompt: {', '.join(labels_without_roles)}")
        
    # 2. Roles without label definitions
    roles_without_labels = [label for label in existing_role_prompts if label not in existing_label_descriptions]
    if roles_without_labels:
        logger.warning(f"These role prompts have no label definition: {', '.join(roles_without_labels)}")
    
    # 3. Used labels without definitions
    used_without_definition = [label for label in all_used_labels if label not in existing_label_descriptions]
    if used_without_definition:
        logger.warning(f"These labels are used but have no definition: {', '.join(used_without_definition)}")
    
    # Default descriptions for missing labels
    default_descriptions = {
        "general_assistant": "A general-purpose assistant that can help with a wide variety of tasks.",
        "reasoning_expert": "An expert in reasoning and logical thinking, particularly good at solving complex problems.",
        "instruction_following": "Specifically designed to accurately follow detailed instructions given by the user.",
        "safety_focused": "Prioritizes safe and ethical responses, avoiding harmful, unethical, or inappropriate content.",
        "math_expert": "Specialized in solving mathematical problems and calculations with high accuracy.",
        "code_expert": "Expert in programming, software development, and code-related tasks.",
        "vision_expert": "Capable of understanding and analyzing visual information from images.",
        "experimental": "A model with experimental features still under development.",
        "role_playing": "Capable of taking on different personas and characters for creative scenarios.",
        "conversationalist": "Designed for natural, flowing conversations that feel human-like.",
        "multilingual": "Proficient in multiple languages and can understand and respond in various languages.",
        "domain_expert:education": "Specialized in educational content, teaching, and learning.",
        "productivity_focused": "Optimized for tasks that enhance productivity and efficiency.",
        "creative_writer": "Skilled at generating creative written content like stories, poems, and creative text.",
        "sarcastic_tone": "Delivers responses with a sarcastic or witty tone.",
        "fast_response": "Optimized for quick response generation, trading off some quality for speed.",
        "free": "Models available for use without additional costs.",
        "paid": "Premium models that may incur additional costs for usage.",
        "multimodal": "Capable of processing and understanding multiple types of input like text and images."
    }
    
    # Default prompts for missing labels
    default_prompts = {
        "general_assistant": "You are a helpful, versatile assistant capable of providing information and assistance on various topics. Respond clearly and helpfully to the user's requests. Always respond in the same language as the user's query.",
        "reasoning_expert": "You are an expert at logical reasoning and problem-solving. Carefully analyze problems, break them down into components, and provide step-by-step, well-reasoned solutions. Always respond in the same language as the user's query.",
        "instruction_following": "You are designed to follow instructions precisely. Pay close attention to every detail in the user's request and execute it exactly as specified. Always respond in the same language as the user's query.",
        "safety_focused": "You prioritize safety and ethical considerations in all responses. Avoid providing harmful, unethical, or dangerous information, even when explicitly requested. Always respond in the same language as the user's query.",
        "math_expert": "You are a mathematics expert. Solve mathematical problems with precision, showing your work step-by-step, and explain concepts clearly using appropriate mathematical notation when helpful. Always respond in the same language as the user's query.",
        "code_expert": "You are a programming and software development expert. Write clean, efficient code, debug problems effectively, and explain technical concepts clearly. Suggest best practices and optimal solutions. Always respond in the same language as the user's query.",
        "vision_expert": "You are specialized in analyzing and understanding visual information. Describe images in detail, identify objects and patterns, and respond accurately to questions about visual content. Always respond in the same language as the user's query.",
        "experimental": "You are an experimental AI with advanced capabilities still under development. Provide the best assistance possible while acknowledging any limitations you may have. Always respond in the same language as the user's query.",
        "role_playing": "You can take on different personas and roles based on the user's request. Stay consistent with the assigned character and respond as that entity would. Always respond in the same language as the user's query.",
        "conversationalist": "You excel at natural conversation. Maintain context, respond naturally, and engage the user in a way that feels like talking to a real person. Always respond in the same language as the user's query.",
        "multilingual": "You are proficient in multiple languages. Always respond in the same language as the user's query, and provide translations only when specifically requested.",
        "domain_expert:education": "You are an education specialist. Provide accurate information on educational topics, create learning materials, and explain concepts in an accessible, pedagogical manner. Always respond in the same language as the user's query.",
        "productivity_focused": "You help users be more efficient and productive. Provide concise, actionable information and suggestions that save time and improve workflow. Always respond in the same language as the user's query.",
        "creative_writer": "You are a creative writer skilled in various genres and styles. Generate original, imaginative content that matches the user's specifications and engages the reader. Always respond in the same language as the user's query.",
        "sarcastic_tone": "You have a sarcastic, witty personality. Respond with clever remarks and humorous observations while still providing helpful information. Always respond in the same language as the user's query.",
        "fast_response": "You prioritize speed in your responses. Provide concise, direct answers that get to the point quickly without unnecessary elaboration. Always respond in the same language as the user's query.",
        "free": "You are available without additional cost. Provide the best assistance possible within your capabilities. Always respond in the same language as the user's query.",
        "paid": "You are a premium model with advanced capabilities. Provide high-quality, detailed responses that reflect your enhanced performance. Always respond in the same language as the user's query.",
        "multimodal": "You can process both text and images. Analyze visual content when provided and integrate that understanding with textual information in your responses. Always respond in the same language as the user's query."
    }
    
    # Create new "labels" and "roles" sections
    new_labels = []
    new_roles = []
    
    # Process all labels
    processed_labels = set()
    
    # First add labels used in model_labels.json
    for label in sorted(all_used_labels):
        # Add description (use existing if available, else default)
        description = existing_label_descriptions.get(label, default_descriptions.get(label, f"A model specialized in {label} capabilities."))
        new_labels.append({
            "label": label,
            "description": description
        })
        
        # Add prompt for label (use existing if available, else default)
        prompt = existing_role_prompts.get(label, default_prompts.get(label, f"You are a specialized AI with expertise in {label}. Provide accurate and helpful responses relevant to this specialization. Always respond in the same language as the user's query."))
        new_roles.append({
            "label": label,
            "prompt": prompt
        })
        
        processed_labels.add(label)
    
    # Add existing defined labels that are not currently used (for future use)
    for label in existing_label_descriptions:
        if label not in processed_labels:
            new_labels.append({
                "label": label,
                "description": existing_label_descriptions[label]
            })
            
            # If prompt exists for label, add it
            if label in existing_role_prompts:
                new_roles.append({
                    "label": label,
                    "prompt": existing_role_prompts[label]
                })
            # Else add default
            else:
                prompt = default_prompts.get(label, f"You are a specialized AI with expertise in {label}. Provide accurate and helpful responses relevant to this specialization. Always respond in the same language as the user's query.")
                new_roles.append({
                    "label": label,
                    "prompt": prompt
                })
            
            processed_labels.add(label)
    
    # Add language consistency instruction to all prompts (if missing)
    updated_roles = []
    for role_entry in new_roles:
        prompt = role_entry["prompt"]
        language_instruction = "Always respond in the same language as the user's query."
        
        # If prompt doesn't already contain language instruction
        if language_instruction not in prompt:
            # Append to end of prompt
            updated_prompt = prompt + " " + language_instruction
            role_entry["prompt"] = updated_prompt
            
        updated_roles.append(role_entry)
    
    # Create new model_roles.json data
    new_model_roles = {
        "labels": new_labels,
        "roles": updated_roles
    }
    
    # Label-role consistency check
    label_names = {entry["label"] for entry in new_labels}
    role_names = {entry["label"] for entry in updated_roles}
    
    missing_roles = label_names - role_names
    missing_labels = role_names - label_names
    
    if missing_roles:
        logger.error(f"These labels have no role definition: {', '.join(missing_roles)}")
        return False
        
    if missing_labels:
        logger.error(f"These roles have no label definition: {', '.join(missing_labels)}")
        return False
    
    # Save changes
    if save_json(MODEL_ROLES_FILE, new_model_roles):
        logger.info(f"Model roles updated:")
        logger.info(f"  - {len(new_labels)} label definitions")
        logger.info(f"  - {len(updated_roles)} role prompts")
        logger.info(f"  - All labels have role definitions")
        logger.info(f"  - All roles have label definitions")
        logger.info(f"  - Language consistency instructions added to all prompts")
        return True
    else:
        logger.error("Error occurred while updating model roles")
        return False

if __name__ == "__main__":
    logger.info("Starting Model Roles Updater...")
    update_model_roles()