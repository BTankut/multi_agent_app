import os
import json
import logging
import datetime
import requests
import regex as re
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    logger.warning("OPENROUTER_API_KEY not found in environment variables. API calls will fail.")

def load_json(file_path):
    """
    Loads and returns JSON data from a file.
    """
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {file_path}")
        return None

def get_openrouter_models():
    """
    Fetches available models from OpenRouter API.
    Includes retry logic for better reliability.
    """
    if not OPENROUTER_API_KEY:
        return []
    
    max_retries = 2
    retry_count = 0
    backoff_factor = 1.5  # seconds
    
    while retry_count <= max_retries:
        try:
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers=headers,
                timeout=15  # Increased from 10 to 15 seconds
            )
            
            response.raise_for_status()
            models_data = response.json().get('data', [])
            
            # Log the number of models retrieved
            logger.info(f"Successfully retrieved {len(models_data)} models from OpenRouter")
            
            return models_data
            
        except requests.exceptions.RequestException as e:
            retry_count += 1
            if retry_count <= max_retries:
                sleep_time = backoff_factor * (2 ** (retry_count - 1))
                logger.warning(f"Retry {retry_count}/{max_retries} for fetching models after {sleep_time}s due to: {str(e)}")
                time.sleep(sleep_time)
            else:
                logger.error(f"Error fetching OpenRouter models after {max_retries} retries: {str(e)}")
                return []
        except Exception as e:
            logger.error(f"Unexpected error fetching OpenRouter models: {str(e)}")
            return []

def call_agent(model_name, role, query, openrouter_models, conversation_history=None):
    """
    Calls an OpenRouter model with error handling and logging.
    """
    try:
        # API call configuration
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Verify model exists in the list
        model_exists = any(m['id'] == model_name for m in openrouter_models)
        if not model_exists:
            logger.warning(f"Model {model_name} not found in OpenRouter models list. Will attempt anyway.")
            
        # Check context length constraints
        context_length = next((int(m['context_length']) for m in openrouter_models 
                              if m['id'] == model_name), 4096)
        estimated_tokens = len(query) / 4  # Simple estimation
        if conversation_history and estimated_tokens > context_length:
            return f"Error: Query exceeds context limit of {context_length} tokens."
        
        # Prepare messages
        messages = []
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "system", "content": role})
        messages.append({"role": "user", "content": query})
        
        # Log the API call attempt
        logger.info(f"Calling model: {model_name} with {len(messages)} messages")
        
        # Make API request with retry logic
        max_retries = 3  # Increased from 2 to 3
        retry_count = 0
        backoff_factor = 2  # Increased from 1 to 2 seconds
        
        while retry_count <= max_retries:
            try:
                # Make API request
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": model_name,
                        "messages": messages
                    },
                    timeout=45  # Increased from 30 to 45 seconds
                )
                response.raise_for_status()
                result = response.json()
                
                # More robust error handling
                if not isinstance(result, dict):
                    raise ValueError(f"Expected dict response, got {type(result).__name__}")
                
                # Check if choices exists in the response
                if 'choices' not in result or not result['choices']:
                    # More detailed error handling for debugging
                    error_msg = "'choices' not found in response or empty"
                    logger.error(f"{error_msg}. Response keys: {list(result.keys())}")
                    raise KeyError(error_msg)
                
                if not result['choices'][0].get('message', {}).get('content'):
                    raise KeyError("No content found in response message")
                    
                return result['choices'][0]['message']['content']
                
            except (requests.exceptions.RequestException, KeyError, ValueError) as e:
                retry_count += 1
                if retry_count <= max_retries:
                    # Exponential backoff
                    sleep_time = backoff_factor * (2 ** (retry_count - 1))
                    logger.warning(f"Retry {retry_count}/{max_retries} for {model_name} after {sleep_time}s due to: {str(e)}")
                    time.sleep(sleep_time)
                else:
                    # All retries failed
                    raise
        
    except requests.exceptions.Timeout:
        logger.error(f"Timeout error calling {model_name}")
        return f"Error: Request to {model_name} timed out."
    except requests.exceptions.RequestException as e:
        logger.error(f"API error calling {model_name}: {str(e)}")
        return f"Error: API request failed: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error calling {model_name}: {str(e)}")
        return f"Error: Unexpected error: {str(e)}"

def log_conversation(coordinator_messages, agent_messages):
    """
    Records the full conversation flow to a log file and updates the session state.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open("conversation_log.txt", "a") as log_file:
        log_file.write(f"--- New Conversation ({timestamp}) ---\n")
        log_file.write(f"Coordinator Messages:\n{json.dumps(coordinator_messages, indent=2)}\n")
        for agent, messages in agent_messages.items():
            log_file.write(f"Agent {agent} Messages:\n{json.dumps(messages, indent=2)}\n")
        log_file.write("--- End of Conversation ---\n\n")

def handle_error(message, error_log_placeholder=None):
    """
    Handles errors, logs them, and optionally displays them to the user.
    """
    logger.error(message)
    if error_log_placeholder:
        error_log_placeholder.error(message)  # Display error in Streamlit