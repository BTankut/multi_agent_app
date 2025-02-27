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
    Returns a tuple of (response_text, metadata) where metadata contains token usage and timing data.
    """
    start_time = time.time()
    token_usage = {"prompt": 0, "completion": 0, "total": 0}
    cost = 0.0
    completion_time = 0
    
    try:
        # API call configuration
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Get model pricing information
        model_info = next((m for m in openrouter_models if m['id'] == model_name), None)
        pricing = None
        if model_info:
            pricing = model_info.get('pricing', {})
            
        # Verify model exists in the list
        model_exists = any(m['id'] == model_name for m in openrouter_models)
        if not model_exists:
            logger.warning(f"Model {model_name} not found in OpenRouter models list. Will attempt anyway.")
            
        # Check context length constraints
        context_length = next((int(m['context_length']) for m in openrouter_models 
                              if m['id'] == model_name), 4096)
        estimated_tokens = len(query) / 4  # Simple estimation
        if conversation_history and estimated_tokens > context_length:
            return (f"Error: Query exceeds context limit of {context_length} tokens.", 
                   {"tokens": token_usage, "time": time.time() - start_time, "cost": 0.0})
        
        # Prepare messages
        messages = []
        # Add conversation history properly if it exists
        if conversation_history:
            # Log that we're using conversation history
            logger.info(f"Using conversation history with {len(conversation_history)} messages")
            messages.extend(conversation_history)
            # Only add system role if not already in the conversation history
            if not any(msg.get("role") == "system" for msg in conversation_history):
                messages.append({"role": "system", "content": role})
            # Add the new user query
            messages.append({"role": "user", "content": query})
        else:
            # Standard message format without history
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
                completion_time = time.time() - start_time
                response.raise_for_status()
                result = response.json()
                
                # More robust error handling
                if not isinstance(result, dict):
                    raise ValueError(f"Expected dict response, got {type(result).__name__}")
                
                # Check if choices exists in the response
                if 'choices' not in result or not result['choices']:
                    # Check if there's an error in the response
                    if 'error' in result:
                        error_details = result.get('error', {})
                        error_msg = error_details.get('message', 'Unknown error')
                        error_code = error_details.get('code', 'Unknown code')
                        error_metadata = error_details.get('metadata', {})
                        
                        # Log detailed error information
                        logger.error(f"API error response for {model_name}: {error_msg} (Code: {error_code})")
                        
                        # Handle different error codes based on OpenRouter docs
                        # See: https://openrouter.ai/docs/api-reference/errors
                        
                        # Handle rate limiting - 429 Too Many Requests
                        if error_code == 429:
                            logger.warning(f"Rate limit exceeded for {model_name}: {error_msg}")
                            # Just raise here to trigger the retry logic with backoff
                            raise ValueError(f"Rate limit exceeded: {error_msg}")
                            
                        # Handle provider-specific errors - 503 Service Unavailable
                        elif error_code == 503 and "Provider returned error" in error_msg:
                            provider_name = error_metadata.get("provider_name", "unknown")
                            logger.error(f"Provider '{provider_name}' error for {model_name}. Response keys: {list(result.keys())}")
                            
                            # If this is a Featherless provider (sao10k models), use special handling
                            if provider_name == "Featherless" and "sao10k" in model_name:
                                logger.error(f"Known Featherless provider issue with model {model_name}")
                                # Return a more specific error message that can be caught by coordinator
                                raise ValueError(f"PROVIDER_SPECIFIC_ERROR: {model_name}")
                            else:
                                # Return a meaningful error message for other provider errors
                                raise ValueError(f"Provider '{provider_name}' error with model {model_name}: {error_msg}")
                        
                        # Handle credit/payment issues - 402 Payment Required
                        elif error_code == 402:
                            logger.error(f"Credit requirement issue for {model_name}: {error_msg}")
                            raise ValueError(f"Insufficient credits for model {model_name}: {error_msg}")
                        
                        # Handle not found errors - 404 Not Found
                        elif error_code == 404:
                            logger.error(f"Model not found: {model_name}")
                            raise ValueError(f"Model {model_name} not found or no longer available")
                        
                        # Handle validation errors - 400 Bad Request, 422 Unprocessable Entity
                        elif error_code in [400, 422]:
                            logger.error(f"Validation error for {model_name}: {error_msg}")
                            raise ValueError(f"Validation error: {error_msg}")
                            
                        # Handle auth errors - 401 Unauthorized, 403 Forbidden
                        elif error_code in [401, 403]:
                            logger.error(f"Authentication error: {error_msg}")
                            raise ValueError(f"Authentication error: {error_msg}")
                            
                        # General error handling for other codes
                        else:
                            raise KeyError(f"API error ({error_code}): {error_msg}")
                    
                    # No specific error information - log the response structure
                    error_msg = "'choices' not found in response or empty"
                    logger.error(f"{error_msg}. Response keys: {list(result.keys())}")
                    
                    # Check if response contains user_id but no choices (common with some API errors)
                    if 'user_id' in result:
                        logger.warning(f"Response contains user_id but no choices - likely a provider error")
                        # This might be a special case provider error without proper error structure
                        raise ValueError(f"Provider returned incomplete response for {model_name}")
                    else:
                        raise KeyError(error_msg)
                
                if not result['choices'][0].get('message', {}).get('content'):
                    raise KeyError("No content found in response message")
                
                # Extract token usage info if available
                if 'usage' in result:
                    usage = result['usage']
                    token_usage = {
                        "prompt": usage.get('prompt_tokens', 0),
                        "completion": usage.get('completion_tokens', 0),
                        "total": usage.get('total_tokens', 0)
                    }
                    
                    # Calculate cost if pricing information is available
                    if pricing:
                        # Get per-token costs (convert from per-million token costs)
                        prompt_cost_per_token = float(pricing.get('prompt', 0))
                        completion_cost_per_token = float(pricing.get('completion', 0))
                        
                        # Calculate cost in dollars
                        cost = (prompt_cost_per_token * token_usage["prompt"] + 
                               completion_cost_per_token * token_usage["completion"])
                        
                        # Convert to dollars per million tokens for display
                        if token_usage["total"] > 0:
                            cost = (cost / token_usage["total"]) * 1000000
                        
                content = result['choices'][0]['message']['content']
                return (content, {
                    "tokens": token_usage,
                    "time": completion_time,
                    "cost": cost,
                    "model": model_name
                })
                
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
        return (f"Error: Request to {model_name} timed out.", 
               {"tokens": token_usage, "time": time.time() - start_time, "cost": 0.0})
    except requests.exceptions.RequestException as e:
        logger.error(f"API error calling {model_name}: {str(e)}")
        return (f"Error: API request failed: {str(e)}",
               {"tokens": token_usage, "time": time.time() - start_time, "cost": 0.0})
    except Exception as e:
        logger.error(f"Unexpected error calling {model_name}: {str(e)}")
        return (f"Error: Unexpected error: {str(e)}",
               {"tokens": token_usage, "time": time.time() - start_time, "cost": 0.0})

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