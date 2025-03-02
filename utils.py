import os
import json
import logging
import datetime
import requests
import regex as re
import time
import traceback
import inspect
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

def get_model_description(model_id):
    """
    Fetches a model's description from its OpenRouter model page.
    
    Args:
        model_id: The model ID in format provider/model-name
    
    Returns:
        A dictionary with model description details or an empty dict if unavailable
    """
    model_url = f"https://openrouter.ai/{model_id}"
    
    try:
        # Log the attempt
        logger.info(f"Attempting to fetch description for model: {model_id} from {model_url}")
        
        # Make a request to the model page (without using BeautifulSoup since it's not installed)
        response = requests.get(model_url, timeout=10)
        
        if response.status_code == 200:
            # Get the content and use regex to find key information
            content = response.text
            
            # Extract key pieces of information using regex
            # This is a simplified version without BeautifulSoup
            description = ""
            
            # Try to get the main description
            description_match = re.search(r'<meta name="description" content="([^"]+)"', content)
            if description_match:
                meta_description = description_match.group(1).strip()
                description += f"{meta_description}. "
            
            # Get context length
            context_match = re.search(r'Context Length[^<]*<[^>]*>([^<]+)', content)
            if context_match:
                context_length = context_match.group(1).strip()
                description += f"Context Length: {context_length}. "
                
            # Try to extract capabilities or features
            capabilities = []
            capability_matches = re.findall(r'<li[^>]*>([^<]+)(?:<[^>]+>)*</li>', content)
            if capability_matches:
                for match in capability_matches[:5]:  # limit to first 5 matches
                    cleaned = match.strip()
                    if len(cleaned) > 5 and len(cleaned) < 100:  # reasonable length
                        capabilities.append(cleaned)
            
            if capabilities:
                description += f"Capabilities: {', '.join(capabilities)}. "
                
            model_info = {
                "description": description,
                "capabilities": capabilities,
                "url": model_url, 
                "success": True
            }
            
            logger.info(f"Successfully retrieved info for model {model_id}: {description[:100]}...")
            return model_info
            
        else:
            logger.warning(f"Failed to fetch model page, status code: {response.status_code}")
            return {"success": False, "error": f"HTTP {response.status_code}"}
            
    except Exception as e:
        logger.error(f"Error fetching model description for {model_id}: {str(e)}")
        logger.debug(traceback.format_exc())
        return {"success": False, "error": str(e)}

def get_openrouter_models():
    """
    Fetches available models from OpenRouter API.
    No retry logic - immediately return error if API call fails.
    """
    if not OPENROUTER_API_KEY:
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
        
        # Log the number of models retrieved
        logger.info(f"Successfully retrieved {len(models_data)} models from OpenRouter")
        
        return models_data
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching OpenRouter models: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching OpenRouter models: {str(e)}")
        return []

def call_agent(model_name, role, query, openrouter_models, conversation_history=None, is_coordinator=False):
    """
    Calls an OpenRouter model with error handling and logging.
    Returns a tuple of (response_text, metadata) where metadata contains token usage and timing data.
    
    Args:
        model_name: The OpenRouter model ID to use
        role: The system role prompt
        query: The user query
        openrouter_models: List of available OpenRouter models
        conversation_history: Previous conversation history (optional)
        is_coordinator: Whether this model is being used as a coordinator (affects error handling)
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
        
        # No more retry logic - directly make the API call once
        max_retries = 0  # No retries
        retry_count = 0  # For tracking
        
        try:
            # Make API request
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={
                    "model": model_name,
                    "messages": messages
                },
                timeout=45
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
                    
                    # Immediately return error without retry attempts
                    error_template = f"OPENROUTER_ERROR_{error_code}: {error_msg}"
                    return (f"Error: {error_template}", {"tokens": token_usage, "time": time.time() - start_time, "cost": 0.0})
                
                # No specific error information - log the response structure
                error_msg = "'choices' not found in response or empty"
                logger.error(f"{error_msg}. Response keys: {list(result.keys())}")
                
                # Check if response contains user_id but no choices (common with some API errors)
                if 'user_id' in result:
                    logger.warning(f"Response contains user_id but no choices - likely a provider error")
                    # Return error immediately
                    return (f"Error: Provider returned incomplete response for {model_name}", 
                           {"tokens": token_usage, "time": time.time() - start_time, "cost": 0.0})
                else:
                    return (f"Error: {error_msg}", 
                           {"tokens": token_usage, "time": time.time() - start_time, "cost": 0.0})
            
            if not result['choices'][0].get('message', {}).get('content'):
                return (f"Error: No content found in response message", 
                       {"tokens": token_usage, "time": time.time() - start_time, "cost": 0.0})
            
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
                    # Get per-million token costs
                    prompt_cost_per_million = float(pricing.get('prompt', 0))
                    completion_cost_per_million = float(pricing.get('completion', 0))
                    
                    # Calculate costs directly for each token type, then display per million
                    prompt_cost = (prompt_cost_per_million / 1000000) * token_usage["prompt"]
                    completion_cost = (completion_cost_per_million / 1000000) * token_usage["completion"]
                    
                    # Calculate actual cost
                    actual_cost = prompt_cost + completion_cost
                    
                    # Convert to per-million for display
                    if token_usage["total"] > 0:
                        # Display as cost per million tokens
                        cost_per_token = actual_cost / token_usage["total"]
                        # Ölçeği artıralım ki çok küçük değerler 0.00 olarak görünmesin
                        cost = cost_per_token * 1000000
                        # Yuvarlama yapmadan tam değeri kullan, sadece çok küçük değerler için alt sınır koy
                        if 0 < cost < 0.000001:
                            cost = 0.000001
                    else:
                        # Fallback if no tokens (shouldn't happen but just in case)
                        cost = (prompt_cost_per_million + completion_cost_per_million) / 2
                    
            content = result['choices'][0]['message']['content']
            return (content, {
                "tokens": token_usage,
                "time": completion_time,
                "cost": cost,
                "model": model_name
            })
            
        except (requests.exceptions.RequestException, KeyError, ValueError) as e:
            # Immediately return error without retry
            logger.error(f"Error calling {model_name}: {str(e)}")
            return (f"Error: {str(e)}", {"tokens": token_usage, "time": time.time() - start_time, "cost": 0.0})
        
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