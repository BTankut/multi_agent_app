import logging
import random
from utils import load_json, call_agent

# Initialize logger
logger = logging.getLogger(__name__)

def get_labels_for_model(model_name):
    """
    Returns the labels associated with a specific model.
    """
    model_labels_data = load_json("data/model_labels.json")
    if not model_labels_data:
        return []
    
    for model_entry in model_labels_data:
        if model_entry["model"] == model_name:
            return model_entry["labels"]
    
    return []

def get_model_roles(selected_models, labels):
    """
    Assigns appropriate roles to selected models based on their capabilities and the query labels.
    """
    roles_data = load_json("data/model_roles.json")
    if not roles_data:
        return {}
    
    model_roles = {}
    for model_name in selected_models:
        model_labels = get_labels_for_model(model_name)
        
        # Find the most specialized role this model can fulfill
        assigned_role = "You are a general-purpose assistant."  # Default role
        
        # Priority: Assign the most specialized role from the query labels
        for label in labels:
            if label in model_labels:
                for role_entry in roles_data["roles"]:
                    if role_entry["label"] == label:
                        assigned_role = role_entry["prompt"]
                        break
                        
        model_roles[model_name] = assigned_role
    
    return model_roles

def get_models_by_labels(labels, option, openrouter_models, min_models=1, max_models=None):
    """
    Selects models based on labels, user option, and OpenRouter models.
    Respects min_models and max_models constraints.
    Limits models per provider to avoid rate limiting issues.
    """
    model_labels_data = load_json("data/model_labels.json")
    if not model_labels_data:
        return []
    
    # Find models matching the required labels
    matching_models = []
    for model_entry in model_labels_data:
        if any(label in model_entry["labels"] for label in labels):
            matching_models.append(model_entry["model"])
            
    # If no models match the specified labels, fallback to general_assistant models
    if not matching_models:
        logger.warning(f"No models found for labels: {', '.join(labels)}. Falling back to general_assistant.")
        for model_entry in model_labels_data:
            if "general_assistant" in model_entry["labels"]:
                matching_models.append(model_entry["model"])
    
    # Filter based on user option
    if option == "free":
        filtered_models = [model for model in matching_models if "free" in get_labels_for_model(model)]
    elif option == "paid":
        filtered_models = [model for model in matching_models if "free" not in get_labels_for_model(model)]
    elif option == "optimized":
        filtered_models = select_optimized_models(matching_models, labels, openrouter_models)
    else:
        return []
    
    # Enforce min_models and max_models constraints
    if len(filtered_models) < min_models and option == "optimized":
        # Try to find additional models from all available models
        all_models = [entry["model"] for entry in model_labels_data]
        additional_models = select_optimized_models(all_models, labels, openrouter_models)
        
        for model in additional_models:
            if model not in filtered_models:
                filtered_models.append(model)
                if len(filtered_models) >= min_models:
                    break
    
    # Apply max_models constraint if specified
    if max_models is not None:
        filtered_models = filtered_models[:max_models]
    
    # Apply provider diversity constraints to ensure we don't overuse any single provider
    filtered_models = limit_models_per_provider(filtered_models, max_per_provider=2)
    
    return filtered_models

def select_optimized_models(matching_models, query_labels, openrouter_models):
    """
    Selects the most cost-effective models from the matching candidates.
    Includes consideration for provider diversity and model family diversity.
    Filters out beta/alpha models for stability.
    """
    if not openrouter_models:
        return []

    model_costs = []
    for model_name in matching_models:
        # Skip beta/alpha models for stability
        if ":beta" in model_name or ":alpha" in model_name:
            continue
            
        # Check if the model exists in our model_labels.json
        model_labels = get_labels_for_model(model_name)
        if not model_labels:
            continue

        # Look for the model in openrouter_models
        matched_model = None
        for model in openrouter_models:
            # Remove any prefix from the model name for matching purposes
            model_id = model.get('id', '')
            if model_id == model_name or model_name in model_id:
                matched_model = model
                break
        
        if matched_model:
            try:
                # Extract pricing information
                prompt_cost = float(matched_model.get('pricing', {}).get('prompt', float('inf')))
                completion_cost = float(matched_model.get('pricing', {}).get('completion', float('inf')))
                context_length = int(matched_model.get('context_length', 1000))
                
                # Dynamic token estimation based on query complexity
                estimated_input_tokens = min(500, len(query_labels) * 100 + 300)
                estimated_output_tokens = min(250, len(query_labels) * 50 + 150)
                
                # Calculate total cost using the formula
                efficiency = ((prompt_cost + completion_cost) * context_length) / 1000.0
                relevance_score = sum(1 for label in query_labels if label in model_labels)
                
                # Extract provider for diversity bonus
                provider = model_name.split('/')[0] if '/' in model_name else "unknown"
                
                # Extract model family
                model_parts = model_name.split('/')
                if len(model_parts) > 1:
                    # Remove version indicators like "r1" to better identify base model family
                    name = model_parts[1]
                    # Extract model family core name (e.g. "dolphin3.0" or "claude-3")
                    if "dolphin" in name.lower():
                        family_key = f"{provider}/dolphin"
                    elif "claude" in name.lower():
                        # For claude models, use more specific versioning (claude-3, claude-2, etc.)
                        name_parts = name.split('-')
                        if len(name_parts) > 1:
                            family_key = f"{provider}/{name_parts[0]}-{name_parts[1]}"
                        else:
                            family_key = f"{provider}/{name_parts[0]}"
                    elif "mistral" in name.lower():
                        family_key = f"{provider}/mistral"
                    elif "llama" in name.lower():
                        family_key = f"{provider}/llama"
                    else:
                        # Default to first two segments for other models
                        name_parts = name.split('-')
                        if len(name_parts) > 1:
                            family_key = f"{provider}/{name_parts[0]}-{name_parts[1]}"
                        else:
                            family_key = f"{provider}/{name_parts[0]}"
                else:
                    family_key = model_name
                
                # Combined score (lower is better)
                total_score = efficiency * 0.3 - relevance_score * 0.7
                model_costs.append((model_name, total_score, provider, family_key))
            except (KeyError, TypeError) as e:
                logger.error(f"Error processing pricing for {model_name}: {e}")
                model_costs.append((model_name, float('inf'), "unknown", "unknown"))
        else:
            # If model not found in OpenRouter, give it an infinite cost
            logger.warning(f"Model {model_name} not found in OpenRouter models")
            provider = model_name.split('/')[0] if '/' in model_name else "unknown"
            
            # Extract model family for consistency
            model_parts = model_name.split('/')
            if len(model_parts) > 1:
                # Remove version indicators like "r1" to better identify base model family
                name = model_parts[1]
                # Extract model family core name (e.g. "dolphin3.0" or "claude-3")
                if "dolphin" in name.lower():
                    family_key = f"{provider}/dolphin"
                elif "claude" in name.lower():
                    # For claude models, use more specific versioning (claude-3, claude-2, etc.)
                    name_parts = name.split('-')
                    if len(name_parts) > 1:
                        family_key = f"{provider}/{name_parts[0]}-{name_parts[1]}"
                    else:
                        family_key = f"{provider}/{name_parts[0]}"
                elif "mistral" in name.lower():
                    family_key = f"{provider}/mistral"
                elif "llama" in name.lower():
                    family_key = f"{provider}/llama"
                else:
                    # Default to first two segments for other models
                    name_parts = name.split('-')
                    if len(name_parts) > 1:
                        family_key = f"{provider}/{name_parts[0]}-{name_parts[1]}"
                    else:
                        family_key = f"{provider}/{name_parts[0]}"
            else:
                family_key = model_name
                
            model_costs.append((model_name, float('inf'), provider, family_key))

    # Sort by score (lower is better)
    model_costs.sort(key=lambda x: x[1])
    
    # Implement manual provider diversity and model family diversity
    provider_counts = {}
    family_selected = set()
    optimized_models = []
    
    for model, _, provider, family_key in model_costs:
        # Skip if we've already selected a model from this family or too many from this provider
        if family_key in family_selected or provider_counts.get(provider, 0) >= 2:
            continue
            
        # Add model and update tracking
        optimized_models.append(model)
        provider_counts[provider] = provider_counts.get(provider, 0) + 1
        family_selected.add(family_key)
    
    return optimized_models

def calculate_similarity(response1, response2):
    """
    A simplified version that returns a random similarity score
    between 0.5 and 1.0 for demonstration purposes.
    In a production environment, this would use embeddings for real similarity.
    """
    if not response1 or not response2:
        return 0.0
    
    # Simple string comparison as fallback
    common_words = set(response1.lower().split()) & set(response2.lower().split())
    all_words = set(response1.lower().split()) | set(response2.lower().split())
    
    if not all_words:
        return 0.0
        
    # Jaccard similarity as a basic measure
    basic_similarity = len(common_words) / len(all_words)
    
    # Add some randomness to simulate more sophisticated embedding similarity
    randomized_similarity = basic_similarity * 0.5 + random.uniform(0.5, 0.9) * 0.5
    
    return min(randomized_similarity, 1.0)

def limit_models_per_provider(models, max_per_provider=2):
    """
    Limits the number of models from each provider to avoid rate limiting issues.
    Returns a list of models with at most max_per_provider models from each provider.
    Also ensures model family diversity (avoids multiple models from same family).
    Filters out beta models for stability.
    
    Example:
        - Input: ["anthropic/claude-3", "anthropic/claude-2", "google/gemini-1", "google/gemini-2", "google/gemini-3"]
        - Output (with max_per_provider=2): ["anthropic/claude-3", "google/gemini-1"]
    """
    if not models:
        return []
        
    provider_counts = {}      # Count by provider (e.g., "anthropic")
    family_selected = set()   # Track which model families we've already selected
    diversified_models = []
    
    # First pass - filter out beta models and sort by preference
    stable_models = []
    for model in models:
        # Skip beta/alpha models for stability
        if ":beta" in model or ":alpha" in model:
            continue
            
        # Add to stable models list
        stable_models.append(model)
    
    # If filtering removed all models, revert to original list
    if not stable_models and models:
        stable_models = models
    
    # Now process the filtered models
    for model in stable_models:
        # Extract provider from model name (assuming format is provider/model_name)
        provider = model.split('/')[0] if '/' in model else "unknown"
        
        # Extract model family (e.g., "claude-3" from "anthropic/claude-3-sonnet")
        model_parts = model.split('/')
        if len(model_parts) > 1:
            # Remove version indicators like "r1" to better identify base model family
            name = model_parts[1]
            # Extract model family core name (e.g. "dolphin3.0" or "claude-3")
            if "dolphin" in name.lower():
                family_key = f"{provider}/dolphin"
            elif "claude" in name.lower():
                # For claude models, use more specific versioning (claude-3, claude-2, etc.)
                name_parts = name.split('-')
                if len(name_parts) > 1:
                    family_key = f"{provider}/{name_parts[0]}-{name_parts[1]}"
                else:
                    family_key = f"{provider}/{name_parts[0]}"
            elif "mistral" in name.lower():
                family_key = f"{provider}/mistral"
            elif "llama" in name.lower():
                family_key = f"{provider}/llama"
            else:
                # Default to first two segments for other models
                name_parts = name.split('-')
                if len(name_parts) > 1:
                    family_key = f"{provider}/{name_parts[0]}-{name_parts[1]}"
                else:
                    family_key = f"{provider}/{name_parts[0]}"
        else:
            family_key = model
            
        # Skip if we've already selected from this model family or reached provider limit
        if family_key in family_selected or provider_counts.get(provider, 0) >= max_per_provider:
            continue
            
        # Add model and increment provider count
        diversified_models.append(model)
        provider_counts[provider] = provider_counts.get(provider, 0) + 1
        family_selected.add(family_key)
            
    # Log the model selection for debugging
    logger.info(f"Selected {len(diversified_models)} models after applying provider diversity constraints")
    for provider, count in provider_counts.items():
        if count > 0:
            logger.info(f"  - {provider}: {count} models")
            
    return diversified_models

def determine_complexity(query, labels):
    """
    Analyzes query complexity to determine appropriate model count.
    """
    # Base complexity from query length
    complexity = 1
    
    # Adjust based on query length
    if len(query) > 200:
        complexity += 2
    elif len(query) > 100:
        complexity += 1
    
    # Adjust based on specialized labels
    if "code_expert" in labels:
        complexity += 1
    if "math_expert" in labels:
        complexity += 1
    if "reasoning_expert" in labels:
        complexity += 1
    if "vision_expert" in labels:
        complexity += 1
    if "creative_writer" in labels:
        complexity += 1
    
    # Additional complexity factors
    keywords = ["analyze", "compare", "evaluate", "synthesize", "design", "explain", "create", "solve"]
    complexity += sum(1 for keyword in keywords if keyword.lower() in query.lower()) / 2
    
    # Convert to model counts
    min_models = max(1, int(complexity))
    max_models = max(3, min_models + 2)
    
    return min_models, max_models