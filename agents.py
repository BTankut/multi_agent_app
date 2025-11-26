import logging
import random
from utils import load_json, call_agent

# Initialize logger
logger = logging.getLogger(__name__)

# Load intelligence data if available
def load_model_intelligence():
    intelligence_data = load_json("data/model_intelligence.json")
    if not intelligence_data:
        return {}
    
    # Convert list to dict for fast lookup
    return {m["id"]: m for m in intelligence_data.get("models", [])}

# Load specific roles if available
def load_specific_roles():
    return load_json("data/model_specific_roles.json") or {}

# Static Fallback Model Tier System (Used only if intelligence data is missing)
STATIC_MODEL_TIERS = {
    # Tier 1: Premium - Latest flagship models (Nov 2025)
    1: [
        # OpenAI GPT-5 Series (Latest - Nov 2025)
        "openai/gpt-5.1",
        "openai/gpt-5.1-chat",
        "openai/gpt-5.1-codex",
        "openai/gpt-5.1-codex-mini",
        "openai/gpt-5-pro",
        "openai/gpt-5-codex",
        "openai/gpt-5-image",
        # OpenAI O-Series (Reasoning)
        "openai/o3-deep-research",
        "openai/o4-mini-deep-research",
        "openai/o1",
        # Google Gemini 3 Series (Latest)
        "google/gemini-3-pro-preview",
        "google/gemini-3-pro-image-preview",
        # Anthropic Claude 4 Series
        "anthropic/claude-sonnet-4.5",
        "anthropic/claude-opus-4",
        "anthropic/claude-opus-4.1",
        "anthropic/claude-haiku-4.5",
        # DeepSeek R1 (Reasoning)
        "deepseek/deepseek-r1",
        "deepseek/deepseek-r1-0528",
        # X.AI Grok 4 Series
        "x-ai/grok-4",
        "x-ai/grok-4-fast",
        "x-ai/grok-4.1-fast",
    ],
    # ... (rest of the static list is implied/preserved in logic below)
}

def get_model_tier(model_name):
    """
    Returns the tier number for a given model.
    Prioritizes dynamic intelligence data, falls back to static list.
    Lower tier = higher priority.
    Returns 4 for unknown models.
    """
    # 1. Try Dynamic Intelligence
    intelligence = load_model_intelligence()
    if intelligence and model_name in intelligence:
        return intelligence[model_name].get("tier", 4)
        
    # 2. Fallback to Static List
    # (We use the global MODEL_TIERS defined below for backward compatibility if this function is called before update)
    # Ideally, we should move the large dictionary to a fallback file, but for now we check the global var if defined
    if 'MODEL_TIERS' in globals():
        for tier, models in globals()['MODEL_TIERS'].items():
            for tier_model in models:
                if model_name == tier_model or tier_model in model_name:
                    return tier
                    
    return 4  # Unknown/Experimental models

# ... (keep get_tier_bonus as is) ...

def get_model_roles(selected_models, labels):
    """
    Assigns appropriate roles to selected models.
    Prioritizes model-specific roles from intelligence analysis.
    Falls back to label-based generic roles.
    """
    roles_data = load_json("data/model_roles.json")
    specific_roles = load_specific_roles()
    
    model_roles = {}
    for model_name in selected_models:
        # 1. Try Specific Role (High Precision)
        if specific_roles and model_name in specific_roles:
            model_roles[model_name] = specific_roles[model_name]
            continue
            
        # 2. Fallback to Label-Based Role (Generic)
        model_labels = get_labels_for_model(model_name)
        assigned_role = "You are a general-purpose assistant."  # Default role
        
        if roles_data:
            # Priority: Assign the most specialized role from the query labels
            for label in labels:
                if label in model_labels:
                    for role_entry in roles_data.get("roles", []):
                        if role_entry["label"] == label:
                            assigned_role = role_entry["prompt"]
                            break
        
        model_roles[model_name] = assigned_role
    
    return model_roles

def get_tier_bonus(tier):
    """
    Returns the scoring bonus for a given tier.
    Lower score = better priority in selection.

    IMPORTANT: Tier bonuses are VERY strong to ensure flagship models
    (GPT-5.1, Claude 4, Gemini-3) are prioritized over unknown models,
    even if unknown models have slightly better relevance scores.
    """
    tier_bonuses = {
        1: -5.0,  # Premium flagship models get massive bonus
        2: -2.5,  # Strong models get large bonus
        3: -1.0,  # Good models get moderate bonus
        4: 0.0    # Unknown models get no bonus
    }
    return tier_bonuses.get(tier, 0.0)

def find_model_in_catalog(model_name, openrouter_models):
    """
    Finds a model entry in the OpenRouter catalog by exact or partial match.
    """
    if not openrouter_models:
        return None

    for model in openrouter_models:
        model_id = model.get('id', '')
        if model_id == model_name or model_name in model_id:
            return model
    return None

def is_text_compatible_model(model_name, openrouter_models):
    """
    Checks if a model is compatible with text-to-text queries.
    Returns True if the model can output text, False if it's image-only.

    Filters out:
    - Models that don't have 'text' in output_modalities
    - Models specifically designed for image generation (image-preview, image-generation in name)
    """
    # Stronger filtering for image-focused models in text queries
    # These patterns indicate models that prioritize image generation or are multimodal-focused in a way that might be overkill/wrong for pure text
    image_only_patterns = [
        '-image-preview', '-image-generation', '/image-', '-to-image', 'image-mini',
        'dall-e', 'midjourney', 'stable-diffusion', 'flux', 
        '-vision-preview', '-video', 'tts', 'audio'
    ]
    
    model_lower = model_name.lower()
    for pattern in image_only_patterns:
        if pattern in model_lower:
            # Exception: Some vision models are also great text models (like gpt-4-vision, claude-3-opus)
            # But specific 'image-preview' or 'image-generation' usually implies generation focus
            # We'll keep "vision" models ONLY if they don't have "image" in the name, assuming they are multimodal chat
            if "vision" in pattern and "image" not in model_lower:
                continue
            
            # STRICT BLOCK: If the name explicitly contains "image" and it's not a known exception
            logger.info(f"Filtering out image/media-focused model: {model_name}")
            return False

    # Check OpenRouter architecture info
    model_info = find_model_in_catalog(model_name, openrouter_models)
    if model_info:
        architecture = model_info.get('architecture', {})
        output_modalities = architecture.get('output_modalities', [])

        # If output_modalities is specified and doesn't include 'text', filter out
        if output_modalities and 'text' not in output_modalities:
            logger.info(f"Filtering out non-text output model: {model_name} (outputs: {output_modalities})")
            return False

    return True

def extract_provider_and_family(model_name):
    """
    Extracts provider and a normalized family key from a model identifier.
    The family key keeps one representative per major model family.
    """
    provider = model_name.split('/')[0] if '/' in model_name else "unknown"
    
    model_parts = model_name.split('/')
    if len(model_parts) > 1:
        name = model_parts[1]
        # Extract model family core name (e.g. dolphin, claude-3, llama)
        if "dolphin" in name.lower():
            family_key = f"{provider}/dolphin"
        elif "claude" in name.lower():
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
            name_parts = name.split('-')
            if len(name_parts) > 1:
                family_key = f"{provider}/{name_parts[0]}-{name_parts[1]}"
            else:
                family_key = f"{provider}/{name_parts[0]}"
    else:
        family_key = model_name
        
    return provider, family_key

def prioritize_models_by_tier(models, openrouter_models):
    """
    Returns models sorted to emphasize tier priority, availability, and recency.
    Lower tier is always favored; newer catalog entries are preferred within the same tier.
    """
    if not models:
        return []

    prioritized = []
    for model in models:
        tier = get_model_tier(model)
        catalog_entry = find_model_in_catalog(model, openrouter_models)
        days_old = 9999
        if catalog_entry:
            created_at = catalog_entry.get('created_at', '')
            if created_at:
                try:
                    import datetime
                    creation_date = datetime.datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    now = datetime.datetime.now(datetime.timezone.utc)
                    days_old = max((now - creation_date).days, 0)
                except (ValueError, TypeError):
                    pass
        availability_penalty = 0 if catalog_entry else 1
        prioritized.append(((tier, availability_penalty, days_old), model))

    prioritized.sort(key=lambda x: x[0])
    return [model for _, model in prioritized]

def deduplicate_preserve_order(models):
    """
    Removes duplicates while preserving the original order.
    """
    if not models:
        return []

    seen = set()
    unique_models = []
    for model in models:
        if model not in seen:
            unique_models.append(model)
            seen.add(model)
    return unique_models

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
    Assigns appropriate roles to selected models.
    Prioritizes model-specific roles from intelligence analysis.
    Falls back to label-based generic roles.
    """
    roles_data = load_json("data/model_roles.json")
    specific_roles = load_specific_roles()
    
    model_roles = {}
    for model_name in selected_models:
        # 1. Try Specific Role (High Precision)
        if specific_roles and model_name in specific_roles:
            model_roles[model_name] = specific_roles[model_name]
            continue
            
        # 2. Fallback to Label-Based Role (Generic)
        model_labels = get_labels_for_model(model_name)
        assigned_role = "You are a general-purpose assistant."  # Default role
        
        if roles_data:
            # Priority: Assign the most specialized role from the query labels
            for label in labels:
                if label in model_labels:
                    for role_entry in roles_data.get("roles", []):
                        if role_entry["label"] == label:
                            assigned_role = role_entry["prompt"]
                            break
        
        model_roles[model_name] = assigned_role
    
    return model_roles

def get_models_by_labels(labels, option, openrouter_models, min_models=1, max_models=None, text_only=True):
    """
    Selects models based on labels, user option, and OpenRouter models.
    Respects min_models and max_models constraints.
    Limits models per provider with tier-aware rules to keep premium models in every mode.

    Args:
        text_only: If True, filters out image-only models that can't handle text-to-text queries (default: True)
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

    # Filter out image-only models for text-to-text queries
    if text_only and openrouter_models:
        text_compatible_models = [m for m in matching_models if is_text_compatible_model(m, openrouter_models)]
        filtered_count = len(matching_models) - len(text_compatible_models)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} image-only models for text query")
        matching_models = text_compatible_models
    
    # Filter based on user option
    if option == "free":
        filtered_models = [model for model in matching_models if "free" in get_labels_for_model(model)]
    elif option == "paid":
        filtered_models = [model for model in matching_models if "free" not in get_labels_for_model(model)]
    elif option == "optimized":
        filtered_models = select_optimized_models(matching_models, labels, openrouter_models)
    else:
        return []

    filtered_models = deduplicate_preserve_order(filtered_models)
    
    # Enforce min_models with option-aware fallbacks
    if len(filtered_models) < min_models:
        if option == "optimized":
            # Try to find additional models from all available models with optimization scoring
            all_models = [entry["model"] for entry in model_labels_data]
            additional_models = select_optimized_models(all_models, labels, openrouter_models)
            for model in additional_models:
                if model not in filtered_models:
                    filtered_models.append(model)
                if len(filtered_models) >= min_models:
                    break
        else:
            # Broaden pool while respecting the free/paid constraint
            allowed_pool = []
            for entry in model_labels_data:
                if option == "free" and "free" in entry["labels"]:
                    allowed_pool.append(entry["model"])
                elif option == "paid" and "free" not in entry["labels"]:
                    allowed_pool.append(entry["model"])
            for model in prioritize_models_by_tier(allowed_pool, openrouter_models):
                if model not in filtered_models:
                    filtered_models.append(model)
                if len(filtered_models) >= min_models:
                    break
    
    # Tier-first ordering before diversity filtering
    filtered_models = prioritize_models_by_tier(filtered_models, openrouter_models)
    
    # Apply provider diversity constraints with tier-aware limits and global caps
    filtered_models = limit_models_per_provider(
        filtered_models, 
        min_models=min_models, 
        max_models=max_models, 
        openrouter_models=openrouter_models
    )
    
    # Ensure we don't exceed the max_models constraint after relaxing diversity rules
    if max_models is not None:
        filtered_models = filtered_models[:max_models]
    
    return filtered_models

def select_optimized_models(matching_models, query_labels, openrouter_models):
    """
    Selects the most cost-effective models from the matching candidates.
    Includes consideration for provider diversity and model family diversity.
    Filters out beta/alpha models for stability.
    Prioritizes newer models over older ones when available.
    Uses tier system to prioritize well-known flagship models (Claude, GPT-4, Gemini, etc.)
    """
    if not openrouter_models:
        return []

    model_costs = []
    for model_name in matching_models:
        # Skip beta/alpha models for stability
        if ":beta" in model_name or ":alpha" in model_name:
            continue

        provider, family_key = extract_provider_and_family(model_name)
            
        # Check if the model exists in our model_labels.json
        model_labels = get_labels_for_model(model_name)
        if not model_labels:
            continue

        # Look for the model in openrouter_models
        matched_model = find_model_in_catalog(model_name, openrouter_models)
        
        if matched_model:
            try:
                # Extract pricing information
                prompt_cost = float(matched_model.get('pricing', {}).get('prompt', float('inf')))
                completion_cost = float(matched_model.get('pricing', {}).get('completion', float('inf')))
                context_length = int(matched_model.get('context_length', 1000))
                
                # Get created_at date for recency (if available)
                created_at = matched_model.get('created_at', '')
                
                # Dynamic token estimation based on query complexity
                estimated_input_tokens = min(500, len(query_labels) * 100 + 300)
                estimated_output_tokens = min(250, len(query_labels) * 50 + 150)
                
                # Calculate total cost using the formula
                efficiency = ((prompt_cost + completion_cost) * context_length) / 1000.0
                relevance_score = sum(1 for label in query_labels if label in model_labels)
                # Combined score (lower is better)
                # Add recency bonus: newer models get a slight boost (negative value improves score)
                recency_bonus = 0
                if created_at:
                    try:
                        import datetime
                        # Estimate how recent the model is (in days)
                        creation_date = datetime.datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        now = datetime.datetime.now(datetime.timezone.utc)
                        days_old = (now - creation_date).days
                        # Apply recency bonus: newer models get better scores
                        # Cap at 90 days (~3 months) for a maximum 0.2 bonus
                        recency_bonus = min(days_old / 450, 0.2)  # Older models get penalties up to 0.2
                    except (ValueError, TypeError):
                        # If date parsing fails, no recency bonus
                        pass
                
                # Get tier bonus for prioritizing known good models
                model_tier = get_model_tier(model_name)
                tier_bonus = get_tier_bonus(model_tier)

                # Final score calculation with recency and tier bonuses
                # Lower score = better (tier bonus is negative for good models)
                total_score = efficiency * 0.3 - relevance_score * 0.7 + recency_bonus + tier_bonus

                # Log tier information for debugging (reduced verbosity)
                if model_tier <= 2:
                    logger.debug(f"Premium/Strong model detected: {model_name} (Tier {model_tier}, bonus: {tier_bonus})")

                model_costs.append((model_name, total_score, provider, family_key, created_at))
            except (KeyError, TypeError) as e:
                logger.error(f"Error processing pricing for {model_name}: {e}")
                model_costs.append((model_name, float('inf'), "unknown", "unknown", ""))
        else:
            # If model not found in OpenRouter, give it an infinite cost
            logger.warning(f"Model {model_name} not found in OpenRouter models")
                
            model_costs.append((model_name, float('inf'), provider, family_key, ""))

    # Sort by score (lower is better)
    model_costs.sort(key=lambda x: x[1])

    # Implement manual provider diversity and model family diversity
    # IMPORTANT: Tier 1-2 models bypass provider limits to ensure premium models are always selected
    provider_counts = {}
    family_selected = set()
    optimized_models = []

    for model, _, provider, family_key, _ in model_costs:
        # Check model tier
        model_tier = get_model_tier(model)

        # Skip if we've already selected a model from this family
        if family_key in family_selected:
            continue

        # For Tier 1-2 models, allow up to 3 models per provider (premium models get priority)
        # For Tier 3-4 models, limit to 2 models per provider
        max_per_provider = 3 if model_tier <= 2 else 2

        # Skip if too many models from this provider (respecting tier-based limits)
        if provider_counts.get(provider, 0) >= max_per_provider:
            continue

        # Add model and update tracking
        optimized_models.append(model)
        provider_counts[provider] = provider_counts.get(provider, 0) + 1
        family_selected.add(family_key)

        # Log tier 1-2 selections for debugging
        if model_tier <= 2:
            logger.debug(f"Selected premium model: {model} (Tier {model_tier})")
    
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

def limit_models_per_provider(models, max_per_provider=1, min_models=0, max_models=None, openrouter_models=None):
    """
    Limits the number of models from each provider to maximize diversity.
    Strictly enforces max_per_provider (default 1) to ensure we get e.g. 1 OpenAI, 1 Anthropic, 1 Google.
    
    Tier 1 (Flagship) Exception: 
    If we really need more models to meet min_models, we might allow a 2nd model from a provider, 
    but ONLY if it's a top-tier model and we can't find other providers.
    """
    if not models:
        return []

    # Strongly prioritize known premium tiers before applying diversity rules
    prioritized_models = prioritize_models_by_tier(models, openrouter_models or [])
    provider_counts = {}      # Count by provider (e.g., "anthropic")
    family_selected = set()   # Track which model families we've already selected
    diversified_models = []
    
    # First pass - filter out beta models and sort by preference
    stable_models = []
    for model in prioritized_models:
        # Skip beta/alpha models for stability
        if ":beta" in model or ":alpha" in model:
            continue
        stable_models.append(model)
    
    # If filtering removed all models, revert to original list
    if not stable_models and models:
        stable_models = prioritized_models
    
    # Now process the filtered models
    for model in stable_models:
        provider, family_key = extract_provider_and_family(model)
        
        # Strict provider limit: Default is 1 per provider
        # Only relax this if we are desperate to meet min_models count later (not in this loop)
        if provider_counts.get(provider, 0) >= max_per_provider:
            continue
            
        # Always allow only 1 model per family (e.g. don't pick both claude-3-opus and claude-3-sonnet)
        if family_key in family_selected:
            continue

        # Add model and update tracking
        diversified_models.append(model)
        provider_counts[provider] = provider_counts.get(provider, 0) + 1
        family_selected.add(family_key)

        # Obey overall max model count if provided
        if max_models is not None and len(diversified_models) >= max_models:
            break
            
    # If we haven't met min_models, do a second pass allowing a 2nd model from same provider
    # But ONLY for different families (e.g. GPT-4 and O1 is okay, but not GPT-4 and GPT-4-Turbo)
    if len(diversified_models) < min_models:
        for model in stable_models:
            if model in diversified_models:
                continue
                
            provider, family_key = extract_provider_and_family(model)
            
            # Still enforce family uniqueness
            if family_key in family_selected:
                continue
                
            # Allow up to 2 per provider in this fallback pass
            if provider_counts.get(provider, 0) >= 2:
                continue
                
            diversified_models.append(model)
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
            family_selected.add(family_key)
            
            if len(diversified_models) >= min_models:
                break
            
    # Log the model selection for debugging
    logger.info(f"Selected {len(diversified_models)} models after applying strict provider diversity (max {max_per_provider}/provider)")
    for provider, count in provider_counts.items():
        if count > 0:
            logger.info(f"  - {provider}: {count} models")
            
    return diversified_models

def determine_complexity(query, labels):
    """
    Analyzes query complexity to determine appropriate model count.
    Optimized to prevent overkill on simple queries.
    """
    # 1. Check for very simple/short queries (Simple Factual)
    # e.g. "Capital of France", "Who is CEO of Apple"
    is_short = len(query) < 80
    has_complex_labels = any(l in labels for l in ["code_expert", "math_expert", "reasoning_expert", "creative_writer"])
    
    # Keywords indicating complexity
    complex_keywords = ["compare", "analyze", "why", "how", "explain", "solve", "code", "script", "debug", "optimize", "story", "essay"]
    has_complex_keywords = any(keyword in query.lower() for keyword in complex_keywords)
    
    if is_short and not has_complex_labels and not has_complex_keywords:
        # Simple query: Strictly 2 models for verification
        return 2, 2

    # 2. Standard Queries
    complexity = 2
    
    # Adjust based on query length
    if len(query) > 300:
        complexity += 1
    
    # Adjust based on specialized labels
    if "code_expert" in labels:
        complexity += 1
    if "math_expert" in labels:
        complexity += 1
    if "reasoning_expert" in labels:
        complexity += 1
        
    # Cap complexity
    min_models = min(max(2, int(complexity)), 3) # Usually 2 or 3
    max_models = min_models + 1                  # Max 3 or 4
    
    return min_models, max_models
