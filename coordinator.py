import re
import logging
from utils import load_json, call_agent, log_conversation
from agents import get_models_by_labels, get_model_roles, determine_complexity, calculate_similarity, get_labels_for_model

# Initialize logger
logger = logging.getLogger(__name__)

def determine_query_labels(query, coordinator_model, openrouter_models, coordinator_history=None):
    """
    Analyzes the query using the coordinator model to identify appropriate labels.
    
    Args:
        query: The user query to analyze
        coordinator_model: The model to use for coordination
        openrouter_models: List of available models
        coordinator_history: Previous conversation with coordinator (optional)
    """
    # Load available labels from the model_roles.json file
    roles_data = load_json('data/model_roles.json')
    if not roles_data:
        logger.error("Could not load model roles data")
        return []
        
    available_labels = [label_entry['label'] for label_entry in roles_data['labels']]
    
    # Create a prompt asking the coordinator to determine relevant labels
    if coordinator_history and len(coordinator_history) > 0:
        # Include context from previous conversation
        context = "\n\nPrevious conversation context:\n"
        
        # Limit to last 5 conversation turns (10 messages) to avoid context length issues
        # Only include user and assistant messages in history (not system messages)
        relevant_history = [msg for msg in coordinator_history if msg["role"] in ["user", "assistant"]]
        # Limit to last 5 conversation turns for clarity and context length
        relevant_history = relevant_history[-10:] if len(relevant_history) > 10 else relevant_history
        
        for message in relevant_history:
            role = message["role"]
            content = message["content"]
            # Improve formatting for better context
            if role == "user":
                context += f"User Question: {content}\n\n"
            else:
                if "Response:" in content:
                    # Extract just the response part if it's in our format
                    response_part = content.split("Response:", 1)[1].strip()
                    context += f"Previous Answer: {response_part}\n\n"
                else:
                    context += f"Previous Answer: {content}\n\n"
        
        # Log what we're doing
        logger.info(f"Using {len(relevant_history)} messages as context for label determination")
        
        prompt = f"""You are a helpful assistant tasked with analyzing a user's query and identifying relevant labels.
        
        Predefined Labels: {', '.join(available_labels)}
        
        {context}
        
        User's new query: {query}
        
        Consider the conversation context and the new query.
        Return ONLY the labels for the new query, separated by commas. No additional explanation.
        Always respond in the same language as the user query.
        """
    else:
        # Standard prompt without context
        logger.info("No conversation history available for label determination")
        prompt = f"""You are a helpful assistant tasked with analyzing a user's query and identifying relevant labels.
        
        Predefined Labels: {', '.join(available_labels)}
        
        User Query: {query}
        
        Return ONLY the labels, separated by commas. No additional explanation.
        Always respond in the same language as the user query.
        """
    
    # Ask the coordinator model to analyze the query
    # Use the conversation history for the coordinator if available
    try:
        # Explicitly mark that this is a coordinator model call
        response_tuple = call_agent(coordinator_model, "You are a helpful assistant.", prompt, openrouter_models, 
                                   conversation_history=coordinator_history, is_coordinator=True)
        
        # Handle the new tuple return format from call_agent
        response = None
        if isinstance(response_tuple, tuple) and len(response_tuple) == 2:
            response, metadata = response_tuple
        else:
            # Backward compatibility
            response = response_tuple
        
        # Check for specific OpenRouter error format we added to utils.py
        if response and isinstance(response, str) and response.startswith("Error: OPENROUTER_ERROR_"):
            error_parts = response.split(":")
            if len(error_parts) >= 3:
                error_code = error_parts[1].strip().split("_")[2]
                error_message = ":".join(error_parts[2:]).strip()
                logger.error(f"Coordinator received OpenRouter error {error_code}: {error_message}")
                
                # Pass the error upward with specific error code for better handling
                raise ValueError(f"OpenRouter error {error_code}: {error_message}")
        
        # Continue with normal processing if no error
        if response and isinstance(response, str) and not response.startswith("Error"):
            extracted_labels = [label.strip().lower() for label in re.split(r'[,\s]+', response) if label.strip()]
            validated_labels = [label for label in extracted_labels if label in available_labels]
            return validated_labels
            
    except Exception as e:
        error_str = str(e).lower()
        # Check for specific OpenRouter error codes
        if any(err in error_str for err in ["400", "401", "402", "403", "408", "429", "502", "503"]):
            logger.error(f"Coordinator API error: {error_str}")
            
            # Try to find the last used coordinator model from history if available
            recent_coordinator_models = []
            try:
                import os
                import json
                coord_history_file = "data/coordinator_history.json"
                if os.path.exists(coord_history_file):
                    with open(coord_history_file, 'r') as f:
                        data = json.load(f)
                        recent_coordinator_models = data.get('models', [])
            except Exception as history_err:
                logger.error(f"Failed to load coordinator history: {str(history_err)}")
            
            # Find an alternative model that is different from the current one
            alt_model = None
            if recent_coordinator_models:
                for model in recent_coordinator_models:
                    if model != coordinator_model:
                        alt_model = model
                        break
            
            # Provide informative error message including alternative model suggestion
            error_msg = f"Coordinator model error: {str(e)}. "
            if alt_model:
                error_msg += f"Try using alternative coordinator model: {alt_model}"
            else:
                # Hardcoded fallback if we can't find an alternative in history
                alt_model = "google/gemini-2.0-pro-exp-02-05:free"
                error_msg += f"Try using alternative coordinator model: {alt_model}"
            
            # Raise the error to be handled in the app.py
            raise ValueError(error_msg)
        else:
            # For other errors, continue to fallback
            logger.warning(f"Coordinator model error: {str(e)}, using fallback method")
    
    # Fallback to basic label determination if the coordinator model fails
    logger.warning("Coordinator model failed to determine labels, using fallback method")
    fallback_labels = ["general_assistant"]  # Always include general_assistant as a fallback
    
    # Simple keyword-based labeling as fallback
    if any(keyword in query.lower() for keyword in ["code", "program", "function", "class", "algorithm", "api"]):
        fallback_labels.append("code_expert")
    
    if any(keyword in query.lower() for keyword in ["math", "calculate", "equation", "formula", "solve", "statistical"]):
        fallback_labels.append("math_expert")
    
    if any(keyword in query.lower() for keyword in ["analyze", "compare", "evaluate", "reason", "logic"]):
        fallback_labels.append("reasoning_expert")
    
    if any(keyword in query.lower() for keyword in ["image", "picture", "photo", "describe", "see"]):
        fallback_labels.append("vision_expert")
    
    if any(keyword in query.lower() for keyword in ["story", "poem", "creative", "write", "fiction"]):
        fallback_labels.append("creative_writer")
    
    return fallback_labels

def coordinate_agents(query, coordinator_model, labels, openrouter_models, option, agent_history=None):
    """
    Coordinates the selected agents, calls them with appropriate roles,
    and synthesizes a final response.
    
    Args:
        query: The user query to process
        coordinator_model: The model to use for coordination
        labels: Labels extracted from the query
        openrouter_models: List of available models
        option: Selection strategy (free, paid, optimized)
        agent_history: Previous conversations with agents (optional)
    
    Returns:
        Tuple of (final_answer, updated_agent_history)
    """
    # Initialize agent history if not provided
    updated_agent_history = {}
    if agent_history:
        updated_agent_history = agent_history.copy()
    
    # Clear previous state if using Streamlit (but preserve conversation history)
    import streamlit as st
    if 'selected_agents' in st.session_state:
        st.session_state.selected_agents = []
        st.session_state.agent_responses = {}
        st.session_state.coordinator_messages = ""
        st.session_state.process_log = []
    
    # Add to process log for debugging
    if 'process_log' in st.session_state:
        st.session_state.process_log.append(f"Query received: {query}")
        st.session_state.process_log.append(f"Detected labels: {', '.join(labels)}")
        st.session_state.process_log.append(f"Using coordinator model: {coordinator_model}")
    
    # Determine dynamic model count based on query complexity
    min_models, max_models = determine_complexity(query, labels)
    
    if 'process_log' in st.session_state:
        st.session_state.process_log.append(f"Complexity determined: min_models={min_models}, max_models={max_models}")
    
    # Select models with updated constraints
    selected_models = get_models_by_labels(labels, option, openrouter_models, 
                                          min_models=min_models, max_models=max_models)
    
    # IMPORTANT: Ensure the coordinator model isn't selected as an agent
    if coordinator_model in selected_models:
        if 'process_log' in st.session_state:
            st.session_state.process_log.append(f"Removing coordinator model {coordinator_model} from agent selection to avoid conflicts")
        selected_models.remove(coordinator_model)
        
    # Log provider diversity information
    if 'process_log' in st.session_state:
        providers = {}
        for model in selected_models:
            provider = model.split('/')[0] if '/' in model else "unknown"
            providers[provider] = providers.get(provider, 0) + 1
        
        st.session_state.process_log.append(f"Provider diversity: {len(providers)} different providers")
        for provider, count in providers.items():
            st.session_state.process_log.append(f"  • {provider}: {count} models")
    
    # Store selected models in session state for UI display
    if 'selected_agents' in st.session_state:
        # Convert to set and back to list to remove duplicates
        st.session_state.selected_agents = list(set(selected_models))
        
        # Create a more detailed log of model selection (with unique models)
        if selected_models:
            # Get unique models
            unique_models = list(set(selected_models))
            st.session_state.process_log.append(f"Selected models: {len(unique_models)} unique models based on labels {', '.join(labels)}")
            for model in unique_models:
                model_labels = get_labels_for_model(model)
                matching_labels = [label for label in model_labels if label in labels]
                if matching_labels:
                    st.session_state.process_log.append(f"  • {model} selected for expertise in: {', '.join(matching_labels)}")
                else:
                    st.session_state.process_log.append(f"  • {model} selected as general model")
    
    if not selected_models:
        logger.error("No suitable models found for the query")
        error_message = "No suitable models were found to process this query."
        # Ensure we always return a tuple to match the expected return format
        return error_message, updated_agent_history
    
    # Assign roles to models
    model_roles = get_model_roles(selected_models, labels)
    agent_responses = {}
    
    # Log the assigned roles in a more readable format
    if 'process_log' in st.session_state:
        st.session_state.process_log.append(f"Assigning specialized roles to each model")
        for model, role in model_roles.items():
            # Get a concise summary of the role for display
            if "general-purpose assistant" in role:
                role_type = "General Assistant"
            elif "code expert" in role.lower():
                role_type = "Code Expert"
            elif "math expert" in role.lower():
                role_type = "Math Expert"
            elif "reasoning expert" in role.lower():
                role_type = "Reasoning Expert"
            elif "creative writer" in role.lower():
                role_type = "Creative Writer"
            elif "vision expert" in role.lower():
                role_type = "Vision Expert"
            else:
                # Extract first sentence for other roles
                role_type = role.split('.')[0] if '.' in role else role[:50] + "..."
            
            st.session_state.process_log.append(f"  • Assigned role to {model}: {role_type}")
    
    # Storage for token usage, cost, and timing data
    usage_data = {
        "total_tokens": 0,
        "total_cost": 0.0,
        "total_time": 0.0,
        "models": {}
    }
    
    # Log that we're calling models in parallel
    if 'process_log' in st.session_state:
        st.session_state.process_log.append(f"📡 Calling {len(selected_models)} agents in parallel")
        
    # Prepare agent histories for parallel calls
    agent_histories = {}
    for model_name in selected_models:
        if 'process_log' in st.session_state:
            st.session_state.process_log.append(f"Preparing call to agent: {model_name}")
            
        if agent_history and model_name in agent_history:
            agent_histories[model_name] = agent_history[model_name]
            if 'process_log' in st.session_state:
                st.session_state.process_log.append(f"  • Using conversation history with {len(agent_history[model_name])} messages")
    
    # Make parallel API calls to all models
    try:
        from utils import call_agents_parallel
        parallel_results = call_agents_parallel(selected_models, model_roles, query, openrouter_models, agent_histories)
        
        # Process all results
        for model_name, response_tuple in parallel_results.items():
            try:
                # Unpack the response
                if isinstance(response_tuple, tuple) and len(response_tuple) == 2:
                    response, metadata = response_tuple
                    
                    # Store the response
                    agent_responses[model_name] = response
                    
                    # Store in session state for UI display
                    if 'agent_responses' in st.session_state:
                        st.session_state.agent_responses[model_name] = response
                        if model_name not in st.session_state.selected_agents:
                            st.session_state.selected_agents.append(model_name)
                    
                    # Update agent conversation history
                    agent_conversation = agent_histories.get(model_name, [])
                    
                    # Add the current exchange to the conversation history
                    role = model_roles.get(model_name, "You are a general-purpose assistant.")
                    agent_conversation.append({"role": "system", "content": role})
                    agent_conversation.append({"role": "user", "content": query})
                    agent_conversation.append({"role": "assistant", "content": response})
                    
                    # Update the agent history
                    updated_agent_history[model_name] = agent_conversation
                    
                    # Store usage statistics
                    usage_data["models"][model_name] = metadata
                    usage_data["total_tokens"] += metadata["tokens"]["total"]
                    usage_data["total_cost"] += metadata["cost"]
                    usage_data["total_time"] += max(metadata["time"], usage_data["total_time"])  # Use max time instead of sum
                    
                    # Log usage data
                    if 'process_log' in st.session_state:
                        token_info = metadata["tokens"]
                        st.session_state.process_log.append(
                            f"  • {model_name} usage: {token_info['prompt']} prompt + {token_info['completion']} completion = {token_info['total']} tokens")
                        if metadata["cost"] > 0:
                            # Türkçe formatta virgül kullanarak göster
                            # En fazla 6 ondalık basamak gösterelim, gereksiz 0'lar olmasın
                            cost_str = f"{metadata['cost']:.6f}".rstrip('0').rstrip('.').replace(".", ",")
                            st.session_state.process_log.append(f"  • Estimated cost: ${cost_str}/1M tokens")
                else:
                    # Old format fallback
                    agent_responses[model_name] = response_tuple
                    if 'agent_responses' in st.session_state:
                        st.session_state.agent_responses[model_name] = response_tuple
            
            except ValueError as e:
                error_str = str(e)
                logger.error(f"ValueError processing response from {model_name}: {error_str}")
                
                # Special handling for known provider errors, particularly with sao10k models
                if "PROVIDER_SPECIFIC_ERROR" in error_str and "sao10k" in model_name:
                    logger.warning(f"Detected provider issue with sao10k model {model_name}, trying alternative")
                    
                    # Try to use a known-good alternative from the same provider
                    alt_model_name = "sao10k/l3-lunaris-8b"  # Our reliable fallback model
                    
                    if 'process_log' in st.session_state:
                        st.session_state.process_log.append(f"⚠️ Provider issue with {model_name}, trying {alt_model_name} instead")
                    
                    try:
                        # Import call_agent here to fix potential reference error
                        from utils import call_agent
                        # Use the new version that returns metadata
                        alt_response_tuple = call_agent(alt_model_name, model_roles.get(model_name, "You are a general-purpose assistant."), query, openrouter_models)
                        
                        # Unpack the response
                        if isinstance(alt_response_tuple, tuple) and len(alt_response_tuple) == 2:
                            alt_response, alt_metadata = alt_response_tuple
                            fallback_name = f"{alt_model_name} (fallback)"
                            
                            # Store the response
                            agent_responses[fallback_name] = alt_response
                            
                            # Store in session state for UI display
                            if 'agent_responses' in st.session_state:
                                st.session_state.agent_responses[fallback_name] = alt_response
                                if alt_model_name not in st.session_state.selected_agents:
                                    st.session_state.selected_agents.append(alt_model_name)
                            
                            # Update agent conversation history for the fallback model
                            alt_conversation = []
                            
                            # Add the current exchange to the conversation history
                            role = model_roles.get(model_name, "You are a general-purpose assistant.")
                            alt_conversation.append({"role": "system", "content": role})
                            alt_conversation.append({"role": "user", "content": query})
                            alt_conversation.append({"role": "assistant", "content": alt_response})
                            
                            # Update the agent history
                            updated_agent_history[alt_model_name] = alt_conversation
                            
                            # Store usage statistics
                            usage_data["models"][fallback_name] = alt_metadata
                            usage_data["total_tokens"] += alt_metadata["tokens"]["total"] 
                            usage_data["total_cost"] += alt_metadata["cost"]
                            usage_data["total_time"] = max(usage_data["total_time"], alt_metadata["time"])
                        else:
                            # Old format fallback
                            fallback_name = f"{alt_model_name} (fallback)"
                            agent_responses[fallback_name] = alt_response_tuple
                            if 'agent_responses' in st.session_state:
                                st.session_state.agent_responses[fallback_name] = alt_response_tuple
                            
                    except Exception as alt_e:
                        logger.error(f"Alternative model also failed: {alt_model_name} - {str(alt_e)}")
                        if 'process_log' in st.session_state:
                            st.session_state.process_log.append(f"❌ Fallback model {alt_model_name} also failed")
                else:
                    # Log other errors
                    if 'process_log' in st.session_state:
                        st.session_state.process_log.append(f"❌ Error with {model_name}: {error_str}")
                        
            except Exception as e:
                logger.error(f"Exception processing result for {model_name}: {str(e)}")
                if 'process_log' in st.session_state:
                    st.session_state.process_log.append(f"❌ Error with {model_name}: {str(e)}")
                    
    except Exception as e:
        logger.error(f"Error in parallel API calls: {str(e)}")
        if 'process_log' in st.session_state:
            st.session_state.process_log.append(f"❌ Error in parallel API calls: {str(e)}")
            
        # Fallback to sequential execution if parallel fails
        if 'process_log' in st.session_state:
            st.session_state.process_log.append("⚠️ Falling back to sequential API calls")
            
        # Call each agent sequentially (original implementation)
        for model_name in selected_models:
            role = model_roles.get(model_name, "You are a general-purpose assistant.")
            if 'process_log' in st.session_state:
                st.session_state.process_log.append(f"Calling agent: {model_name}")
            
            agent_conversation = agent_histories.get(model_name, None)
            
            try:
                from utils import call_agent
                response_tuple = call_agent(model_name, role, query, openrouter_models, conversation_history=agent_conversation)
                
                # Process response using same logic as above
                if isinstance(response_tuple, tuple) and len(response_tuple) == 2:
                    response, metadata = response_tuple
                    agent_responses[model_name] = response
                    
                    if 'agent_responses' in st.session_state:
                        st.session_state.agent_responses[model_name] = response
                        if model_name not in st.session_state.selected_agents:
                            st.session_state.selected_agents.append(model_name)
                    
                    if agent_conversation is None:
                        agent_conversation = []
                    
                    agent_conversation.append({"role": "system", "content": role})
                    agent_conversation.append({"role": "user", "content": query})
                    agent_conversation.append({"role": "assistant", "content": response})
                    
                    updated_agent_history[model_name] = agent_conversation
                    
                    usage_data["models"][model_name] = metadata
                    usage_data["total_tokens"] += metadata["tokens"]["total"]
                    usage_data["total_cost"] += metadata["cost"]
                    usage_data["total_time"] += metadata["time"]  # Sum for sequential
                    
                    if 'process_log' in st.session_state:
                        token_info = metadata["tokens"]
                        st.session_state.process_log.append(
                            f"  • {model_name} usage: {token_info['prompt']} prompt + {token_info['completion']} completion = {token_info['total']} tokens")
                else:
                    agent_responses[model_name] = response_tuple
                    if 'agent_responses' in st.session_state:
                        st.session_state.agent_responses[model_name] = response_tuple
                
            except Exception as model_e:
                logger.error(f"Exception in sequential fallback for {model_name}: {str(model_e)}")
                if 'process_log' in st.session_state:
                    st.session_state.process_log.append(f"❌ Error with {model_name}: {str(model_e)}")
    
    # Store usage data in session state
    if 'usage_data' not in st.session_state:
        st.session_state.usage_data = {}
    st.session_state.usage_data = usage_data
    
    # Log total usage summary
    if 'process_log' in st.session_state:
        if usage_data['total_cost'] > 0:
            # Türkçe formatta virgül kullanarak göster
            # En fazla 6 ondalık basamak gösterelim, gereksiz 0'lar olmasın
            cost_str = f"{usage_data['total_cost']:.6f}".rstrip('0').rstrip('.').replace(".", ",")
            cost_info = f"${cost_str}/1M tokens"
        else:
            cost_info = "Free"
            
        st.session_state.process_log.append(f"📊 Total usage: {usage_data['total_tokens']} tokens, {cost_info}, {usage_data['total_time']:.2f} seconds")
    
    # Special handling for code/math queries - check for conflicts
    # Also handle agent error scenarios by skipping conflict resolution if there are errors
    has_errors = any("Error:" in str(response) or "unavailable" in str(response) or "rate limit" in str(response).lower() 
                    for response in agent_responses.values())
    
    if not has_errors and ("code_expert" in labels or "math_expert" in labels) and len(selected_models) > 1:
        responses = list(agent_responses.values())
        if responses:
            first_response = responses[0]
            for resp in responses[1:]:
                similarity_score = calculate_similarity(first_response, resp)
                
                if 'process_log' in st.session_state:
                    st.session_state.process_log.append(f"Similarity check: score={similarity_score:.2f}")
                
                if similarity_score < 0.7:  # Threshold for significant difference
                    if 'process_log' in st.session_state:
                        st.session_state.process_log.append("Conflict detected between responses. Calling tiebreaker.")
                    
                    # Call a tiebreaker model with reasoning expertise
                    tiebreaker_models = get_models_by_labels(["reasoning_expert"], option, 
                                                          openrouter_models, min_models=1, max_models=1)
                    if tiebreaker_models:
                        tiebreaker_model = tiebreaker_models[0]
                        if 'process_log' in st.session_state:
                            st.session_state.process_log.append(f"Selected tiebreaker: {tiebreaker_model}")
                        
                        tiebreaker_role = get_model_roles([tiebreaker_model], ["reasoning_expert"])[tiebreaker_model]
                        tiebreaker_prompt = f"""These are conflicting answers from other agents:
                        
                        Agent 1: {responses[0]}
                        
                        Agent 2: {responses[1]}
                        
                        Original Query: {query}
                        
                        Determine which answer is correct or provide a better alternative.
                        Always respond in the same language as the original user query.
                        """
                        
                        if 'process_log' in st.session_state:
                            st.session_state.process_log.append(f"Calling tiebreaker agent: {tiebreaker_model}")
                        
                        # Ensure call_agent is properly imported here
                        from utils import call_agent
                        tiebreaker_response_tuple = call_agent(tiebreaker_model, tiebreaker_role, tiebreaker_prompt, openrouter_models)
                        
                        # Handle the tuple return format for tiebreaker
                        tiebreaker_response = None
                        if isinstance(tiebreaker_response_tuple, tuple) and len(tiebreaker_response_tuple) == 2:
                            tiebreaker_response, tiebreaker_metadata = tiebreaker_response_tuple
                            
                            # Add tiebreaker usage to the total
                            usage_data["models"][tiebreaker_model + " (tiebreaker)"] = tiebreaker_metadata
                            usage_data["total_tokens"] += tiebreaker_metadata["tokens"]["total"]
                            usage_data["total_cost"] += tiebreaker_metadata["cost"]
                            usage_data["total_time"] += tiebreaker_metadata["time"]
                            
                            # Log usage
                            if 'process_log' in st.session_state:
                                token_info = tiebreaker_metadata["tokens"]
                                st.session_state.process_log.append(
                                    f"  • Tiebreaker usage: {token_info['prompt']} prompt + {token_info['completion']} completion = {token_info['total']} tokens")
                        else:
                            # Backward compatibility
                            tiebreaker_response = tiebreaker_response_tuple
                        
                        if tiebreaker_response:
                            agent_responses[tiebreaker_model + " (tiebreaker)"] = tiebreaker_response
                            
                            # Store in session state for UI display
                            if 'agent_responses' in st.session_state:
                                st.session_state.agent_responses[tiebreaker_model + " (tiebreaker)"] = tiebreaker_response
                                st.session_state.selected_agents.append(tiebreaker_model)
                            
                            # Update agent conversation history for tiebreaker
                            tiebreaker_conversation = []
                            tiebreaker_key = tiebreaker_model + " (tiebreaker)"
                            
                            # Add the current exchange to the conversation history
                            tiebreaker_conversation.append({"role": "system", "content": tiebreaker_role})
                            tiebreaker_conversation.append({"role": "user", "content": tiebreaker_prompt})
                            tiebreaker_conversation.append({"role": "assistant", "content": tiebreaker_response})
                            
                            # Update the agent history
                            updated_agent_history[tiebreaker_key] = tiebreaker_conversation
                            
                            break
    
    # Generate final consolidated answer
    if agent_responses:
        combined_input = f"Original Query: {query}\n\n"
        for agent, response in agent_responses.items():
            combined_input += f"Agent {agent}: {response}\n\n"
        combined_input += """Based on these responses, provide a single, concise, consolidated answer.

Follow these guidelines:
1. Be very direct and to the point - users value brevity.
2. Prioritize the most relevant information first.
3. Maintain the structured format from agent responses when appropriate.
4. DO NOT use LaTeX formatting (like \\boxed{}, \\frac{}, \\sqrt{}, etc.).
5. Present math formulas in plain text format (use * for multiplication, / for division).
6. Use markdown boldface (e.g., **12**) for highlighting instead of \\boxed{12}.
7. Use bullet points for multiple items rather than paragraphs.
8. Skip pleasantries and unnecessary explanations.
9. Always respond in the same language as the original user query."""
        
        # Store the coordinator messages for UI display
        if 'coordinator_messages' in st.session_state:
            st.session_state.coordinator_messages = combined_input
            st.session_state.process_log.append(f"Synthesizing final answer with coordinator: {coordinator_model}")
        
        try:
            # Call the coordinator for final synthesis
            # Ensure call_agent is properly imported here
            from utils import call_agent
            # Explicitly mark that this is a coordinator model call
            response_tuple = call_agent(coordinator_model, 
                                     "You synthesize information from multiple sources.", 
                                     combined_input, openrouter_models,
                                     is_coordinator=True)
            
            # Handle the tuple return format
            final_answer = None
            coordinator_metadata = {}
            
            # Check for specific OpenRouter error format we added to utils.py
            if isinstance(response_tuple, tuple) and len(response_tuple) == 2:
                response, coordinator_metadata = response_tuple
                
                # Check for OpenRouter error
                if isinstance(response, str) and response.startswith("Error: OPENROUTER_ERROR_"):
                    error_parts = response.split(":")
                    if len(error_parts) >= 3:
                        error_code = error_parts[1].strip().split("_")[2]
                        error_message = ":".join(error_parts[2:]).strip()
                        logger.error(f"Coordinator synthesis received OpenRouter error {error_code}: {error_message}")
                        
                        # Pass the error upward with specific error code for better handling
                        raise ValueError(f"OpenRouter error {error_code}: {error_message}")
                
                # Regular processing for successful response
                final_answer = response
                
                # Add coordinator usage to the total
                usage_data["models"]["coordinator"] = coordinator_metadata
                usage_data["total_tokens"] += coordinator_metadata["tokens"]["total"]
                usage_data["total_cost"] += coordinator_metadata["cost"]
                usage_data["total_time"] += coordinator_metadata["time"]
                
                # Log the coordinator usage
                if 'process_log' in st.session_state:
                    token_info = coordinator_metadata["tokens"]
                    st.session_state.process_log.append(
                        f"  • Coordinator usage: {token_info['prompt']} prompt + {token_info['completion']} completion = {token_info['total']} tokens")
                    if coordinator_metadata["cost"] > 0:
                        # Türkçe formatta virgül kullanarak göster
                        # En fazla 6 ondalık basamak gösterelim, gereksiz 0'lar olmasın
                        cost_str = f"{coordinator_metadata['cost']:.6f}".rstrip('0').rstrip('.').replace(".", ",")
                        st.session_state.process_log.append(f"  • Coordinator cost: ${cost_str}/1M tokens")
            else:
                # Backward compatibility
                final_answer = response_tuple
                
        except Exception as e:
            error_str = str(e).lower()
            # Check for specific OpenRouter error codes
            if any(err in error_str for err in ["400", "401", "402", "403", "408", "429", "502", "503"]):
                logger.error(f"Coordinator synthesis error: {error_str}")
                
                # Try to find the last used coordinator model from history if available
                recent_coordinator_models = []
                try:
                    import os
                    import json
                    coord_history_file = "data/coordinator_history.json"
                    if os.path.exists(coord_history_file):
                        with open(coord_history_file, 'r') as f:
                            data = json.load(f)
                            recent_coordinator_models = data.get('models', [])
                except Exception as history_err:
                    logger.error(f"Failed to load coordinator history: {str(history_err)}")
                
                # Find an alternative model that is different from the current one
                alt_model = None
                if recent_coordinator_models:
                    for model in recent_coordinator_models:
                        if model != coordinator_model:
                            alt_model = model
                            break
                
                # Provide informative error message
                error_msg = f"Coordinator synthesis error: {str(e)}. "
                if alt_model:
                    error_msg += f"Try using alternative coordinator model: {alt_model}"
                else:
                    # Hardcoded fallback if we can't find an alternative in history
                    alt_model = "google/gemini-2.0-pro-exp-02-05:free"
                    error_msg += f"Try using alternative coordinator model: {alt_model}"
                
                # Raise the error to be handled in the app.py
                raise ValueError(error_msg)
            else:
                # For other errors, respond with a simpler fallback
                logger.warning(f"Coordinator synthesis error: {str(e)}, using fallback response")
                
                # Check for rate limiting errors in agent responses
                rate_limited = False
                for response in agent_responses.values():
                    if isinstance(response, str) and ("Rate limit exceeded" in response or "429" in response):
                        rate_limited = True
                        break
                
                if rate_limited:
                    # If rate limiting was detected, provide more helpful message
                    logger.warning("Detected rate limiting in individual agent responses, providing a specific message")
                    final_answer = (
                        "It appears we've hit rate limits with some of our API providers. "
                        "This typically happens when making many requests in a short period. "
                        "Here are the agent responses we did receive:\n\n"
                    )
                else:
                    # Standard error synthesis
                    final_answer = f"Error synthesizing responses. Here are the individual agent responses:\n\n"
                
                # Include all agent responses we received
                for agent, response in agent_responses.items():
                    final_answer += f"--- {agent} ---\n{response}\n\n"
        
        if 'process_log' in st.session_state:
            st.session_state.process_log.append("Final answer synthesized successfully")
        
        # Pass session_state for more detailed logging
        import streamlit as st
        log_conversation(combined_input, agent_responses, session_state=st.session_state if 'st' in globals() else None)
        
        # Store usage data in session state
        if 'usage_data' in st.session_state:
            st.session_state.usage_data = usage_data
            
        return final_answer, updated_agent_history
    
    return "No responses were received from the agents.", updated_agent_history

def process_query(query, coordinator_model, option, openrouter_models, coordinator_history=None, agent_history=None):
    """
    Main function to process a user query through the multi-agent system.
    
    Args:
        query: The user query to process
        coordinator_model: The model to use for coordination
        option: Selection strategy (free, paid, optimized)
        openrouter_models: List of available models
        coordinator_history: Previous conversation with coordinator (optional)
        agent_history: Previous conversations with agents (optional)
    
    Returns:
        Tuple of (final_answer, labels, updated_histories)
    """
    # Updated histories to return
    updated_histories = {
        'coordinator': [],
        'agents': {}
    }
    
    # Initialize with existing history if provided
    if coordinator_history:
        updated_histories['coordinator'] = coordinator_history.copy()
    if agent_history:
        updated_histories['agents'] = agent_history.copy()
    
    # Track if the coordinator model encounters any errors
    coordinator_error_encountered = False
    error_message = None
    
    try:
        # Step 1: Determine appropriate labels for the query
        labels = determine_query_labels(query, coordinator_model, openrouter_models, 
                                       coordinator_history=coordinator_history)
        
        if not labels:
            logger.warning("No labels determined for query")
            labels = ["general_assistant"]  # Fallback to general_assistant label
        
        logger.info(f"Query labels: {labels}")
        
        # Step 2: Coordinate the agents to generate a response
        final_answer, updated_agent_history = coordinate_agents(
            query, coordinator_model, labels, openrouter_models, option,
            agent_history=agent_history
        )
        
    except ValueError as e:
        error_str = str(e).lower()
        # Check if this is a coordinator API error that we want to handle
        if "openrouter error" in error_str or "coordinator model error" in error_str or "coordinator synthesis error" in error_str:
            # This is a critical coordinator error that should be passed upward
            coordinator_error_encountered = True
            error_message = str(e)
            
            # Construct a basic response with the error information
            labels = ["general_assistant"]  # Use default labels
            final_answer = f"Error: {error_message}"
            updated_agent_history = agent_history.copy() if agent_history else {}
            
            # Clearly show this as an error to the caller
            raise ValueError(error_message)
        else:
            # For other ValueErrors, use a fallback approach
            logger.error(f"Unexpected ValueError in process_query: {str(e)}")
            labels = ["general_assistant"]
            final_answer = f"An unexpected error occurred: {str(e)}"
            updated_agent_history = agent_history.copy() if agent_history else {}
    
    except Exception as e:
        # Handle other unexpected errors
        logger.error(f"Unexpected error in process_query: {str(e)}")
        labels = ["general_assistant"]
        final_answer = f"An unexpected error occurred: {str(e)}"
        updated_agent_history = agent_history.copy() if agent_history else {}
    
    # If there was a coordinator error, don't update the histories
    # and just return the error message, keeping previous history intact
    if coordinator_error_encountered:
        return final_answer, labels, updated_histories
    
    # A completely fresh approach to conversation history
    # Start with a clean slate but preserve existing history if provided
    if coordinator_history:
        # In old version, we accumulated everything, but we need clean state
        # Copy the previous history directly to our updated history
        for msg in coordinator_history:
            updated_histories['coordinator'].append(msg)
        
        # Log the history we're working with
        logger.info(f"Starting with existing history: {len(coordinator_history)} messages")
    
    # Now add the new interaction to the history
    # First add the query as user message
    updated_histories['coordinator'].append({
        "role": "user", 
        "content": query
    })
    
    # Then add the assistant response - but simplified format to make it clearer for the model
    updated_histories['coordinator'].append({
        "role": "assistant", 
        "content": final_answer
    })
    
    # Log what we did
    history_len = len(updated_histories['coordinator'])
    logger.info(f"Updated history now has {history_len} messages ({history_len//2} turns)")
    logger.info(f"Last message: {updated_histories['coordinator'][-1]['content'][:50]}...")
    
    # Update agent history with the new conversation turns
    if agent_history is not None:
        updated_histories['agents'] = updated_agent_history
    
    return final_answer, labels, updated_histories