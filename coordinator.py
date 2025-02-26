import re
import logging
from utils import load_json, call_agent, log_conversation
from agents import get_models_by_labels, get_model_roles, determine_complexity, calculate_similarity, get_labels_for_model

# Initialize logger
logger = logging.getLogger(__name__)

def determine_query_labels(query, coordinator_model, openrouter_models):
    """
    Analyzes the query using the coordinator model to identify appropriate labels.
    """
    # Load available labels from the model_roles.json file
    roles_data = load_json('data/model_roles.json')
    if not roles_data:
        logger.error("Could not load model roles data")
        return []
        
    available_labels = [label_entry['label'] for label_entry in roles_data['labels']]
    
    # Create a prompt asking the coordinator to determine relevant labels
    prompt = f"""You are a helpful assistant tasked with analyzing a user's query and identifying relevant labels.
    
    Predefined Labels: {', '.join(available_labels)}
    
    User Query: {query}
    
    Return ONLY the labels, separated by commas. No additional explanation.
    """
    
    # Ask the coordinator model to analyze the query
    response = call_agent(coordinator_model, "You are a helpful assistant.", prompt, openrouter_models)
    
    # Extract and validate the labels from the response
    if response and not response.startswith("Error"):
        extracted_labels = [label.strip().lower() for label in re.split(r'[,\s]+', response) if label.strip()]
        validated_labels = [label for label in extracted_labels if label in available_labels]
        return validated_labels
    
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

def coordinate_agents(query, coordinator_model, labels, openrouter_models, option):
    """
    Coordinates the selected agents, calls them with appropriate roles,
    and synthesizes a final response.
    """
    # Clear previous state if using Streamlit
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
        return "No suitable models were found to process this query."
    
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
    
    # Call each agent with its role
    for model_name in selected_models:
        role = model_roles.get(model_name, "You are a general-purpose assistant.")
        if 'process_log' in st.session_state:
            st.session_state.process_log.append(f"Calling agent: {model_name}")
        
        response = call_agent(model_name, role, query, openrouter_models)
        agent_responses[model_name] = response
        
        # Store in session state for UI display
        if 'agent_responses' in st.session_state:
            st.session_state.agent_responses[model_name] = response
    
    # Special handling for code/math queries - check for conflicts
    if ("code_expert" in labels or "math_expert" in labels) and len(selected_models) > 1:
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
                        """
                        
                        if 'process_log' in st.session_state:
                            st.session_state.process_log.append(f"Calling tiebreaker agent: {tiebreaker_model}")
                        
                        tiebreaker_response = call_agent(tiebreaker_model, tiebreaker_role, tiebreaker_prompt, openrouter_models)
                        if tiebreaker_response:
                            agent_responses[tiebreaker_model] = tiebreaker_response
                            
                            # Store in session state for UI display
                            if 'agent_responses' in st.session_state:
                                st.session_state.agent_responses[tiebreaker_model] = tiebreaker_response
                                st.session_state.selected_agents.append(tiebreaker_model)
                            break
    
    # Generate final consolidated answer
    if agent_responses:
        combined_input = f"Original Query: {query}\n\n"
        for agent, response in agent_responses.items():
            combined_input += f"Agent {agent}: {response}\n\n"
        combined_input += "Based on these responses, provide a single, consolidated answer."
        
        # Store the coordinator messages for UI display
        if 'coordinator_messages' in st.session_state:
            st.session_state.coordinator_messages = combined_input
            st.session_state.process_log.append(f"Synthesizing final answer with coordinator: {coordinator_model}")
        
        final_answer = call_agent(coordinator_model, 
                                 "You synthesize information from multiple sources.", 
                                 combined_input, openrouter_models)
        
        if 'process_log' in st.session_state:
            st.session_state.process_log.append("Final answer synthesized successfully")
        
        log_conversation(combined_input, agent_responses)
        return final_answer
    
    return "No responses were received from the agents."

def process_query(query, coordinator_model, option, openrouter_models):
    """
    Main function to process a user query through the multi-agent system.
    """
    # Step 1: Determine appropriate labels for the query
    labels = determine_query_labels(query, coordinator_model, openrouter_models)
    
    if not labels:
        logger.warning("No labels determined for query")
        labels = ["general_assistant"]  # Fallback to general_assistant label
    
    logger.info(f"Query labels: {labels}")
    
    # Step 2: Coordinate the agents to generate a response
    final_answer = coordinate_agents(query, coordinator_model, labels, openrouter_models, option)
    
    return final_answer, labels