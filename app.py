import streamlit as st
import logging
import time
import re
from utils import get_openrouter_models, handle_error
from coordinator import process_query

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pastel colors for UI elements
PASTEL_BLUE = "#E6F0FA"
PASTEL_GREEN = "#E6FAF0"
PASTEL_YELLOW = "#FAF8E6"
PASTEL_PINK = "#FAE6F0"
PASTEL_GRAY = "#F5F6F5"

# Apply custom styling to Streamlit interface
st.markdown(f"""
    <style>
        .main {{background-color: {PASTEL_GRAY};}}
        .stButton>button {{background-color: {PASTEL_BLUE}; color: black;}}
        .stSelectbox {{background-color: white;}}
        .stRadio {{background-color: {PASTEL_YELLOW};}}
        .stTextArea {{background-color: white;}}
        .stProgress .st-eb {{background-color: {PASTEL_GREEN};}}
    </style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    # Models and general app state
    if 'openrouter_models' not in st.session_state:
        st.session_state.openrouter_models = []
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'recent_coordinator_models' not in st.session_state:
        st.session_state.recent_coordinator_models = []
        
    # Processing state - ESSENTIAL for query handling
    if 'is_ready_to_process' not in st.session_state:
        st.session_state.is_ready_to_process = False
    
    # Response tracking
    if 'selected_agents' not in st.session_state:
        st.session_state.selected_agents = []
    if 'agent_responses' not in st.session_state:
        st.session_state.agent_responses = {}
    if 'coordinator_messages' not in st.session_state:
        st.session_state.coordinator_messages = ""
    if 'process_log' not in st.session_state:
        st.session_state.process_log = []
        
    # Conversation context history
    if 'agent_chat_history' not in st.session_state:
        st.session_state.agent_chat_history = {}
    if 'coordinator_chat_history' not in st.session_state:
        st.session_state.coordinator_chat_history = []
        
    # Current query
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    
    # UI state
    if 'need_rerun' not in st.session_state:
        st.session_state.need_rerun = False
        
    # Action tracking to prevent cross-widget interactions
    if 'last_action' not in st.session_state:
        st.session_state.last_action = None
        
    # Initialize button states
    if 'refresh_button_clicked' not in st.session_state:
        st.session_state.refresh_button_clicked = False
    if 'process_button_clicked' not in st.session_state:
        st.session_state.process_button_clicked = False

def main():
    """Main function to run the Streamlit app."""
    st.title("Multi_Agent System")
    st.subheader("Intelligent query processing with multiple AI models")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Initialize list to track eligible coordinator models
        if 'coordinator_models' not in st.session_state:
            st.session_state.coordinator_models = []
            
        # Add a section for recent coordinator models
        st.subheader("Recent Coordinator Models")
        if st.session_state.recent_coordinator_models:
            # Create a container for the quick access buttons
            recent_models_container = st.container()
            with recent_models_container:
                # Display last 5 used coordinator models as clickable buttons
                cols = st.columns(1)
                with cols[0]:
                    for model in st.session_state.recent_coordinator_models[-5:]:
                        if st.button(f"▶ {model.split('/')[-1]}", key=f"recent_{model}"):
                            # Set the coordinator model to this model
                            st.session_state.selected_coordinator = model
                            st.rerun()
        else:
            st.info("No recent coordinator models yet.")
            
        # Butonlar için satır oluştur
        col1, col2 = st.columns(2)
        
        # Coordinator modellerini yenile butonu
        with col1:
            if st.button("Refresh Coordinator Models"):
                with st.spinner("Fetching models from OpenRouter..."):
                    # Get fresh models from OpenRouter
                    st.session_state.openrouter_models = get_openrouter_models()
                    
                    # Define base trusted models for coordination
                    base_coordinator_models = [
                        "anthropic/claude-3.5-sonnet", 
                        "qwen/qwen-max",
                        "mistralai/mistral-large-2411",
                        "openai/gpt-4o-mini",
                        "anthropic/claude-3.5-haiku", 
                        "anthropic/claude-3-sonnet", 
                        "openai/gpt-4o",
                        "openai/gpt-3.5-turbo",
                        "mistralai/mistral-small-24b-instruct-2501:free",
                        "google/gemini-2.0-pro-exp-02-05:free"  # Moved to end due to potential issues
                    ]
                    
                    # Reset coordinator models list
                    st.session_state.coordinator_models = base_coordinator_models.copy()
                    
                    # Add all OpenRouter models to the coordinator list
                    if st.session_state.openrouter_models:
                        for model in st.session_state.openrouter_models:
                            model_id = model.get('id', '')
                            if model_id and model_id not in st.session_state.coordinator_models:
                                st.session_state.coordinator_models.append(model_id)
                    
                    # Show success message
                    st.success(f"Found {len(st.session_state.coordinator_models)} available models")
        
        # Tek bir güncelleme butonu
        if st.button("Update Model Data", type="primary"):
            with st.spinner("Updating models and roles from OpenRouter..."):
                try:
                    # External Python script aracılığıyla model etiketlerini ve rollerini güncelle
                    import subprocess
                    import sys
                    import tempfile
                    
                    # Model etiketlerini güncelle
                    labels_result = subprocess.run(
                        [sys.executable, "update_model_labels.py"],
                        capture_output=True,
                        text=True
                    )
                    
                    # Sonra model rollerini güncelle (sıralama önemli)
                    roles_result = subprocess.run(
                        [sys.executable, "update_model_roles.py"],
                        capture_output=True,
                        text=True
                    )
                    
                    # Her iki güncelleme de başarılı mıydı?
                    if labels_result.returncode == 0 and roles_result.returncode == 0:
                        # Model bilgilerini çıkart
                        # Etiket güncelleme bilgileri
                        # Debug çıktısını görüntüle
                        logger.info(f"Labels output: {labels_result.stdout}")
                        logger.info(f"Roles output: {roles_result.stdout}")
                        
                        # Türkçe log mesajlarıyla eşleşecek regex'ler
                        total_match = re.search(r'(\d+) toplam model', labels_result.stdout)
                        new_match = re.search(r'(\d+) yeni model', labels_result.stdout)
                        updated_match = re.search(r'(\d+) mevcut model', labels_result.stdout)
                        
                        total = total_match.group(1) if total_match else str(len(get_openrouter_models()))
                        new = new_match.group(1) if new_match else "0"
                        updated = updated_match.group(1) if updated_match else "0"
                        
                        # Rol güncelleme bilgileri
                        labels_match = re.search(r'(\d+) etiket tanımı', roles_result.stdout)
                        roles_match = re.search(r'(\d+) rol promptu', roles_result.stdout)
                        
                        # Eğer eşleşme bulunamazsa, en azından bir değer göster
                        labels = labels_match.group(1) if labels_match else str(len(get_openrouter_models()))
                        roles = roles_match.group(1) if roles_match else str(len(get_openrouter_models()))
                        
                        # Başarılı güncelleme mesajı
                        st.success(f"""✅ Model data successfully updated:
                        - {total} models total ({new} new, {updated} updated)
                        - {labels} label definitions
                        - {roles} role prompts""")
                        
                        # Otomatik olarak OpenRouter modellerini yenile
                        st.session_state.openrouter_models = get_openrouter_models()
                    else:
                        # Hata durumu
                        if labels_result.returncode != 0:
                            st.error(f"❌ Error updating model labels: {labels_result.stderr}")
                        if roles_result.returncode != 0:
                            st.error(f"❌ Error updating model roles: {roles_result.stderr}")
                except Exception as e:
                    st.error(f"❌ Failed to update model data: {str(e)}")
        
        # Show note about coordinator models (only once)
        if st.session_state.coordinator_models:
            # Message already shown in success message
            pass
        else:
            # Initial set of coordinator models
            st.session_state.coordinator_models = [
                "anthropic/claude-3.5-sonnet", 
                "qwen/qwen-max",
                "mistralai/mistral-large-2411",
                "openai/gpt-4o-mini",
                "anthropic/claude-3.5-haiku", 
                "anthropic/claude-3-sonnet", 
                "openai/gpt-4o",
                "openai/gpt-3.5-turbo",
                "mistralai/mistral-small-24b-instruct-2501:free",
                "google/gemini-2.0-pro-exp-02-05:free"  # Moved to end due to potential issues
            ]
            st.info("Click 'Refresh Coordinator Models' to see all available models from OpenRouter")
        
        # Initialize selected_coordinator if it doesn't exist
        if 'selected_coordinator' not in st.session_state:
            st.session_state.selected_coordinator = st.session_state.coordinator_models[0] if st.session_state.coordinator_models else None
        
        # Initialize model filter
        if 'model_filter' not in st.session_state:
            st.session_state.model_filter = ""
            
        # Filter input with clear button
        st.write("**Select Coordinator Model**")
        
        filter_col1, filter_col2 = st.columns([5, 1])
        with filter_col1:
            model_filter = st.text_input(
                "Filter models", 
                value=st.session_state.model_filter,
                label_visibility="collapsed", 
                key="model_filter_input"
            )
        with filter_col2:
            if st.button("✕", help="Clear filter"):
                st.session_state.model_filter = ""
                model_filter = ""
                st.rerun()
                
        # Update model filter in session state for persistence
        st.session_state.model_filter = model_filter
        
        # Filter models based on input - support multiple search terms
        if model_filter:
            # Split filter into individual terms
            filter_terms = model_filter.lower().split()
            
            # A model matches if ALL filter terms are found in it (AND logic)
            filtered_models = [model for model in st.session_state.coordinator_models 
                              if all(term in model.lower() for term in filter_terms)]
        else:
            # If no filter, show all models
            filtered_models = st.session_state.coordinator_models.copy()
        
        # If no models match filter, show all
        if not filtered_models:
            filtered_models = st.session_state.coordinator_models
            st.warning("No models match your filter. Showing all models.")
        
        # Select model from filtered list
        default_index = 0
        if st.session_state.selected_coordinator in filtered_models:
            default_index = filtered_models.index(st.session_state.selected_coordinator)
            
        coordinator_model = st.selectbox(
            "Select from filtered models",
            options=filtered_models,
            index=min(default_index, len(filtered_models)-1) if filtered_models else 0,
            label_visibility="collapsed"
        )
        
        # Update selected_coordinator and recent models list
        if coordinator_model != st.session_state.selected_coordinator:
            st.session_state.selected_coordinator = coordinator_model
            
            # Update recent models list - remove if exists, then add to end
            if coordinator_model in st.session_state.recent_coordinator_models:
                st.session_state.recent_coordinator_models.remove(coordinator_model)
            st.session_state.recent_coordinator_models.append(coordinator_model)
        
        # Option selection (free, paid, optimized)
        option = st.radio(
            "Model Selection Strategy",
            options=["free", "paid", "optimized"],
            index=0
        )
        
        # Completely different approach with a dedicated container and key
        refresh_container = st.container()
        
        # Define a specific function to ONLY refresh models
        def refresh_models_only():
            # Set action flag to prevent process button from activating
            st.session_state.last_action = "refresh_models"
            st.session_state.refresh_button_clicked = True
            st.session_state.process_button_clicked = False
            st.session_state.is_ready_to_process = False
            
        # Place button in container with specific key
        with refresh_container:
            refresh_btn = st.button(
                "Refresh Available Models", 
                on_click=refresh_models_only,
                key="unique_refresh_models_btn"
            )
        
        # Handle the refresh independently of the button click
        if st.session_state.refresh_button_clicked and st.session_state.last_action == "refresh_models":
            # Reset the flag immediately
            st.session_state.refresh_button_clicked = False
            
            # Do the actual refresh
            with st.spinner("Fetching models from OpenRouter..."):
                st.session_state.openrouter_models = get_openrouter_models()
                st.success(f"Found {len(st.session_state.openrouter_models)} models")
        
        # We don't need this now as we show success message in the refresh operation
        
        # Remove error log expander from sidebar
        
        # Show model info
        if st.session_state.openrouter_models:
            with st.expander("Available Models", expanded=False):
                for model in st.session_state.openrouter_models[:10]:  # Show first 10 models
                    st.write(f"- {model.get('id', 'Unknown model')}")
                if len(st.session_state.openrouter_models) > 10:
                    st.write(f"... and {len(st.session_state.openrouter_models) - 10} more")
    
    # Main area - Input and processing
    col1, col2 = st.columns([4, 1])
    
    # Simple text area for input
    with col1:
        # Override previous input value if stored
        if 'query_input' not in st.session_state:
            st.session_state.query_input = ""
            
        # Function to handle changes
        def on_text_change():
            if st.session_state.query_input.strip():
                # Just save the entered text, don't trigger any processing
                st.session_state.current_query = st.session_state.query_input
                
        # Simple text area
        st.text_area(
            "Enter your query:", 
            height=150, 
            key="query_input",
            on_change=on_text_change
        )
    
    with col2:
        # Reset button to clear current query results and conversation history
        if st.button("Reset Conversation", key="reset_button"):
            # Log the reset action
            logger.info("Reset button pressed - clearing all state")
            
            # Clear UI state
            st.session_state.selected_agents = []
            st.session_state.agent_responses = {}
            st.session_state.coordinator_messages = ""
            st.session_state.process_log = []
            
            # Clear conversation context
            st.session_state.agent_chat_history = {}
            st.session_state.coordinator_chat_history = []
            
            # Clear query state
            st.session_state.current_query = ""
            st.session_state.query_input = ""
            
            # Reset processing flags
            st.session_state.is_ready_to_process = False
            
            # Add a success message
            st.success("Conversation history cleared! You can start a new conversation.")
            
            # Force full UI refresh
            st.rerun()
            
        # Show minimal conversation history status - only if we have history and only as small text
        if st.session_state.coordinator_chat_history and len(st.session_state.coordinator_chat_history) > 0:
            # Calculate conversation turns
            msg_count = len(st.session_state.coordinator_chat_history) // 2
            if msg_count > 0:
                # Use a much smaller, less intrusive indicator
                st.caption(f"Conversation continues with {msg_count} previous turns")
    
    # Create a dedicated container for process button
    process_container = st.container()
    
    # Define function to ONLY set process flag
    def process_query_only():
        # Get the query text
        query = st.session_state.query_input
        
        # Only if we have something to process
        if query and query.strip():
            # Set action flag to prevent model refresh from happening
            st.session_state.last_action = "process_query"
            # Update state
            st.session_state.current_query = query
            st.session_state.process_button_clicked = True
            st.session_state.refresh_button_clicked = False
            st.session_state.is_ready_to_process = True
            # Log
            logger.info(f"Process button clicked for query: {query[:30]}...")
            # Clear input immediately
            st.session_state.query_input = ""
    
    # Create button in its own container
    with process_container:
        process_btn = st.button(
            "Process Query",
            on_click=process_query_only,
            type="primary",
            use_container_width=True,
            key="unique_process_btn"
        )
    
    # Handle empty query warnings separately
    if process_btn and not st.session_state.query_input.strip():
        st.warning("Please enter a query in the text box first.")
        if 'process_log' in st.session_state:
            st.session_state.process_log.append("⚠️ Empty query - processing skipped")
        
    # Display the current query if it exists
    if st.session_state.current_query:
        current_query_container = st.container()
        with current_query_container:
            st.markdown("**Current Query:**")
            st.markdown(f"```\n{st.session_state.current_query}\n```")
            st.markdown("---")
        
    # THE MAIN PROCESSING LOGIC - completely rewritten for reliability
    # Log the current state for debugging
    if 'process_log' in st.session_state:
        state_summary = f"Ready to process: {'Yes' if st.session_state.is_ready_to_process else 'No'}"
        state_summary += f", Current query: {'Yes' if st.session_state.current_query else 'No'}"
        st.session_state.process_log.append(f"DEBUG STATE: {state_summary}")
    
    # Only run processing if the process flag is set and we have a current query
    # Also check that last_action is process_query to prevent accidental processing
    if (st.session_state.is_ready_to_process and 
        st.session_state.current_query and 
        st.session_state.last_action == "process_query"):
        # Get the query from the session state
        query = st.session_state.current_query
        
        # Immediately clear the ready flag to prevent reprocessing
        st.session_state.is_ready_to_process = False
        
        # Log that we're starting processing
        logger.info(f"Beginning processing for query: {query[:50]}")
        if 'process_log' in st.session_state:
            st.session_state.process_log.append(f"🔄 Processing: {query[:50]}...")
        
        # Check if OpenRouter models are loaded
        if not st.session_state.openrouter_models:
            with st.spinner("Fetching models from OpenRouter..."):
                st.session_state.openrouter_models = get_openrouter_models()
        
        # Create placeholders for progress updates
        progress_bar = st.progress(0)
        status_text = st.empty()
        result_placeholder = st.empty()
        agent_info_placeholder = st.empty()
        
        success = False
        try:
            # Step 1: Analyze query (20%)
            status_text.text("Step 1/4: Analyzing query to determine appropriate labels...")
            progress_bar.progress(20)
            time.sleep(0.5)  # Simulated delay for visual feedback
            
            # Step 2: Selecting models (40%)
            status_text.text("Step 2/4: Selecting specialized models based on query labels...")
            progress_bar.progress(40)
            time.sleep(0.5)  # Simulated delay for visual feedback
            
            # Step 3: Processing with models (70%)
            status_text.text("Step 3/4: Querying selected models with appropriate roles...")
            progress_bar.progress(70)
            
            # Actual processing - always use conversation history
            coordinator_history = st.session_state.coordinator_chat_history 
            agent_history = st.session_state.agent_chat_history
            
            # Debug log
            if coordinator_history:
                logger.info(f"Using coordinator history with {len(coordinator_history)} messages")
                for i, msg in enumerate(coordinator_history):
                    role = msg.get('role', 'unknown')
                    content_preview = (msg.get('content', '')[:50] + '...') if len(msg.get('content', '')) > 50 else msg.get('content', '')
                    logger.info(f"History message {i}: {role} - {content_preview}")
            
            if agent_history:
                logger.info(f"Using agent history for {len(agent_history)} models")
                for model, history in agent_history.items():
                    logger.info(f"Agent {model}: {len(history)} messages")
            
            final_answer, labels, updated_histories = process_query(
                query, 
                coordinator_model, 
                option, 
                st.session_state.openrouter_models,
                coordinator_history=coordinator_history,
                agent_history=agent_history
            )
            
            # Always update conversation history
            st.session_state.coordinator_chat_history = updated_histories.get('coordinator', [])
            st.session_state.agent_chat_history = updated_histories.get('agents', {})
            
            # Add history info to process log
            if 'process_log' in st.session_state:
                n_coord = len(st.session_state.coordinator_chat_history)
                n_agents = len(st.session_state.agent_chat_history)
                n_turns = n_coord // 2  # Each turn is a user message + assistant response
                st.session_state.process_log.append(f"👉 Conversation history updated: {n_turns} turns ({n_coord} messages), {n_agents} agent models")
            
            # Step 4: Finalizing response (100%)
            status_text.text("Step 4/4: Synthesizing final response from all agents...")
            progress_bar.progress(100)
            
            # Display the result after sanitizing any LaTeX commands
            # Replace common LaTeX patterns with simple text alternatives
            sanitized_answer = final_answer
            
            # Common LaTeX commands to sanitize
            latex_replacements = [
                (r'\\boxed\{([^}]*)\}', r'**\1**'),  # Replace \boxed{x} with **x**
                (r'\$([^$]*)\$', r'\1'),              # Remove $ math delimiters
                (r'\\frac\{([^}]*)\}\{([^}]*)\}', r'\1/\2'),  # Replace \frac{a}{b} with a/b
                (r'\\sqrt\{([^}]*)\}', r'sqrt(\1)')   # Replace \sqrt{x} with sqrt(x)
            ]
            
            # Apply all replacements
            for pattern, replacement in latex_replacements:
                sanitized_answer = re.sub(pattern, replacement, sanitized_answer)
                
            # Display the sanitized answer
            result_placeholder.markdown(f"### Response\n{sanitized_answer}")
            
            # Mark as successful
            success = True
            
            # Move agent information inside an expander
            if st.session_state.selected_agents:
                # Ensure we count unique models only
                unique_models = list(set(st.session_state.selected_agents))
                
                with agent_info_placeholder.container():
                    with st.expander("Agent Information", expanded=False):
                        st.markdown(f"✅ **{len(unique_models)} unique models** were used to answer this query.")
                        
                        # More compact labels display
                        label_text = "**Labels:** "
                        label_text += ", ".join([f"`{label}`" for label in labels])
                        st.markdown(label_text)
                        
                        # Add usage statistics if available
                        if 'usage_data' in st.session_state:
                            usage = st.session_state.usage_data
                            st.markdown("**📊 Usage Statistics:**")
                            st.markdown(f"- Total tokens: **{usage['total_tokens']}**")
                            if usage['total_cost'] > 0:
                                st.markdown(f"- Estimated cost: **${usage['total_cost']:.4f}/1M tokens**")
                            else:
                                st.markdown(f"- Estimated cost: **Free**")
                            st.markdown(f"- Processing time: **{usage['total_time']:.2f}s**")
                        
                        # Simpler model display to avoid potential formatting issues
                        st.markdown("**Models:**")
                        # Display all unique models in a single list with usage if available
                        model_list = ""
                        for model in unique_models:
                            # Basic model info
                            model_line = f"• {model}"
                            
                            # Add usage info if available
                            if 'usage_data' in st.session_state and model in st.session_state.usage_data.get("models", {}):
                                model_data = st.session_state.usage_data["models"][model]
                                if "tokens" in model_data:
                                    tokens = model_data["tokens"]
                                    model_line += f" ({tokens.get('total', 0)} tokens"
                                    if model_data.get("cost", 0) > 0:
                                        cost = model_data['cost']
                                        model_line += f", ${cost:.4f}/1M"
                                    model_line += ")"
                            
                            model_list += model_line + "  \n"  # Two spaces for line break in markdown
                        st.markdown(model_list)
            
            # Add to conversation history
            history_entry = {
                "query": query,
                "answer": final_answer,
                "labels": labels,
                "models": st.session_state.selected_agents,
                "agent_responses": st.session_state.agent_responses.copy() if hasattr(st.session_state, 'agent_responses') else {}
            }
            
            # Add usage data if available
            if 'usage_data' in st.session_state:
                history_entry["usage_data"] = st.session_state.usage_data.copy()
                
            st.session_state.conversation_history.append(history_entry)
            
            # Clear status text and progress bar after 2 seconds
            time.sleep(2)
            status_text.empty()
            
        except Exception as e:
            # Create a more visible and detailed error display
            error_message = f"Error processing query: {str(e)}"
            error_str = str(e).lower()
            
            # Log the error
            logger.error(error_message)
            
            # Display a prominent error message
            st.error(f"⚠️ {error_message}")
            
            # Add to process log
            if 'process_log' in st.session_state:
                st.session_state.process_log.append(f"ERROR: {error_message}")
            
            # Check for coordinator model issues (rate limits, timeouts, etc)
            coordinator_error = False
            coordinator_needs_change = False
            
            # Specific error message handling
            if "rate limit" in error_str or "429" in error_str:
                coordinator_error = True
                coordinator_needs_change = True
                error_type = "Rate limit exceeded"
            elif "timeout" in error_str or "408" in error_str:
                coordinator_error = True
                coordinator_needs_change = True
                error_type = "Request timeout"
            elif "provider returned error" in error_str or "503" in error_str:
                coordinator_error = True
                coordinator_needs_change = True
                error_type = "Provider error"
            elif "insufficient credits" in error_str or "402" in error_str:
                coordinator_error = True
                coordinator_needs_change = True
                error_type = "Insufficient credits"
            elif "authentication" in error_str or "401" in error_str or "403" in error_str:
                coordinator_error = True
                error_type = "Authentication error"
            else:
                error_type = "Unexpected error"
                
            # Special handling for coordinator errors
            if coordinator_error:
                if coordinator_needs_change:
                    result_placeholder.warning(f"""
                    ### Coordinator Model Error: {error_type}
                    
                    The coordinator model `{coordinator_model}` returned an error.
                    
                    **Please select a different coordinator model** from the sidebar and try again.
                    
                    If this problem persists, try models from a different provider or wait a few minutes.
                    """)
                else:
                    result_placeholder.warning(f"""
                    ### Coordinator Error: {error_type}
                    
                    There was an error with the coordinator model `{coordinator_model}`.
                    
                    Please try again or select a different model if the problem persists.
                    """)
            else:
                # Generic error for other cases
                result_placeholder.warning("""
                Sorry, something went wrong while processing your query. 
                
                This could be due to:
                - Temporary API connection issues
                - Selected model unavailability
                - Rate limiting from OpenRouter
                
                Please try again or select different models/options.
                """)
        
        finally:
            # Log the completion of processing
            logger.info(f"Processing completed (success={success})")
            
            # We don't reset anything here - we've already reset the is_ready_to_process flag
            # This ensures we don't reprocess automatically but keep the query visible
            
            if success and 'process_log' in st.session_state:
                turns = len(st.session_state.coordinator_chat_history) // 2 if st.session_state.coordinator_chat_history else 0
                st.session_state.process_log.append(f"✅ Processing complete. Conversation now has {turns} turns.")

    # Add expanders for additional information (all default collapsed)
    if st.session_state.selected_agents:
        # Using expanders instead of tabs for better control
        
        # 1. Agent Responses Section (collapsed by default) - simplified
        with st.expander("Agent Responses", expanded=False):
            if st.session_state.agent_responses:
                # Show each unique agent's response (take last response if duplicate)
                displayed_agents = set()
                for agent, response in st.session_state.agent_responses.items():
                    # If we've already shown this agent's response, skip
                    if agent in displayed_agents:
                        continue
                    
                    displayed_agents.add(agent)
                    st.subheader(f"Response from {agent}")
                    # Add a unique key using the agent name to avoid duplicate widget key errors
                    st.text_area(f"Response:", response, height=200, key=f"response_{agent}")
                    st.markdown("---")
            else:
                st.info("No agent responses available.")
        
        # 2. Process Log Section (collapsed by default)
        with st.expander("Process Log", expanded=False):
            if st.session_state.process_log:
                st.subheader("Process Log")
                
                # Simplified process log display
                st.text("Process Log Steps:")
                for i, log_entry in enumerate(st.session_state.process_log):
                    # Simpler format with just numbers and text
                    st.text(f"Step {i+1}: {log_entry}")
            else:
                st.info("No process logs available.")
        
        # 3. Coordinator Messages Section (collapsed by default) - simplified
        with st.expander("Coordinator Communication", expanded=False):
            if st.session_state.coordinator_messages:
                # Just show the raw text for simplicity and to avoid parsing errors
                st.text_area("Coordinator Messages:", st.session_state.coordinator_messages, height=400)
            else:
                st.info("No coordinator messages available.")
    
    # Display conversation history with improved formatting
    if st.session_state.conversation_history:
        with st.expander("Conversation History", expanded=False):
            for i, item in enumerate(reversed(st.session_state.conversation_history)):
                # Create a container for each entry
                with st.container():
                    # Query with counting from newest to oldest
                    st.subheader(f"Query {len(st.session_state.conversation_history) - i}")
                    st.info(item['query'])
                    
                    # Create two columns for metadata
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Display labels as a list
                        st.markdown("**🏷️ Labels:**")
                        for label in item['labels']:
                            st.markdown(f"- {label}")
                        
                        # Add usage data if available
                        if 'usage_data' in item:
                            usage = item['usage_data']
                            st.markdown("**📊 Usage:**")
                            st.markdown(f"- {usage['total_tokens']} tokens")
                            if usage['total_cost'] > 0:
                                st.markdown(f"- ${usage['total_cost']:.4f}/1M tokens")
                            else:
                                st.markdown(f"- Free")
                            st.markdown(f"- {usage['total_time']:.2f}s")
                    
                    with col2:
                        # Display models used as a list
                        if 'models' in item and item['models']:
                            st.markdown("**🤖 Models used:**")
                            for model in item['models']:
                                model_text = f"- {model}"
                                # Add per-model usage if available
                                if 'usage_data' in item and model in item['usage_data'].get('models', {}):
                                    model_data = item['usage_data']['models'][model]
                                    if 'tokens' in model_data:
                                        model_text += f" ({model_data['tokens'].get('total', 0)} tokens)"
                                st.markdown(model_text)
                    
                    # Display the answer
                    st.markdown("**💬 Answer:**")
                    st.success(item['answer'])
                    
                    # Add a divider between entries
                    if i < len(st.session_state.conversation_history) - 1:
                        st.markdown("---")

    # About section removed as requested

if __name__ == "__main__":
    main()