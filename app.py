import streamlit as st
import logging
import time
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
    if 'openrouter_models' not in st.session_state:
        st.session_state.openrouter_models = []
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    if 'selected_agents' not in st.session_state:
        st.session_state.selected_agents = []
    if 'agent_responses' not in st.session_state:
        st.session_state.agent_responses = {}
    if 'coordinator_messages' not in st.session_state:
        st.session_state.coordinator_messages = ""
    if 'process_log' not in st.session_state:
        st.session_state.process_log = []

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
            
        # Button to refresh coordinator models
        if st.button("Refresh Coordinator Models"):
            with st.spinner("Fetching models from OpenRouter..."):
                # Get fresh models from OpenRouter
                st.session_state.openrouter_models = get_openrouter_models()
                
                # Define base trusted models for coordination
                base_coordinator_models = [
                    "anthropic/claude-3.5-haiku", 
                    "anthropic/claude-3.5-sonnet",
                    "anthropic/claude-3-sonnet", 
                    "openai/gpt-4o",
                    "openai/gpt-3.5-turbo",
                    "google/gemini-2.0-pro-exp-02-05:free",
                    "mistralai/mistral-small-24b-instruct-2501:free"
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
        
        # Show note about coordinator models (only once)
        if st.session_state.coordinator_models:
            # Message already shown in success message
            pass
        else:
            # Initial set of coordinator models
            st.session_state.coordinator_models = [
                "anthropic/claude-3.5-haiku", 
                "anthropic/claude-3.5-sonnet",
                "anthropic/claude-3-sonnet", 
                "openai/gpt-4o",
                "openai/gpt-3.5-turbo",
                "google/gemini-2.0-pro-exp-02-05:free",
                "mistralai/mistral-small-24b-instruct-2501:free"
            ]
            st.info("Click 'Refresh Coordinator Models' to see all available models from OpenRouter")
        
        # Select coordinator model
        coordinator_model = st.selectbox(
            "Select Coordinator Model",
            options=st.session_state.coordinator_models,
            index=0
        )
        
        # Option selection (free, paid, optimized)
        option = st.radio(
            "Model Selection Strategy",
            options=["free", "paid", "optimized"],
            index=0
        )
        
        # Refresh OpenRouter models button
        if st.button("Refresh Available Models"):
            with st.spinner("Fetching models from OpenRouter..."):
                st.session_state.openrouter_models = get_openrouter_models()
                st.success(f"Found {len(st.session_state.openrouter_models)} models")
        
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
    
    with col1:
        query = st.text_area("Enter your query:", height=150)
    
    with col2:
        # Reset button to clear current query results
        if st.button("Reset"):
            if 'selected_agents' in st.session_state:
                st.session_state.selected_agents = []
                st.session_state.agent_responses = {}
                st.session_state.coordinator_messages = ""
                st.session_state.process_log = []
            st.rerun()
    
    # Process button row with two columns
    col1, col2 = st.columns([4, 1])
    
    with col1:
        process_btn = st.button("Process Query", type="primary")
    
    # Process query when button is clicked
    if process_btn and query and not st.session_state.is_processing:
        st.session_state.is_processing = True
        
        # Check if OpenRouter models are loaded
        if not st.session_state.openrouter_models:
            with st.spinner("Fetching models from OpenRouter..."):
                st.session_state.openrouter_models = get_openrouter_models()
        
        # Create placeholders for progress updates
        progress_bar = st.progress(0)
        status_text = st.empty()
        result_placeholder = st.empty()
        agent_info_placeholder = st.empty()
        
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
            
            # Actual processing
            final_answer, labels = process_query(
                query, 
                coordinator_model, 
                option, 
                st.session_state.openrouter_models
            )
            
            # Step 4: Finalizing response (100%)
            status_text.text("Step 4/4: Synthesizing final response from all agents...")
            progress_bar.progress(100)
            
            # Display only the result
            result_placeholder.markdown(f"### Response\n{final_answer}")
            
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
                        
                        # Simpler model display to avoid potential formatting issues
                        st.markdown("**Models:**")
                        # Display all unique models in a single list
                        model_list = ""
                        for model in unique_models:
                            model_list += f"• {model}  \n"  # Two spaces for line break in markdown
                        st.markdown(model_list)
            
            # Add to conversation history
            st.session_state.conversation_history.append({
                "query": query,
                "answer": final_answer,
                "labels": labels,
                "models": st.session_state.selected_agents,
                "agent_responses": st.session_state.agent_responses.copy() if hasattr(st.session_state, 'agent_responses') else {}
            })
            
            # Clear status text and progress bar after 2 seconds
            time.sleep(2)
            status_text.empty()
            
        except Exception as e:
            # Create a more visible and detailed error display
            error_message = f"Error processing query: {str(e)}"
            
            # Log the error
            logger.error(error_message)
            
            # Display a prominent error message
            st.error(f"⚠️ {error_message}")
            
            # Add to process log
            if 'process_log' in st.session_state:
                st.session_state.process_log.append(f"ERROR: {error_message}")
            
            # More user-friendly details in the result area
            result_placeholder.warning("""
            Sorry, something went wrong while processing your query. 
            
            This could be due to:
            - Temporary API connection issues
            - Selected model unavailability
            - Rate limiting from OpenRouter
            
            Please try again or select different models/options.
            """)
        
        finally:
            st.session_state.is_processing = False

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
                    st.text_area(f"Response:", response, height=200)
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
                    
                    with col2:
                        # Display models used as a list
                        if 'models' in item and item['models']:
                            st.markdown("**🤖 Models used:**")
                            for model in item['models']:
                                st.markdown(f"- {model}")
                    
                    # Display the answer
                    st.markdown("**💬 Answer:**")
                    st.success(item['answer'])
                    
                    # Add a divider between entries
                    if i < len(st.session_state.conversation_history) - 1:
                        st.markdown("---")

    # About section removed as requested

if __name__ == "__main__":
    main()