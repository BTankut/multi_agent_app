# Multi_Agent System Project Documentation

## Project Overview

Multi_Agent System is an AI orchestration platform that intelligently routes user queries to specialized AI models based on content type and complexity. The system analyzes queries, selects appropriate specialized models from OpenRouter, dynamically manages the number of agents, and synthesizes their responses into coherent answers.

## GitHub Repository

- **URL**: https://github.com/BTankut/multi_agent_app
- **First Commit**: 2025-02-26
- **License**: MIT

## Core Components

- **Coordinator**: Analyzes queries and orchestrates communication between models
- **Agents**: Multiple specialized AI models chosen based on query needs
- **Web/CLI Interface**: Streamlit web app and CLI for interacting with the system

## Key Features

- **Intelligent Query Analysis**: Automatically extracts query labels like code_expert, math_expert, etc.
- **Dynamic Model Selection**: Chooses appropriate models based on query needs
- **Provider Diversity**: Limits models per provider to avoid rate limiting (max 2 per provider)
- **Model Type Configuration**: Supports free, paid, or optimized model selection strategies  
- **Response Synthesis**: Combines multiple model responses into cohesive answers
- **Conflict Resolution**: Detects conflicting answers and uses a tiebreaker model
- **API Error Handling**: Robust retry logic and graceful error management
- **Automatic Model Updates**: One-click updates for model data from OpenRouter API
- **Detailed Process Log**: Transparent view of all system operations
- **Process Visualization**: Progress tracking and detailed agent information

## Technical Architecture

### File Structure
```
multi_agent_app/
  ├── app.py                # Streamlit web interface
  ├── agents.py             # Model selection and labeling logic
  ├── coordinator.py        # Query analysis and response synthesis
  ├── utils.py              # API calls and helper functions  
  ├── multi_agent_cli.py    # Command line interface
  ├── data/
  │   ├── model_labels.json # Model capabilities metadata
  │   ├── model_roles.json  # System prompts for different roles
  │   └── backups/          # Automatic backups of configuration files
  ├── update_model_labels.py # Updates model list from OpenRouter API
  ├── update_model_roles.py  # Syncs model roles with current labels
  ├── api_test.py            # Tests API connectivity to models
  ├── model_id_check.py      # Diagnostics for model ID format issues
  └── .env                   # API keys and configuration
```

### Data Models

- **Model Labels**: Each AI model is tagged with capabilities (math_expert, code_expert, etc.)
- **Model Roles**: System prompts for each role/expertise to guide model behavior
  - Labels section: Descriptions of what each capability means
  - Roles section: System prompts to send to models with those capabilities
- **Query Labels**: Categories extracted from user queries to match with model expertise

### Key Workflows

1. **Query Processing**:
   - Parse user query
   - Extract relevant labels using coordinator model
   - Determine query complexity to decide agent count
   - Select appropriate models based on labels with provider diversity
   - Call each model with specialized role instructions
   - Detect and resolve conflicts if needed
   - Synthesize final response

2. **API Interaction**:
   - Uses OpenRouter API to access multiple AI providers
   - Configurable for free, paid, or optimized models
   - Implements enhanced retry logic with exponential backoff
   - Manages rate limits and errors gracefully
   - Limits models per provider to avoid API rate limits

3. **Model Management**:
   - Automatic model update system from OpenRouter API
   - Smart labeling for new models based on model characteristics
   - Synchronizes model roles with available labels
   - Maintains backup of configuration files

## Current Status (as of 2025-02-27)

- Successfully implemented both CLI and Streamlit web interfaces
- All core functionality working correctly
- API integration complete with OpenRouter with provider diversity
- Error handling improved with enhanced retry logic and exponential backoff
- UI refined with model update capabilities
- Provider diversity implemented to prevent rate limiting
- Automatic model labels and roles updating system
- Conversation history preservation for maintaining context
- Automatic backup management with file rotation (5 most recent per type)
- LaTeX sanitization for improved math formula display
- Enhanced error detection and user-friendly error messages

## Development Notes

### API Usage

- The system uses 2 messages per model call (1 system role, 1 user query)
- Coordinator analysis: 1 API call
- Agent processing: 1 API call per selected agent (max 2 per provider)
- Response synthesis: 1 API call

### Common Labels

- general_assistant: Basic chatbot capabilities
- code_expert: Programming and software development
- math_expert: Mathematical calculations and problems
- reasoning_expert: Logic, analysis, and critical thinking
- creative_writer: Creative content generation
- vision_expert: Image analysis capabilities
- fast_response: Optimized for quick responses
- instruction_following: Precise instruction execution
- multilingual: Support for multiple languages

### Testing & Debugging

- Run CLI tests with: `python multi_agent_cli.py`
- Quick test for components: `python quick_test.py`
- Launch web app: `streamlit run app.py`
- Test API connectivity: `python api_test.py`
- Check model ID formats: `python model_id_check.py`

### Common Issues

- API timeouts can occur with heavy query load
- Some models may return 'choices'/'user_id' errors from OpenRouter
- Coordinator model should be excluded from agent selection
- Model ID formats must match OpenRouter's format (provider/model-name)

## Future Improvements

- Implement cost estimation and monitoring
- Enhance query analysis for better label extraction
- Add multimodal support for images
- Improve visualization of coordinator-agent interactions
- Resolve remaining OpenRouter API issues with 'choices' not found errors
- Enhance automatic labeling for new models
- Create more comprehensive model compatibility tests
- Implement user preference settings persistence
- Add model response caching for similar queries
- Enhance analytics for query patterns and model performance

## Terminal Commands

- Start Streamlit app: `streamlit run app.py`
- Run CLI interface: `python multi_agent_cli.py [free|paid|optimized]`
- Test API connection: `python api_test.py`
- Update model definitions: `python update_model_labels.py`
- Update model roles: `python update_model_roles.py`
- Component testing: `python quick_test.py`

## Git Commands

- Clone repository: `git clone https://github.com/BTankut/multi_agent_app.git`
- Update repository: `git pull origin main`
- Check status: `git status`
- Create new branch: `git checkout -b feature/new-feature`
- Commit changes: `git add . && git commit -m "Description of changes"`
- Push changes: `git push origin branch-name`