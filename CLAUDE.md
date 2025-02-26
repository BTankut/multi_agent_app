# Multi_Agent System Project Documentation

## Project Overview

Multi_Agent System is an AI orchestration platform that intelligently routes user queries to specialized AI models based on content type and complexity. The system analyzes queries, selects appropriate specialized models from OpenRouter, dynamically manages the number of agents, and synthesizes their responses into coherent answers.

## Core Components

- **Coordinator**: Analyzes queries and orchestrates communication between models
- **Agents**: Multiple specialized AI models chosen based on query needs
- **Web/CLI Interface**: Streamlit web app and CLI for interacting with the system

## Key Features

- **Intelligent Query Analysis**: Automatically extracts query labels like code_expert, math_expert, etc.
- **Dynamic Model Selection**: Chooses appropriate models based on query needs
- **Model Type Configuration**: Supports free, paid, or optimized model selection strategies  
- **Response Synthesis**: Combines multiple model responses into cohesive answers
- **Conflict Resolution**: Detects conflicting answers and uses a tiebreaker model
- **API Error Handling**: Robust retry logic and graceful error management
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
  │   └── model_roles.json  # System prompts for different roles
  └── .env                  # API keys and configuration
```

### Data Models

- **Model Labels**: Each AI model is tagged with capabilities (math_expert, code_expert, etc.)
- **Model Roles**: System prompts for each role/expertise to guide model behavior
- **Query Labels**: Categories extracted from user queries to match with model expertise

### Key Workflows

1. **Query Processing**:
   - Parse user query
   - Extract relevant labels using coordinator model
   - Determine query complexity to decide agent count
   - Select appropriate models based on labels
   - Call each model with specialized role instructions
   - Detect and resolve conflicts if needed
   - Synthesize final response

2. **API Interaction**:
   - Uses OpenRouter API to access multiple AI providers
   - Configurable for free or paid models
   - Implements retry logic for API resilience
   - Manages rate limits and errors gracefully

## Current Status (as of 2025-02-26)

- Successfully implemented both CLI and Streamlit web interfaces
- All core functionality working correctly
- API integration complete with OpenRouter
- Error handling improved with retry logic
- UI refined for better user experience

## Development Notes

### API Usage

- The system uses 2 messages per model call (1 system role, 1 user query)
- Coordinator analysis: 1 API call
- Agent processing: 1 API call per selected agent
- Response synthesis: 1 API call

### Common Labels

- general_assistant: Basic chatbot capabilities
- code_expert: Programming and software development
- math_expert: Mathematical calculations and problems
- reasoning_expert: Logic, analysis, and critical thinking
- creative_writer: Creative content generation
- vision_expert: Image analysis capabilities

### Testing & Debugging

- Run CLI tests with: `python multi_agent_cli.py`
- Quick test for components: `python quick_test.py`
- Launch web app: `streamlit run app.py`

### Common Issues

- API timeouts can occur with heavy query load
- Some models (especially free ones) may return 'choices' not found errors
- Coordinator model should be excluded from agent selection

## Future Improvements

- Add conversation history support for context retention
- Implement cost estimation and monitoring
- Enhance query analysis for better label extraction
- Add multimodal support for images
- Improve visualization of coordinator-agent interactions

## Terminal Commands

- Start Streamlit app: `streamlit run app.py`
- Run CLI interface: `python multi_agent_cli.py [free|paid|optimized]`
- Test API connection: `python test_api.py`
- Component testing: `python quick_test.py`