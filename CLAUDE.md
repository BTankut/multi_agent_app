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
- **Smart Model Capability Detection**: Analyzes OpenRouter model pages for automatic tagging of new models
- **Provider Diversity**: Limits models per provider to avoid rate limiting (max 2 per provider)
- **Model Type Configuration**: Supports free, paid, or optimized model selection strategies  
- **Response Synthesis**: Combines multiple model responses into cohesive answers
- **Conflict Resolution**: Detects conflicting answers and uses a tiebreaker model
- **API Error Handling**: Immediate error detection without retries for faster response times
- **Automatic Model Updates**: One-click updates for model data from OpenRouter API
- **Detailed Process Log**: Transparent view of all system operations
- **Process Visualization**: Progress tracking and detailed agent information
- **Conversation History**: Maintains context between queries for coherent multi-turn interactions
- **Robust UI Design**: Isolated form processing and automatic query clearing for better UX
- **Response Metadata**: Detailed information about tokens, processing time, and costs
- **LaTeX Formula Handling**: Sanitization for improved formula display in responses
- **Intelligent Error Recovery**: Detailed error analysis and user-friendly error messages

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
  ├── logs/                 # Application logs directory
  │   ├── app_{date}.log    # General application logs (INFO level)
  │   ├── dev_{date}.log    # Detailed developer logs (DEBUG level)
  │   ├── conversation_{id}.json # Full conversation logs in JSON format
  │   └── error_{id}.json   # Detailed error logs with context
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
   - Implements immediate error handling without retries to reduce latency
   - Manages rate limits and errors gracefully with guaranteed fallback options
   - Limits models per provider to avoid API rate limits

3. **Model Management**:
   - Automatic model update system from OpenRouter API
   - Smart labeling for new models based on model characteristics and web descriptions
   - Automatically scrapes model capabilities from OpenRouter model pages
   - Analyzes web content for more accurate model expertise classification
   - Synchronizes model roles with available labels
   - Maintains backup of configuration files
   - Ensures complete label-role definition consistency

## Current Status (as of 2025-03-03)

- Successfully implemented both CLI and Streamlit web interfaces
- All core functionality working correctly
- API integration complete with OpenRouter with provider diversity
- Enhanced model capability detection via OpenRouter web page scraping
- Intelligent model expertise classification from model descriptions
- Advanced error handling with improved coordinator model failure recovery
- Robust coordinator model failover system with automatic alternative model suggestion
- Parallel API calls to agents for significantly faster response times
- Staggered API requests by provider to reduce rate limiting errors
- Enhanced rate limit detection and user-friendly error messages
- Optimized model prompts with structured response formats for faster processing
- Reduced token usage through concise instructions and response templates
- Improved error messages for common API errors (Provider errors, rate limits)
- Enhanced conflict resolution to handle API error scenarios gracefully
- Dedicated tiebreaker system that's always separate from agent models
- Comprehensive tiebreaker analysis of all conflicting agent responses
- Advanced logical problem solving in the tiebreaker prompt instructions
- Special coordinator weighting of tiebreaker conclusions in final synthesis
- Expanded conflict detection to all query types, not just code/math
- Pairwise similarity checks between all agent responses for better conflict detection
- Visual indicators in process log for tiebreaker operations and progress
- Streamlined UI for error states with clear recovery options
- Enhanced model family diversity to prevent selecting multiple models from same family
- Improved model family detection logic for variants of same model family (dolphin, claude, mistral, llama)
- Beta/alpha model filtering for improved stability
- Provider diversity implemented to prevent rate limiting
- Automatic model labels and roles updating system with strict consistency validation
- Language consistency ensured across all models - always respond in query language
- Robust etiket-rol tutarlılığı mekanizması with automatic validation
- Smart multi-level fallback mechanism when specialized models aren't available
- Optimized error handling that immediately suggests alternatives for API errors
- Guaranteed fallback model suggestions for all error scenarios
- Display of coordinator model in agent information panel
- Tanımsız etiketleri otomatik temizleme mekanizması
- Conversation history preservation for maintaining context
- Automatic backup management with file rotation (5 most recent per type)
- LaTeX sanitization for improved math formula display
- Enhanced error detection and user-friendly error messages
- Forms with automatic clearing for improved UX and cleaner interface
- Simplified UI structure with responsive layout and progress indicators
- Recent coordinator model tracking for quick selection of frequently used models
- Smart session state management to handle Streamlit re-rendering challenges
- Detailed conversation history with metadata and usage statistics
- Improved formatting for costs with proper Turkish currency format (comma as decimal separator)
- Fixed query text display to properly wrap long texts in UI
- Enhanced model information display with consistent typography and formatting
- Accurate cost representation without rounding issues
- Upgraded to pure JSON-based logging system for better organization and analysis
- Eliminated legacy text-based logs (conversation_log.txt) in favor of structured JSON logs
- Comprehensive logging system with multiple log levels (app_{date}.log, dev_{date}.log)
- Structured session tracking with unique session IDs
- Detailed JSON error logs with stack traces and context
- Complete conversation logs in JSON format for debugging and analysis (conversation_{id}.json)
- Fixed cost display logic for optimized and paid model selections

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
- Run automatic compatibility testing: `python auto_test.py`
- Debug model responses: Check the 'Conversation History' and 'Agent Responses' expanders
- Inspect API calls and model selection: See 'Process Log' in the UI for detailed traces
- Track token usage: View the agent information panel for token counts and costs
- Debug UI issues: Check Streamlit logs with `streamlit run app.py --log_level=debug`
- View application logs: Check `logs/app_{date}.log` for general operation logs
- Analyze detailed logs: Check `logs/dev_{date}.log` for detailed function-level logs
- Examine conversation data: Inspect `logs/conversation_{id}.json` for complete conversation records
- Review error context: Review `logs/error_{id}.json` for detailed error information

### Common Issues

- API timeouts can occur with heavy query load
- Some models may return 'choices'/'user_id' errors from OpenRouter
- Coordinator model should be excluded from agent selection
- Tiebreaker models must be different from the models that provided agent responses
- Logical problems may yield inconsistent answers due to different reasoning approaches
- Tiebreaker analysis should be prioritized for logical/mathematical conflicts
- Model ID formats must match OpenRouter's format (provider/model-name)
- Rate limiting may occur with multiple concurrent requests to the same provider
- Beta/alpha models may be unstable and should be avoided when possible
- Models from the same family (e.g., claude-3-sonnet and claude-3-haiku) offer less diverse insights
- Some models have context length limitations that can affect long conversations
- Error handling may need adjustment for specific API provider error formats
- Pairwise similarity checks between all model responses can be computationally intensive
- Streamlit session state requires careful management to prevent UI conflicts
- Conversation history can grow large and impact performance with extended use
- Model-role consistency requires proper update order: first roles, then labels
- Language consistency requires all role prompts to include language instructions
- Model labels must be validated against model_roles.json to prevent errors
- When updating models, always run update_model_roles.py before update_model_labels.py

## Future Improvements

- Implement budget controls with spending limits and alerts
- Enhance query analysis for better label extraction
- Add multimodal support for images and file uploads
- Further refine model family detection for improved diversity
- Implement model capability rating system to track model performance
- Automatically detect and avoid problematic models based on error history
- Extend model capability scraping with AI analysis of model descriptions
- Use ML techniques to better classify model capabilities from web content
- Support detection of more specialized capabilities (chemistry, biology, etc.)
- Implement better alternatives to Streamlit for more robust UI control
- Improve visualization of coordinator-agent interactions with interactive diagrams
- Enhance error recovery with context-aware model selection for alternatives
- Add smart model switching for query types (e.g., math queries to math experts)
- Resolve remaining OpenRouter API issues with 'choices' not found errors
- Add more comprehensive language detection and response validation
- Further enhance model-role-label consistency with automated testing
- Improve auto-fallback to general_assistant when specialized models unavailable
- Create more comprehensive model compatibility tests and automatic compatibility checks
- Add offline mode with cached responses for continued operation during API outages
- Implement user preference settings persistence for coordinator model and options
- Add model response caching for similar queries to reduce API costs
- Enhance analytics for query patterns and model performance metrics
- Implement pagination for conversation history with filtering options
- Add export functionality for conversation history in markdown and JSON formats
- Support for local model APIs alongside OpenRouter for hybrid operations
- Implement role-based access control for team usage
- Add theme customization and dark mode support
- Add internationalization support with multiple languages for UI
- Implement additional layout options for different screen sizes and devices
- Further optimize parallel API processing for even faster response times
- Implement dynamic timeout adjustments based on model response patterns
- Add comprehensive log visualization and analysis tools for debugging

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

## Development Workflow Rules

- **Commit and Push**: After every commit, immediately push to remote to keep the repository in sync
- **Local/Remote Sync**: Local and remote repositories should always be at the same commit
- **Backup Rotation**: Maintain only 5 most recent backups per file type to prevent clutter
- **Model Labeling**: Ensure specialized models (Claude, O3, etc.) are properly tagged with appropriate capabilities
- **UI Updates**: Test all UI changes in both desktop and mobile views before committing