# Multi_Agent System

A sophisticated system designed to handle user queries by intelligently coordinating multiple AI models through OpenRouter. The system analyzes query content to determine appropriate specialized models, dynamically manages the number of agents based on query complexity, and synthesizes their responses into a cohesive answer.

## Features

- Process queries using multiple AI models based on query type and difficulty
- Optimize model selection considering cost, specialization, and response quality
- Resolve conflicting responses using specialized reasoning models
- Provide detailed logging and transparent cost management
- Present a user-friendly interface with real-time progress tracking

## Installation

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file from the `.env-example` template and add your OpenRouter API key:
   ```
   cp .env-example .env
   ```
   Then edit the `.env` file to add your OpenRouter API key.

## Usage

### Web Interface (Recommended)

Run the Streamlit application using our launcher script:
```
python run_app.py
```

This will start the Streamlit server and open the web interface in your browser. The interface allows you to:
- Enter your query in the text area
- Select a coordinator model (e.g., Claude, GPT)
- Choose a model selection strategy (free, paid, or optimized)
- View the processing progress in real-time
- See the final response
- Browse your conversation history

### Command Line Interface

For a simpler experience, you can use the CLI version:
```
python multi_agent_cli.py [option]
```

Where `[option]` is one of:
- `free`: Use only free models (default)
- `paid`: Use only paid models
- `mixed`: Use a combination of free and paid models

### Testing

To verify that the system is working correctly:
```
python quick_test.py
```

This will test the core components without making full API calls.

## System Architecture

The system consists of the following components:

- **Coordinator**: Orchestrates the multi-agent workflow, analyzes queries, and synthesizes responses
- **Agents**: Manages model selection, role assignment, and response generation
- **Utils**: Handles API integration, error handling, and helper functions
- **Data Files**: Contains model capabilities and role definitions

## Configuration

Model configurations and roles are defined in the data directory:
- `model_labels.json`: Maps models to their capabilities (e.g., code_expert, math_expert)
- `model_roles.json`: Defines specific prompting instructions for each role

You can customize these files to add or modify model capabilities and roles.

## Dependencies

- Streamlit: Web interface
- Requests: API communication
- Python-dotenv: Environment variable management
- Regex: Pattern matching

## Troubleshooting

If you encounter issues:
1. Verify your OpenRouter API key is correctly set in the `.env` file
2. Check that all dependencies are installed with `pip install -r requirements.txt`
3. Run the `quick_test.py` script to verify core components
4. Ensure you have sufficient credits in your OpenRouter account

For paid models, you'll need to have your payment method set up with OpenRouter.