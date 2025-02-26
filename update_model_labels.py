import json
import os
from dotenv import load_dotenv
import requests

# Load OpenRouter API key
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("ERROR: OPENROUTER_API_KEY not found in environment variables.")
    exit(1)

# Fetch available models from OpenRouter
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.get(
    "https://openrouter.ai/api/v1/models",
    headers=headers,
    timeout=10
)

response.raise_for_status()
openrouter_models = response.json().get('data', [])

# Create a lookup dictionary for model IDs
openrouter_model_ids = {model['id'].lower() for model in openrouter_models}

# Load current model_labels.json
with open('data/model_labels.json', 'r') as f:
    model_labels = json.load(f)

# Create mapping from old names to new names
model_mapping = {
    "Perplexity: R1 1776": "perplexity/r1-1776",
    "Mistral: Saba": "mistralai/mistral-saba",
    "Dolphin3.0 R1 Mistral 24B (free)": "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
    "Dolphin3.0 Mistral 24B (free)": "cognitivecomputations/dolphin3.0-mistral-24b:free",
    "Llama Guard 3 8B": "meta-llama/llama-guard-3-8b",
    "OpenAI: o3 Mini High": "openai/o3-mini-high",
    "Llama 3.1 Tulu 3 405B": "allenai/llama-3.1-tulu-3-405b",
    "DeepSeek: R1 Distill Llama 8B": "deepseek/deepseek-r1-distill-llama-8b",
    "Google: Gemini Flash 2.0": "google/gemini-2.0-flash-001",
    "Google: Gemini Flash Lite 2.0 Preview (free)": "google/gemini-2.0-flash-lite-preview-02-05:free",
    "Google: Gemini Pro 2.0 Experimental (free)": "google/gemini-2.0-pro-exp-02-05:free",
    "Qwen: Qwen VL Plus (free)": "qwen/qwen-vl-plus:free",
    "AionLabs: Aion-1.0": "aion-labs/aion-1.0",
    "AionLabs: Aion-1.0-Mini": "aion-labs/aion-1.0-mini",
    "AionLabs: Aion-RP 1.0 (8B)": "aion-labs/aion-rp-llama-3.1-8b",
    "Qwen: Qwen-Turbo": "qwen/qwen-turbo",
    "Qwen2.5 VL 72B Instruct (free)": "qwen/qwen2.5-vl-72b-instruct:free",
    "Qwen: Qwen-Plus": "qwen/qwen-plus",
    "Qwen: Qwen-Max": "qwen/qwen-max",
    "OpenAI: o3 Mini": "openai/o3-mini",
    "DeepSeek: R1 Distill Qwen 1.5B": "deepseek/deepseek-r1-distill-qwen-1.5b",
    "Mistral: Mistral Small 3 (free)": "mistralai/mistral-small-24b-instruct-2501:free",
    "Mistral: Mistral Small 3": "mistralai/mistral-small-24b-instruct-2501",
    "DeepSeek: R1 Distill Qwen 32B": "deepseek/deepseek-r1-distill-qwen-32b",
    "DeepSeek: R1 Distill Qwen 14B": "deepseek/deepseek-r1-distill-qwen-14b",
    "Perplexity: Sonar Reasoning": "perplexity/sonar-reasoning",
    "Perplexity: Sonar": "perplexity/sonar",
    "Liquid: LFM 7B": "liquid/lfm-7b",
    "Liquid: LFM 3B": "liquid/lfm-3b",
    "DeepSeek: R1 Distill Llama 70B (free)": "deepseek/deepseek-r1-distill-llama-70b:free",
    "DeepSeek: R1 Distill Llama 70B": "deepseek/deepseek-r1-distill-llama-70b",
    "Google: Gemini 2.0 Flash Thinking Experimental 01-21 (free)": "google/gemini-2.0-flash-thinking-exp-1219:free",
    "DeepSeek: R1 (free)": "deepseek/deepseek-r1:free",
    "DeepSeek: R1": "deepseek/deepseek-r1",
    "Rogue Rose 103B v0.2 (free)": "sophosympatheia/rogue-rose-103b-v0.2:free",
    "MiniMax: MiniMax-01": "minimax/minimax-01",
    "Mistral: Codestral 2501": "mistralai/codestral-2501",
    "Microsoft: Phi 4": "microsoft/phi-4",
    "Sao10K: Llama 3.1 70B Hanami x1": "sao10k/l3.1-70b-hanami-x1",
    "DeepSeek: DeepSeek V3 (free)": "deepseek/deepseek-chat:free",
    "DeepSeek: DeepSeek V3": "deepseek/deepseek-chat",
    "Qwen: QvQ 72B Preview": "qwen/qvq-72b-preview",
    "Google: Gemini 2.0 Flash Thinking Experimental (free)": "google/gemini-2.0-flash-thinking-exp:free",
    "Sao10K: Llama 3.3 Euryale 70B": "sao10k/l3.3-euryale-70b",
    "OpenAI: o1": "openai/o1",
    "EVA Llama 3.33 70B": "eva-unit-01/eva-llama-3.33-70b",
    "xAI: Grok 2 Vision 1212": "x-ai/grok-2-vision-1212",
    "xAI: Grok 2 1212": "x-ai/grok-2-1212",
    "Cohere: Command R7B (12-2024)": "cohere/command-r7b-12-2024",
    "Google: Gemini Flash 2.0 Experimental (free)": "google/gemini-2.0-flash-exp:free",
    "Google: Gemini Experimental 1206 (free)": "google/gemini-exp-1206:free",
    "Meta: Llama 3.3 70B Instruct (free)": "meta-llama/llama-3.3-70b-instruct:free",
    "Meta: Llama 3.3 70B Instruct": "meta-llama/llama-3.3-70b-instruct",
    "Amazon: Nova Lite 1.0": "amazon/nova-lite-v1",
    "Amazon: Nova Micro 1.0": "amazon/nova-micro-v1",
    "Amazon: Nova Pro 1.0": "amazon/nova-pro-v1",
    "Qwen: QwQ 32B Preview": "qwen/qwq-32b-preview",
    "Google: LearnLM 1.5 Pro Experimental (free)": "google/learnlm-1.5-pro-experimental:free",
    "EVA Qwen2.5 72B": "eva-unit-01/eva-qwen-2.5-72b",
    "OpenAI: GPT-4o (2024-11-20)": "openai/gpt-4o-2024-11-20",
    "Mistral Large 2411": "mistralai/mistral-large-2411",
    "Mistral Large 2407": "mistralai/mistral-large-2407",
    "Mistral: Pixtral Large 2411": "mistralai/pixtral-large-2411",
    "xAI: Grok Vision Beta": "x-ai/grok-vision-beta",
    "Infermatic: Mistral Nemo Inferor 12B": "infermatic/mn-inferor-12b",
    "Qwen2.5 Coder 32B Instruct": "qwen/qwen-2.5-coder-32b-instruct",
    "SorcererLM 8x22B": "raifle/sorcererlm-8x22b"
}

# Update model names in model_labels.json
updated_model_labels = []
for entry in model_labels:
    old_model_name = entry["model"]
    
    # Use mapping if available
    if old_model_name in model_mapping:
        new_model_name = model_mapping[old_model_name]
    # Try to guess based on name patterns
    else:
        # Extract model ID from various formats
        if ": " in old_model_name:
            provider, model_name = old_model_name.split(": ", 1)
            # Convert to lowercase with dashes instead of spaces
            provider_lower = provider.lower()
            model_name_lower = model_name.replace(" ", "-").lower()
            potential_name = f"{provider_lower}/{model_name_lower}"
            
            # Check if this exists in OpenRouter
            if potential_name.lower() in openrouter_model_ids:
                new_model_name = potential_name
            else:
                # Keep the old name if we can't find a match
                new_model_name = old_model_name
                print(f"Could not find matching OpenRouter ID for: {old_model_name}")
        else:
            new_model_name = old_model_name
            print(f"Could not parse model name format: {old_model_name}")
    
    # Update entry with new model name
    entry["model"] = new_model_name
    updated_model_labels.append(entry)

# Save updated model_labels.json
with open('data/model_labels.json', 'w') as f:
    json.dump(updated_model_labels, f, indent=2)

print(f"Updated model_labels.json with {len(updated_model_labels)} entries.")