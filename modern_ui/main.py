import sys
import os
from pathlib import Path

# Add parent directory to path to allow importing modules from root
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import uvicorn
import logging

# Import existing logic
from utils import get_openrouter_models
from coordinator import process_query

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("modern_ui")

app = FastAPI(title="Multi-Agent AI System")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="modern_ui/static"), name="static")

# Pydantic models
class ChatRequest(BaseModel):
    query: str
    coordinator_model: str
    option: str = "free" # free, paid, optimized
    coordinator_history: List[Dict[str, Any]] = []
    agent_history: Dict[str, Any] = {}
    reasoning_mode: str = "disabled"

class ChatResponse(BaseModel):
    answer: str
    labels: List[str]
    coordinator_history: List[Dict[str, Any]]
    agent_history: Dict[str, Any]
    error: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("modern_ui/static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/models")
async def get_models():
    """Fetch available models from OpenRouter."""
    try:
        models = get_openrouter_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat query."""
    try:
        logger.info(f"Processing query: {request.query[:50]}...")
        
        # Fetch models internally if needed for the coordinator logic
        # (The coordinator needs the full list to select agents)
        openrouter_models = get_openrouter_models()
        
        final_answer, labels, updated_histories = process_query(
            query=request.query,
            coordinator_model=request.coordinator_model,
            option=request.option,
            openrouter_models=openrouter_models,
            coordinator_history=request.coordinator_history,
            agent_history=request.agent_history,
            reasoning_mode=request.reasoning_mode
        )
        
        return ChatResponse(
            answer=final_answer,
            labels=labels,
            coordinator_history=updated_histories['coordinator'],
            agent_history=updated_histories['agents']
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        # Return a valid response with the error field set, 
        # so the frontend can display it nicely instead of crashing
        return ChatResponse(
            answer="An error occurred while processing your request.",
            labels=[],
            coordinator_history=request.coordinator_history,
            agent_history=request.agent_history,
            error=str(e)
        )

if __name__ == "__main__":
    uvicorn.run("modern_ui.main:app", host="127.0.0.1", port=8000, reload=True)
