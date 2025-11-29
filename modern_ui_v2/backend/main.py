import sys
import os
import json
import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

# Import core logic
from utils import get_openrouter_models, logger as app_logger
from coordinator import process_query
import update_model_roles
import model_intelligence

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("modern_ui_v2")

app = FastAPI(title="Multi-Agent AI Orchestrator")

# Mount static files
static_path = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Store active connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_json(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

# Custom Log Handler to stream events to WebSocket
class WebSocketLogHandler(logging.Handler):
    def __init__(self, websocket: WebSocket):
        super().__init__()
        self.websocket = websocket
        self.loop = asyncio.get_event_loop()
        self.current_coordinator = None

    def set_coordinator(self, model_name):
        self.current_coordinator = self.normalize_model_name(model_name)

    def normalize_model_name(self, name):
        if not name: return ""
        return name.lower().replace(":free", "").strip()

    def emit(self, record):
        try:
            # Parse structured logs if possible, or verify log message content
            msg = self.format(record)
            
            # Determine event type based on log content/level
            event_type = "log"
            payload = {"message": msg, "level": record.levelname}

            # Smart parsing for visualization
            if "Calling model:" in msg:
                raw_model_name = msg.split("Calling model:")[1].split("with")[0].strip()
                model_name = self.normalize_model_name(raw_model_name)
                
                # If the called model is the coordinator itself, it's a planning step, not a worker task
                if self.current_coordinator and model_name == self.current_coordinator:
                    event_type = "coordinator_thinking"
                else:
                    event_type = "agent_start"
                    payload["model"] = raw_model_name # Keep original name for display
            elif "received response from:" in msg:
                event_type = "agent_finish"
                payload["model"] = msg.split("received response from:")[1].strip()
            elif "Selected premium model:" in msg:
                event_type = "agent_selected"
                payload["model"] = msg.split("Selected premium model:")[1].split("(")[0].strip()
            elif "Synthesizing final answer" in msg:
                event_type = "coordinator_thinking"
            
            # Send via WebSocket (fire and forget task)
            asyncio.run_coroutine_threadsafe(
                self.websocket.send_json({
                    "type": event_type,
                    "data": payload,
                    "timestamp": record.created
                }),
                self.loop
            )
        except Exception:
            self.handleError(record)

@app.get("/")
async def get():
    return FileResponse(str(static_path / "index.html"))

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Global Cache
GLOBAL_CACHE = {
    "models": [],
    "last_updated": None
}

def load_models_cache():
    """Load models from file cache or API."""
    global GLOBAL_CACHE
    cache_file = Path("data/models_cache.json")
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                GLOBAL_CACHE["models"] = data.get("models", [])
                GLOBAL_CACHE["last_updated"] = data.get("last_updated")
                logger.info(f"Loaded {len(GLOBAL_CACHE['models'])} models from disk cache")
                return
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            
    # Fallback to API
    try:
        models = get_openrouter_models()
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        GLOBAL_CACHE["models"] = models
        GLOBAL_CACHE["last_updated"] = timestamp
        
        # Save to disk
        os.makedirs("data", exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(GLOBAL_CACHE, f)
    except Exception as e:
        logger.error(f"Startup model fetch failed: {e}")

@app.on_event("startup")
async def startup_event():
    load_models_cache()

@app.get("/api/models")
async def get_models(force_refresh: bool = False):
    """Fetch available models with caching."""
    global GLOBAL_CACHE
    cache_file = Path("data/models_cache.json")
    
    # Try to serve from memory cache first
    if not force_refresh and GLOBAL_CACHE["models"]:
        return GLOBAL_CACHE
        
    # Try to serve from disk cache
    if not force_refresh and cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                GLOBAL_CACHE = cached_data # Update memory
                return GLOBAL_CACHE
        except Exception:
            pass
    
    # Fetch fresh data
    try:
        if force_refresh:
            logger.info("Force refresh requested. Starting AI Model Intelligence analysis...")
            
            # First update roles definitions (static structure)
            update_model_roles.update_model_roles()
            
            # Then run the AI analyst to update labels and specific roles
            # This might take time but it supports resuming
            result = model_intelligence.analyze_models()
            
            if result.get("success"):
                logger.info(f"AI Analysis complete. {result.get('count', 0)} models analyzed/updated.")
            else:
                logger.error(f"AI Analysis failed: {result.get('error')}")

        models = get_openrouter_models()
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        GLOBAL_CACHE["models"] = models
        GLOBAL_CACHE["last_updated"] = timestamp
        
        # Save to cache
        os.makedirs("data", exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(GLOBAL_CACHE, f)
            
        return GLOBAL_CACHE
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return {"models": [], "last_updated": "Unknown"}

@app.post("/api/analyze-models")
async def analyze_models_endpoint(background_tasks: BackgroundTasks):
    """
    Triggers a comprehensive analysis of all models using Grok 4.1 (or configured analyst).
    This is a long-running process, so it runs in the background.
    """
    def run_analysis():
        try:
            logger.info("Starting background model analysis...")
            result = model_intelligence.analyze_models()
            if result.get("success"):
                logger.info(f"Analysis complete. Processed {result.get('count')} models.")
                # Update global cache to reflect new intelligence
                load_models_cache()
            else:
                logger.error(f"Analysis failed: {result.get('error')}")
        except Exception as e:
            logger.error(f"Exception in background analysis: {e}")

    background_tasks.add_task(run_analysis)
    return {"status": "started", "message": "Model analysis started in background. Check logs for progress."}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    # Attach custom log handler for this session
    ws_handler = WebSocketLogHandler(websocket)
    formatter = logging.Formatter('%(message)s')
    ws_handler.setFormatter(formatter)
    
    # Add handler to both main app logger and the coordinator/utils loggers
    app_logger.addHandler(ws_handler)
    logging.getLogger("coordinator").addHandler(ws_handler)
    logging.getLogger("agents").addHandler(ws_handler)
    
    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)
            
            if request.get("type") == "query":
                payload = request.get("payload", {})
                query = payload.get("query")
                coordinator_model = payload.get("coordinator_model", "anthropic/claude-3.5-sonnet")
                option = payload.get("option", "free")
                reasoning_mode = payload.get("reasoning_mode", "disabled")
                
                # Update handler with current coordinator
                ws_handler.set_coordinator(coordinator_model)
                
                await manager.send_json({"type": "status", "data": "started"}, websocket)

                # Track query start time
                query_start_time = time.time()

                # Run blocking process_query in a separate thread to not block WebSocket loop
                # and allow logs to stream
                loop = asyncio.get_event_loop()

                # Wrapper to run sync function
                def run_process():
                    # Ensure we have models loaded
                    if not GLOBAL_CACHE["models"]:
                        load_models_cache()

                    return process_query(
                        query=query,
                        coordinator_model=coordinator_model,
                        option=option,
                        reasoning_mode=reasoning_mode,
                        openrouter_models=GLOBAL_CACHE["models"]
                    )

                try:
                    # Execute logic
                    final_answer, labels, histories, usage_data = await loop.run_in_executor(None, run_process)

                    # Calculate total query duration
                    query_duration = time.time() - query_start_time

                    # Send final result with metrics
                    await manager.send_json({
                        "type": "result",
                        "data": {
                            "answer": final_answer,
                            "labels": labels,
                            "history": histories,
                            "metrics": {
                                "total_tokens": usage_data.get("total_tokens", 0),
                                "total_cost": usage_data.get("total_cost", 0.0),
                                "query_duration": round(query_duration, 2),
                                "models_used": len(usage_data.get("models", {})),
                                "model_details": usage_data.get("models", {})
                            }
                        }
                    }, websocket)
                    
                except Exception as e:
                    await manager.send_json({
                        "type": "error",
                        "data": str(e)
                    }, websocket)
                    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        app_logger.removeHandler(ws_handler)
        logging.getLogger("coordinator").removeHandler(ws_handler)
        logging.getLogger("agents").removeHandler(ws_handler)

if __name__ == "__main__":
    uvicorn.run("modern_ui_v2.backend.main:app", host="127.0.0.1", port=8000, reload=True)