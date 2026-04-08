from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any

from app.env import CustomerSupportEnv

app = FastAPI()

# =========================
# MODELS
# =========================

class Action(BaseModel):
    action_type: str
    order_id: Optional[str] = None
    message: Optional[str] = None

class ResetRequest(BaseModel):
    # FIX: Made task_id optional to prevent 422 Validation Errors on empty requests
    task_id: Optional[str] = None

# =========================
# GLOBAL ENV
# =========================

env: Optional[CustomerSupportEnv] = None

# =========================
# ENDPOINTS
# =========================

@app.post("/openenv/reset")
def reset_env(request: ResetRequest):
    global env
    
    # FIX: Assign a fallback if the platform sends an empty body
    assigned_task = request.task_id if request.task_id else "default_task"
    
    env = CustomerSupportEnv(task_id=assigned_task)
    return env.reset()

@app.post("/openenv/step")
def step_env(action: Action):
    global env

    if env is None:
        return {"error": "Environment not initialized. Call /reset first."}

    obs, reward, done, info = env.step(action)

    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/openenv/validate")
def validate():
    return {"status": "ok"}
