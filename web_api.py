"""
Web API for Customer Support Environment
Provides REST endpoints for HF Spaces deployment
"""

import asyncio
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
import os

from app.env import CustomerSupportEnv
from app.models import Action, Observation, Reward

# Create FastAPI app FIRST
app = FastAPI(
    title="Customer Support Environment API",
    description="OpenEnv-compatible customer support tasks API",
    version="1.0.0"
)

# Mount static files (create 'static' folder with index.html)
# Check if static directory exists before mounting
static_dir = "static"
if os.path.exists(static_dir) and os.path.isdir(static_dir):
    app.mount("/ui", StaticFiles(directory=static_dir, html=True), name="ui")

# Store active sessions
sessions = {}

class StepRequest(BaseModel):
    """Request model for step action"""
    task_id: str
    action: Dict[str, Any]

class StepResponse(BaseModel):
    """Response model for step action"""
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]

class ResetResponse(BaseModel):
    """Response model for reset"""
    observation: Dict[str, Any]
    task_id: str

class TaskInfo(BaseModel):
    """Task information"""
    task_id: str
    name: str
    description: str
    difficulty: str
    max_steps: int

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Customer Support Environment",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/": "This info",
            "/health": "Health check",
            "/tasks": "List available tasks",
            "/reset/{task_id}": "Reset environment for a task",
            "/step": "Take an action in the environment",
            "/score/{session_id}": "Get current score for a session",
            "/docs": "Interactive API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for HF Spaces"""
    return {
        "status": "healthy",
        "service": "customer-support-env",
        "ready": True,
        "sessions": len(sessions)
    }

@app.get("/tasks")
async def list_tasks():
    """List all available tasks"""
    tasks = [
        TaskInfo(
            task_id="order_status_easy",
            name="Order Status Query",
            description="Look up order status using lookup_order action",
            difficulty="easy",
            max_steps=3
        ),
        TaskInfo(
            task_id="refund_policy_medium",
            name="Refund Policy Explanation",
            description="Explain refund policy in a reply message",
            difficulty="medium",
            max_steps=3
        ),
        TaskInfo(
            task_id="address_change_hard",
            name="Address Change Request",
            description="Handle address change request in 3 steps",
            difficulty="hard",
            max_steps=5
        )
    ]
    return {"tasks": tasks}

@app.get("/reset/{task_id}")
async def reset_environment(task_id: str):
    """Reset environment for a specific task"""
    valid_tasks = ["order_status_easy", "refund_policy_medium", "address_change_hard"]
    if task_id not in valid_tasks:
        raise HTTPException(status_code=400, detail=f"Invalid task_id: {task_id}. Must be one of {valid_tasks}")
    
    try:
        env = CustomerSupportEnv(task_id)
        obs = env.reset()
        
        # Store session
        session_id = f"{task_id}_{id(env)}"
        sessions[session_id] = {
            "env": env,
            "task_id": task_id,
            "rewards": [],
            "steps": 0
        }
        
        return {
            "session_id": session_id,
            "task_id": task_id,
            "observation": {
                "task_id": obs.task_id,
                "history": obs.history,
                "done": obs.done
            },
            "message": f"Environment reset for task: {task_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting environment: {str(e)}")

@app.post("/step")
async def step_environment(request: StepRequest):
    """Take a step in the environment"""
    # Find session
    session_id = None
    for sid, session in sessions.items():
        if session["task_id"] == request.task_id:
            session_id = sid
            break
    
    if not session_id:
        # Create new session if doesn't exist
        try:
            env = CustomerSupportEnv(request.task_id)
            obs = env.reset()
            session_id = f"{request.task_id}_{id(env)}"
            sessions[session_id] = {
                "env": env,
                "task_id": request.task_id,
                "rewards": [],
                "steps": 0
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")
    
    env = sessions[session_id]["env"]
    
    try:
        # Validate action has required fields
        if "action_type" not in request.action:
            raise HTTPException(status_code=400, detail="Action must contain 'action_type' field")
        
        # Create action from request
        action = Action(**request.action)
        
        # Take step
        obs, reward, done, info = env.step(action)
        
        # Store reward and increment steps
        sessions[session_id]["rewards"].append(reward.value)
        sessions[session_id]["steps"] += 1
        
        # Calculate current score
        total_score = sum(sessions[session_id]["rewards"])
        normalized_score = min(max(total_score, 0.0), 1.0)
        
        return {
            "session_id": session_id,
            "step": sessions[session_id]["steps"],
            "observation": {
                "task_id": obs.task_id,
                "history": obs.history,
                "done": obs.done
            },
            "reward": reward.value,
            "done": done,
            "score": normalized_score,
            "total_reward": total_score,
            "info": info
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error taking step: {str(e)}")

@app.get("/score/{session_id}")
async def get_score(session_id: str):
    """Get current score for a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    
    rewards = sessions[session_id]["rewards"]
    total_score = sum(rewards)
    normalized_score = min(max(total_score, 0.0), 1.0)
    
    return {
        "session_id": session_id,
        "task_id": sessions[session_id]["task_id"],
        "rewards": rewards,
        "total_reward": total_score,
        "normalized_score": normalized_score,
        "steps": sessions[session_id]["steps"],
        "max_possible_score": 1.0,
        "completion_percentage": f"{normalized_score * 100:.1f}%"
    }

@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "active_sessions": len(sessions),
        "sessions": [
            {
                "session_id": sid,
                "task_id": session["task_id"],
                "steps": session["steps"],
                "rewards": session["rewards"],
                "total_reward": sum(session["rewards"])
            }
            for sid, session in sessions.items()
        ]
    }

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    
    del sessions[session_id]
    return {"message": f"Session {session_id} deleted successfully"}

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up sessions on shutdown"""
    sessions.clear()

# For direct execution
if __name__ == "__main__":
    print("Starting Customer Support Environment API Server...")
    print("Available endpoints:")
    print("  - http://localhost:7860/")
    print("  - http://localhost:7860/health")
    print("  - http://localhost:7860/tasks")
    print("  - http://localhost:7860/docs")
    print("  - http://localhost:7860/redoc")
    print("\nPress CTRL+C to stop the server")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=7860,
        log_level="info"
    )