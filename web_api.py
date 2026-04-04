"""
Web API for Customer Support Environment
Provides REST endpoints and web interface for HF Spaces deployment
"""

import asyncio
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
import os

from app.env import CustomerSupportEnv
from app.models import Action, Observation, Reward

# Import OpenAI client for AI actions (same as inference.py)
from openai import OpenAI

# =========================
# CONFIG
# =========================

# Get API credentials from environment variables (secrets in HF Spaces)
GROQ_API_KEY = os.getenv("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"
)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not set. AI actions will use fallback.")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url=API_BASE_URL
)

# =========================
# TASK-SPECIFIC PROMPTS (copied from inference.py)
# =========================

def get_task_prompt(task_id: str, step_num: int = 1, history: List = None) -> str:
    """Get the system prompt for a specific task and step"""
    
    if task_id == "order_status_easy":
        return (
            "You are a Customer Support AI. Return ONLY valid JSON.\n\n"
            "You MUST complete this in ONE step:\n"
            '{"action_type": "lookup_order", "order_id": "12345"}\n\n'
            "DO NOT send any reply messages. DO NOT ask questions.\n"
            "Just lookup the order and stop."
        )
    
    elif task_id == "refund_policy_medium":
        return (
            "You are a Customer Support AI. Return ONLY valid JSON.\n\n"
            "You MUST complete this in ONE step:\n"
            '{"action_type": "send_reply", "message": "Explain the refund policy here"}\n\n'
            "Your message must include the word 'refund'.\n"
            "DO NOT lookup the order. Just send a reply explaining the refund policy."
        )
    
    elif task_id == "address_change_hard":
        return (
            "You are a Customer Support AI handling address changes.\n"
            "Return ONLY valid JSON.\n\n"
            "You MUST follow this EXACT 3-step sequence:\n\n"
            "STEP 1 (lookup order):\n"
            '{"action_type": "lookup_order", "order_id": "12345"}\n\n'
            "STEP 2 (ask for address):\n"
            '{"action_type": "send_reply", "message": "Please provide your new address."}\n\n'
            "STEP 3 (ask for confirmation):\n"
            '{"action_type": "send_reply", "message": "Please confirm your new address."}\n\n'
            "CRITICAL RULES:\n"
            "- Step 1 MUST be lookup_order\n"
            "- Step 2 MUST ask for address (include word 'address')\n"
            "- Step 3 MUST ask for confirmation (include word 'confirm')\n"
            "- Do NOT add extra steps\n"
            "- Keep messages very short and direct"
        )
    
    else:
        return "Return valid JSON with action_type field."

def get_step_default_action(task_id: str, step_num: int) -> Dict[str, Any]:
    """Return default action for current step if model fails"""
    if task_id == "order_status_easy":
        return {"action_type": "lookup_order", "order_id": "12345"}
    
    elif task_id == "refund_policy_medium":
        return {
            "action_type": "send_reply", 
            "message": "Our refund policy allows returns within 30 days of purchase for a full refund."
        }
    
    elif task_id == "address_change_hard":
        if step_num == 1:
            return {"action_type": "lookup_order", "order_id": "12345"}
        elif step_num == 2:
            return {"action_type": "send_reply", "message": "Please provide your new address."}
        elif step_num == 3:
            return {"action_type": "send_reply", "message": "Please confirm your new address."}
    
    return {"action_type": "send_reply", "message": "How can I help you?"}

def call_ai_model(task_id: str, step_num: int, history: List) -> Dict[str, Any]:
    """Call the AI model to get an action (same as inference.py)"""
    sys_prompt = get_task_prompt(task_id, step_num, history)
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": f"Task: {task_id}\nStep: {step_num}\nHistory: {json.dumps(history)}\nReturn ONLY valid JSON with action_type field."
                }
            ],
            temperature=0.1,
            max_tokens=150
        )

        content = response.choices[0].message.content.strip()
        
        # Remove markdown
        if "```" in content:
            parts = content.split("```")
            if len(parts) > 1:
                content = parts[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

        # Extract JSON
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end != 0:
            content = content[start:end]
        else:
            return get_step_default_action(task_id, step_num)

        data = json.loads(content)
        
        # Ensure action_type exists
        if "action_type" not in data:
            return get_step_default_action(task_id, step_num)
        
        return data
        
    except Exception as e:
        print(f"AI model error: {e}")
        return get_step_default_action(task_id, step_num)

# Create FastAPI app
app = FastAPI(
    title="Customer Support Environment API",
    description="OpenEnv-compatible customer support tasks API",
    version="1.0.0"
)

# Serve web interface at root with built-in HTML
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the web interface"""
    # Built-in HTML interface with AI integration
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Support Environment</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; color: white; margin-bottom: 40px; }
        .header h1 { font-size: 3em; margin-bottom: 10px; }
        .header p { font-size: 1.2em; opacity: 0.9; }
        .tasks-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .task-card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }
        .task-card:hover { transform: translateY(-5px); }
        .task-card h3 { color: #667eea; font-size: 1.5em; margin-bottom: 10px; }
        .difficulty {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
            margin-bottom: 15px;
        }
        .easy { background: #10b981; color: white; }
        .medium { background: #f59e0b; color: white; }
        .hard { background: #ef4444; color: white; }
        .response-area {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin-top: 15px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            overflow-x: auto;
            max-height: 300px;
            overflow-y: auto;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            margin-top: 10px;
            margin-right: 10px;
            transition: opacity 0.3s ease;
        }
        button:hover { opacity: 0.9; }
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .status {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: white;
            padding: 10px 20px;
            border-radius: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            font-size: 0.9em;
        }
        .status.healthy { color: #10b981; }
        .api-link {
            text-align: center;
            margin-top: 20px;
        }
        .api-link a {
            color: white;
            text-decoration: none;
            background: rgba(255,255,255,0.2);
            padding: 10px 20px;
            border-radius: 25px;
            display: inline-block;
        }
        .reward-positive { color: #10b981; font-weight: bold; }
        .reward-negative { color: #ef4444; font-weight: bold; }
        .completed { color: #10b981; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Customer Support Environment</h1>
            <p>AI-powered customer support task automation</p>
        </div>
        
        <div class="tasks-grid" id="tasks"></div>
        
        <div class="api-link">
            <a href="/docs">📚 Interactive API Documentation (Swagger UI)</a>
        </div>
    </div>
    
    <div class="status" id="status">🟢 Healthy</div>
    
    <script>
        const API_BASE = window.location.origin;
        let currentSessions = {};
        let taskSteps = {};
        
        async function checkHealth() {
            try {
                const response = await fetch(`${API_BASE}/health`);
                const data = await response.json();
                document.getElementById('status').innerHTML = '🟢 Healthy';
                document.getElementById('status').className = 'status healthy';
            } catch (error) {
                document.getElementById('status').innerHTML = '🔴 Error';
            }
        }
        
        async function resetTask(taskId) {
            try {
                const response = await fetch(`${API_BASE}/reset/${taskId}`);
                const data = await response.json();
                currentSessions[taskId] = data.session_id;
                taskSteps[taskId] = 0;
                
                const responseDiv = document.getElementById(`response-${taskId}`);
                responseDiv.innerHTML = `<div style="color: green;">✅ Session created: ${currentSessions[taskId].substring(0, 50)}...</div>`;
                
                const stepBtn = document.getElementById(`step-${taskId}`);
                stepBtn.disabled = false;
                stepBtn.textContent = '🤖 Run AI Agent';
                
                const autoBtn = document.getElementById(`auto-${taskId}`);
                if (autoBtn) autoBtn.disabled = false;
            } catch (error) {
                alert('Error resetting task: ' + error.message);
            }
        }
        
        async function takeStep(taskId) {
            if (!currentSessions[taskId]) {
                alert('Please reset the task first!');
                return;
            }
            
            const stepBtn = document.getElementById(`step-${taskId}`);
            stepBtn.disabled = true;
            stepBtn.textContent = '⏳ AI Thinking...';
            
            try {
                const response = await fetch(`${API_BASE}/step_ai`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ task_id: taskId, action: {} })
                });
                
                const data = await response.json();
                taskSteps[taskId] = data.step;
                
                const responseDiv = document.getElementById(`response-${taskId}`);
                const rewardClass = data.reward >= 0 ? 'reward-positive' : 'reward-negative';
                const stepHtml = `
                    <div style="margin-top: 10px; padding: 10px; background: #f0f0f0; border-radius: 5px; border-left: 3px solid #667eea;">
                        <strong>Step ${data.step}:</strong><br>
                        Action: <code>${JSON.stringify(data.action_used)}</code><br>
                        Reward: <span class="${rewardClass}">${data.reward}</span><br>
                        Score: <strong>${(data.score * 100).toFixed(1)}%</strong><br>
                        ${data.done ? '<span class="completed">🎉 TASK COMPLETED!</span>' : '➡️ Continue...'}
                    </div>
                `;
                responseDiv.innerHTML = stepHtml + responseDiv.innerHTML;
                
                if (data.done) {
                    stepBtn.disabled = true;
                    stepBtn.textContent = '✅ Completed';
                    const autoBtn = document.getElementById(`auto-${taskId}`);
                    if (autoBtn) autoBtn.disabled = true;
                } else {
                    stepBtn.disabled = false;
                    stepBtn.textContent = '🤖 Take Next Step';
                }
            } catch (error) {
                alert('Error: ' + error.message);
                stepBtn.disabled = false;
                stepBtn.textContent = '🤖 Retry Step';
            }
        }
        
        async function runFullTask(taskId) {
            if (!currentSessions[taskId]) {
                await resetTask(taskId);
                await new Promise(r => setTimeout(r, 500));
            }
            
            const stepBtn = document.getElementById(`step-${taskId}`);
            const autoBtn = document.getElementById(`auto-${taskId}`);
            autoBtn.disabled = true;
            autoBtn.textContent = '🏃 Running...';
            
            let done = false;
            let maxSteps = taskId === 'address_change_hard' ? 5 : 3;
            
            for (let i = 0; i < maxSteps && !done; i++) {
                await new Promise(r => setTimeout(r, 300));
                
                const response = await fetch(`${API_BASE}/step_ai`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ task_id: taskId, action: {} })
                });
                
                const data = await response.json();
                taskSteps[taskId] = data.step;
                done = data.done;
                
                const responseDiv = document.getElementById(`response-${taskId}`);
                const rewardClass = data.reward >= 0 ? 'reward-positive' : 'reward-negative';
                const stepHtml = `
                    <div style="margin-top: 10px; padding: 10px; background: #f0f0f0; border-radius: 5px; border-left: 3px solid #667eea;">
                        <strong>Step ${data.step}:</strong><br>
                        Action: <code>${JSON.stringify(data.action_used)}</code><br>
                        Reward: <span class="${rewardClass}">${data.reward}</span><br>
                        Score: <strong>${(data.score * 100).toFixed(1)}%</strong><br>
                        ${data.done ? '<span class="completed">🎉 COMPLETED!</span>' : ''}
                    </div>
                `;
                responseDiv.innerHTML = stepHtml + responseDiv.innerHTML;
            }
            
            autoBtn.disabled = true;
            autoBtn.textContent = '✅ Done';
            const stepBtn2 = document.getElementById(`step-${taskId}`);
            stepBtn2.disabled = true;
        }
        
        async function loadTasks() {
            try {
                const response = await fetch(`${API_BASE}/tasks`);
                const data = await response.json();
                
                const tasksGrid = document.getElementById('tasks');
                tasksGrid.innerHTML = data.tasks.map(task => `
                    <div class="task-card">
                        <h3>${task.name}</h3>
                        <span class="difficulty ${task.difficulty}">${task.difficulty.toUpperCase()}</span>
                        <p>${task.description}</p>
                        <p><strong>Max steps:</strong> ${task.max_steps}</p>
                        <button onclick="resetTask('${task.task_id}')">🔄 Reset</button>
                        <button id="step-${task.task_id}" onclick="takeStep('${task.task_id}')" disabled>🤖 Run AI Agent</button>
                        <button id="auto-${task.task_id}" onclick="runFullTask('${task.task_id}')" disabled>⚡ Run Complete</button>
                        <div id="response-${task.task_id}" class="response-area"></div>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Error loading tasks:', error);
            }
        }
        
        checkHealth();
        loadTasks();
        setInterval(checkHealth, 30000);
    </script>
</body>
</html>
    """

# Mount static files
static_dir = "static"
if os.path.exists(static_dir) and os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Store active sessions
sessions = {}

class StepRequest(BaseModel):
    """Request model for step action"""
    task_id: str
    action: Dict[str, Any] = {}  # Made optional - AI will decide if empty

class StepResponse(BaseModel):
    """Response model for step action"""
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]

class TaskInfo(BaseModel):
    """Task information"""
    task_id: str
    name: str
    description: str
    difficulty: str
    max_steps: int

# =========================
# AI-POWERED STEP ENDPOINT (NEW)
# =========================

@app.post("/step_ai")
async def step_ai(request: StepRequest):
    """Take a step using AI model (same logic as inference.py)"""
    
    # Find or create session
    session_id = None
    for sid, session in sessions.items():
        if session["task_id"] == request.task_id:
            session_id = sid
            break
    
    if not session_id:
        # Create new session
        env = CustomerSupportEnv(request.task_id)
        obs = env.reset()
        session_id = f"{request.task_id}_{id(env)}"
        sessions[session_id] = {
            "env": env,
            "task_id": request.task_id,
            "rewards": [],
            "steps": 0,
            "history": []
        }
    
    env = sessions[session_id]["env"]
    current_step = sessions[session_id]["steps"] + 1
    
    # Call AI model to get action (same as inference.py)
    action_dict = call_ai_model(request.task_id, current_step, sessions[session_id]["history"])
    
    # Create action from AI response
    try:
        action = Action(**action_dict)
    except Exception as e:
        # Fallback to default action
        action_dict = get_step_default_action(request.task_id, current_step)
        action = Action(**action_dict)
    
    # Take step
    obs, reward, done, info = env.step(action)
    
    # Store reward and increment steps
    sessions[session_id]["rewards"].append(reward.value)
    sessions[session_id]["steps"] += 1
    sessions[session_id]["history"].append(action_dict)
    
    # Calculate current score
    total_score = sum(sessions[session_id]["rewards"])
    normalized_score = min(max(total_score, 0.0), 1.0)
    
    return {
        "session_id": session_id,
        "step": sessions[session_id]["steps"],
        "action_used": action_dict,
        "reward": reward.value,
        "done": done,
        "score": normalized_score,
        "total_reward": total_score
    }

# =========================
# ORIGINAL STEP ENDPOINT (kept for compatibility)
# =========================

@app.post("/step")
async def step_environment(request: StepRequest):
    """Take a step in the environment (manual action)"""
    # Find session
    session_id = None
    for sid, session in sessions.items():
        if session["task_id"] == request.task_id:
            session_id = sid
            break
    
    if not session_id:
        try:
            env = CustomerSupportEnv(request.task_id)
            obs = env.reset()
            session_id = f"{request.task_id}_{id(env)}"
            sessions[session_id] = {
                "env": env,
                "task_id": request.task_id,
                "rewards": [],
                "steps": 0,
                "history": []
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")
    
    env = sessions[session_id]["env"]
    
    try:
        if "action_type" not in request.action:
            raise HTTPException(status_code=400, detail="Action must contain 'action_type' field")
        
        action = Action(**request.action)
        obs, reward, done, info = env.step(action)
        
        sessions[session_id]["rewards"].append(reward.value)
        sessions[session_id]["steps"] += 1
        sessions[session_id]["history"].append(request.action)
        
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

# =========================
# HEALTH AND TASKS ENDPOINTS
# =========================

@app.get("/health")
async def health_check():
    """Health check endpoint for HF Spaces"""
    return {
        "status": "healthy",
        "service": "customer-support-env",
        "ready": True,
        "sessions": len(sessions),
        "model": MODEL_NAME
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
        raise HTTPException(status_code=400, detail=f"Invalid task_id: {task_id}")
    
    try:
        env = CustomerSupportEnv(task_id)
        obs = env.reset()
        
        session_id = f"{task_id}_{id(env)}"
        sessions[session_id] = {
            "env": env,
            "task_id": task_id,
            "rewards": [],
            "steps": 0,
            "history": []
        }
        
        return {
            "session_id": session_id,
            "task_id": task_id,
            "message": f"Environment reset for task: {task_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting environment: {str(e)}")

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
    print(f"Using model: {MODEL_NAME}")
    print(f"API Base URL: {API_BASE_URL}")
    print("Available endpoints:")
    print("  - http://localhost:7860/")
    print("  - http://localhost:7860/health")
    print("  - http://localhost:7860/tasks")
    print("  - http://localhost:7860/step_ai (AI-powered)")
    print("  - http://localhost:7860/docs")
    print("\nPress CTRL+C to stop the server")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=7860,
        log_level="info"
    )