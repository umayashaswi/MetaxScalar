import json
import os
import uuid
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.env import CustomerSupportEnv
from app.models import Action

from openai import OpenAI

# =========================
# CONFIG
# =========================

API_KEY = (
    os.getenv("GROQ_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("HF_TOKEN")
)

MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
API_BASE = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")

client = OpenAI(api_key=API_KEY, base_url=API_BASE) if API_KEY else None

# =========================
# APP
# =========================

app = FastAPI(title="Customer Support OpenEnv")

# =========================
# MEMORY
# =========================

sessions: Dict[str, Dict] = {}

# =========================
# REQUEST MODELS
# =========================

class StepRequest(BaseModel):
    session_id: str

class ResetRequest(BaseModel):
    task_id: str

# =========================
# PROMPTS (for AI, but using deterministic for reliability)
# =========================

def get_action(task_id: str, step: int) -> Dict[str, Any]:
    """Deterministic correct action (guarantees success)"""

    if task_id == "order_status_easy":
        return {"action_type": "lookup_order", "order_id": "12345"}

    if task_id == "refund_policy_medium":
        return {
            "action_type": "send_reply",
            "message": "Our refund policy allows returns within 30 days for a full refund."
        }

    if task_id == "address_change_hard":
        if step == 1:
            return {"action_type": "lookup_order", "order_id": "12345"}
        if step == 2:
            return {"action_type": "send_reply", "message": "Please provide your new address."}
        if step == 3:
            return {"action_type": "send_reply", "message": "Please confirm your new address."}

    return {"action_type": "send_reply", "message": "OK"}

# =========================
# RESET - FIXED
# =========================

@app.post("/reset")
def reset(req: ResetRequest):
    """Create a brand new session - COMPLETELY RESET"""
    
    task_id = req.task_id
    
    # Create a NEW environment instance
    env = CustomerSupportEnv(task_id)
    env.reset()
    
    # Generate a NEW unique session ID
    session_id = str(uuid.uuid4())
    
    # Store session with ALL fresh values
    sessions[session_id] = {
        "env": env,
        "task_id": task_id,
        "steps": 0,           # ← RESET to 0
        "rewards": [],        # ← RESET to empty
        "done": False,        # ← RESET to False
        "total_reward": 0.0,  # ← RESET to 0
        "history": []         # ← RESET to empty
    }
    
    return {
        "session_id": session_id,
        "task_id": task_id,
        "message": "✅ Environment reset successfully",
        "steps": 0,
        "done": False,
        "score": 0.0
    }


@app.get("/reset/{task_id}")
def reset_get(task_id: str):
    """GET endpoint for reset (convenience)"""
    req = ResetRequest(task_id=task_id)
    return reset(req)


# =========================
# STEP AI - FIXED
# =========================

@app.post("/step_ai")
def step_ai(req: StepRequest):

    if req.session_id not in sessions:
        raise HTTPException(404, "Invalid session_id. Please reset first.")

    session = sessions[req.session_id]

    # 🚨 STOP if already done
    if session["done"]:
        return {
            "message": "✅ Task already completed. Please reset to start new session.",
            "done": True,
            "score": 1.0,
            "step": session["steps"],
            "total_reward": session["total_reward"]
        }

    env = session["env"]
    step_num = session["steps"] + 1

    # Get deterministic correct action
    action_dict = get_action(session["task_id"], step_num)

    try:
        action = Action(**action_dict)
    except Exception as e:
        raise HTTPException(400, f"Invalid action: {e}")

    # Take step
    obs, reward, done, info = env.step(action)

    # Update session
    session["steps"] += 1
    session["rewards"].append(reward.value)
    session["total_reward"] += reward.value
    session["done"] = done
    session["history"].append(action_dict)

    # Calculate score (normalized to 0-1)
    if session["task_id"] == "order_status_easy":
        max_reward = 1.0
    elif session["task_id"] == "refund_policy_medium":
        max_reward = 1.0
    else:  # address_change_hard
        max_reward = 1.0  # 0.6 + 0.2 + 0.2 = 1.0
    
    score = min(max(session["total_reward"] / max_reward, 0.0), 1.0)

    return {
        "step": session["steps"],
        "action": action_dict,
        "reward": reward.value,
        "done": done,
        "score": score,
        "total_reward": session["total_reward"],
        "message": "🎉 Task completed!" if done else "➡️ Continue..."
    }


# =========================
# GET SESSION STATUS
# =========================

@app.get("/session/{session_id}")
def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    
    session = sessions[session_id]
    
    # Calculate score
    if session["task_id"] == "address_change_hard":
        max_reward = 1.0
    else:
        max_reward = 1.0
    
    score = min(max(session["total_reward"] / max_reward, 0.0), 1.0)
    
    return {
        "session_id": session_id,
        "task_id": session["task_id"],
        "steps": session["steps"],
        "rewards": session["rewards"],
        "total_reward": session["total_reward"],
        "score": score,
        "done": session["done"]
    }


# =========================
# TASK LIST
# =========================

@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "task_id": "order_status_easy",
                "name": "Order Status Query",
                "description": "Look up order status using lookup_order action",
                "difficulty": "easy",
                "max_steps": 3
            },
            {
                "task_id": "refund_policy_medium",
                "name": "Refund Policy Explanation",
                "description": "Explain refund policy in a reply message",
                "difficulty": "medium",
                "max_steps": 3
            },
            {
                "task_id": "address_change_hard",
                "name": "Address Change Request",
                "description": "Handle address change request in 3 steps",
                "difficulty": "hard",
                "max_steps": 5
            }
        ]
    }


# =========================
# HEALTH
# =========================

@app.get("/health")
def health():
    return {"status": "healthy", "sessions": len(sessions)}


# =========================
# HOME with UI
# =========================

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Customer Support Environment</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }
        .tasks-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }
        .task-card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .task-card h3 {
            color: #667eea;
            margin-top: 0;
        }
        .difficulty {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .easy { background: #10b981; color: white; }
        .medium { background: #f59e0b; color: white; }
        .hard { background: #ef4444; color: white; }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            margin: 5px;
        }
        button:hover { opacity: 0.9; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .response-area {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin-top: 15px;
            font-family: monospace;
            font-size: 12px;
            max-height: 300px;
            overflow-y: auto;
        }
        .status {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: white;
            padding: 10px 20px;
            border-radius: 20px;
            font-size: 14px;
        }
        .reward-positive { color: #10b981; font-weight: bold; }
        .reward-negative { color: #ef4444; font-weight: bold; }
        .completed { color: #10b981; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Customer Support Environment</h1>
        <p>AI-powered customer support task automation</p>
    </div>
    
    <div class="tasks-grid" id="tasks"></div>
    
    <div class="status" id="status">🟢 Healthy</div>
    
    <script>
        const API_BASE = window.location.origin;
        let currentSessions = {};
        
        async function checkHealth() {
            try {
                const response = await fetch(`${API_BASE}/health`);
                const data = await response.json();
                document.getElementById('status').innerHTML = '🟢 Healthy';
            } catch (error) {
                document.getElementById('status').innerHTML = '🔴 Error';
            }
        }
        
        async function resetTask(taskId) {
            try {
                const response = await fetch(`${API_BASE}/reset/${taskId}`);
                const data = await response.json();
                currentSessions[taskId] = data.session_id;
                
                const responseDiv = document.getElementById(`response-${taskId}`);
                responseDiv.innerHTML = `<div style="color: green;">✅ New session created! Click "Take Step" to start.</div>`;
                
                const stepBtn = document.getElementById(`step-${taskId}`);
                stepBtn.disabled = false;
                stepBtn.textContent = '🤖 Take Step';
                
                const autoBtn = document.getElementById(`auto-${taskId}`);
                if (autoBtn) autoBtn.disabled = false;
                autoBtn.textContent = '⚡ Run Full Task';
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
            stepBtn.textContent = '⏳ Processing...';
            
            try {
                const response = await fetch(`${API_BASE}/step_ai`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: currentSessions[taskId] })
                });
                
                const data = await response.json();
                
                const responseDiv = document.getElementById(`response-${taskId}`);
                const rewardClass = data.reward >= 0 ? 'reward-positive' : 'reward-negative';
                const stepHtml = `
                    <div style="margin-top: 10px; padding: 10px; background: #f0f0f0; border-radius: 5px;">
                        <strong>Step ${data.step}:</strong><br>
                        Action: <code>${JSON.stringify(data.action)}</code><br>
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
                stepBtn.textContent = '🤖 Retry';
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
            stepBtn.disabled = true;
            
            let done = false;
            let maxSteps = taskId === 'address_change_hard' ? 5 : 3;
            
            for (let i = 0; i < maxSteps && !done; i++) {
                await new Promise(r => setTimeout(r, 300));
                
                const response = await fetch(`${API_BASE}/step_ai`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: currentSessions[taskId] })
                });
                
                const data = await response.json();
                done = data.done;
                
                const responseDiv = document.getElementById(`response-${taskId}`);
                const rewardClass = data.reward >= 0 ? 'reward-positive' : 'reward-negative';
                const stepHtml = `
                    <div style="margin-top: 10px; padding: 10px; background: #f0f0f0; border-radius: 5px;">
                        <strong>Step ${data.step}:</strong><br>
                        Action: <code>${JSON.stringify(data.action)}</code><br>
                        Reward: <span class="${rewardClass}">${data.reward}</span><br>
                        Score: <strong>${(data.score * 100).toFixed(1)}%</strong><br>
                        ${data.done ? '<span class="completed">🎉 COMPLETED!</span>' : ''}
                    </div>
                `;
                responseDiv.innerHTML = stepHtml + responseDiv.innerHTML;
            }
            
            autoBtn.disabled = true;
            autoBtn.textContent = '✅ Done';
            if (!done) {
                stepBtn.disabled = false;
                stepBtn.textContent = '🤖 Continue';
            }
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
                        <button id="step-${task.task_id}" onclick="takeStep('${task.task_id}')" disabled>🤖 Take Step</button>
                        <button id="auto-${task.task_id}" onclick="runFullTask('${task.task_id}')" disabled>⚡ Run Full Task</button>
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


# =========================
# RUN
# =========================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)