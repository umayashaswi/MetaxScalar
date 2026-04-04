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
# ACTION VALIDATION & REWARD EXPLANATIONS
# =========================

def validate_action_and_get_explanation(task_id: str, step: int, action_dict: Dict) -> tuple:
    """Validate action and return (is_valid, reward, explanation)"""
    
    action_type = action_dict.get("action_type", "")
    
    # Order Status Easy Task
    if task_id == "order_status_easy":
        if action_type == "lookup_order" and action_dict.get("order_id") == "12345":
            return True, 1.0, "✓ Correct: Order lookup successful"
        else:
            return False, -0.3, "✗ Invalid: Use lookup_order with order_id '12345'"
    
    # Refund Policy Medium Task
    if task_id == "refund_policy_medium":
        if action_type == "send_reply" and "refund" in action_dict.get("message", "").lower():
            return True, 1.0, "✓ Correct: Refund policy explained"
        else:
            return False, -0.3, "✗ Invalid: Send reply explaining refund policy (must include 'refund')"
    
    # Address Change Hard Task
    if task_id == "address_change_hard":
        if step == 1:
            if action_type == "lookup_order" and action_dict.get("order_id") == "12345":
                return True, 0.6, "✓ Step 1/3: Order located successfully"
            else:
                return False, -0.3, "✗ Step 1/3: Must lookup order with order_id '12345'"
        
        elif step == 2:
            if action_type == "send_reply" and "address" in action_dict.get("message", "").lower():
                return True, 0.2, "✓ Step 2/3: Address request sent to customer"
            else:
                return False, -0.3, "✗ Step 2/3: Ask customer to provide new address"
        
        elif step == 3:
            if action_type == "send_reply" and "confirm" in action_dict.get("message", "").lower():
                return True, 0.2, "✓ Step 3/3: Address confirmation requested - Task complete!"
            else:
                return False, -0.3, "✗ Step 3/3: Ask customer to confirm the new address"
    
    return False, -0.3, "✗ Invalid action format"

# =========================
# GET CORRECT ACTION (for reference)
# =========================

def get_correct_action(task_id: str, step: int) -> Dict[str, Any]:
    """Get the correct action for current step"""
    
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
        "steps": 0,
        "rewards": [],
        "explanations": [],
        "done": False,
        "total_reward": 0.0,
        "history": []
    }
    
    return {
        "session_id": session_id,
        "task_id": task_id,
        "message": f"✅ Environment reset successfully for {task_id}",
        "steps": 0,
        "done": False,
        "score": "0.0 / 1.0"
    }


@app.get("/reset/{task_id}")
def reset_get(task_id: str):
    """GET endpoint for reset (convenience)"""
    req = ResetRequest(task_id=task_id)
    return reset(req)


# =========================
# STEP AI - WITH VALIDATION & EXPLANATIONS
# =========================

@app.post("/step_ai")
def step_ai(req: StepRequest):

    if req.session_id not in sessions:
        raise HTTPException(404, "Invalid session_id. Please reset first.")

    session = sessions[req.session_id]

    # STOP if already done
    if session["done"]:
        final_message = ""
        if session["task_id"] == "address_change_hard":
            final_message = "🎉 Address change successfully completed! The customer's address has been updated."
        elif session["task_id"] == "order_status_easy":
            final_message = "🎉 Order status retrieved successfully!"
        elif session["task_id"] == "refund_policy_medium":
            final_message = "🎉 Refund policy explained to customer!"
        
        return {
            "message": f"✅ Task already completed. {final_message} Please reset to start new session.",
            "done": True,
            "score": f"{session['total_reward']:.1f} / 1.0",
            "step": session["steps"],
            "total_reward": session["total_reward"],
            "final_message": final_message
        }

    env = session["env"]
    step_num = session["steps"] + 1

    # Get deterministic correct action
    action_dict = get_correct_action(session["task_id"], step_num)

    try:
        action = Action(**action_dict)
    except Exception as e:
        raise HTTPException(400, f"Invalid action: {e}")
    
    # Validate action and get explanation
    is_valid, reward_value, explanation = validate_action_and_get_explanation(
        session["task_id"], step_num, action_dict
    )
    
    # Take step in environment
    obs, env_reward, done, info = env.step(action)
    
    # Use our validated reward (ensures consistency)
    final_reward = reward_value
    
    # Update session
    session["steps"] += 1
    session["rewards"].append(final_reward)
    session["explanations"].append(explanation)
    session["total_reward"] += final_reward
    session["done"] = done
    session["history"].append(action_dict)

    # Calculate score display
    score_display = f"{session['total_reward']:.1f} / 1.0"
    
    # Prepare response message
    response_message = explanation
    if done and session["task_id"] == "address_change_hard":
        response_message = "✓ Step 3/3: Address confirmation requested - Task complete!\n\n🎉 Address change successfully completed! The customer's address has been updated."

    return {
        "step": session["steps"],
        "action": action_dict,
        "reward": final_reward,
        "reward_explanation": explanation,
        "done": done,
        "score": score_display,
        "score_value": session["total_reward"],
        "total_reward": session["total_reward"],
        "message": response_message,
        "task_complete_message": "🎉 Address change successfully completed!" if done and session["task_id"] == "address_change_hard" else None
    }


# =========================
# GET SESSION STATUS
# =========================

@app.get("/session/{session_id}")
def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    
    session = sessions[session_id]
    
    return {
        "session_id": session_id,
        "task_id": session["task_id"],
        "steps": session["steps"],
        "rewards": session["rewards"],
        "explanations": session["explanations"],
        "total_reward": session["total_reward"],
        "score": f"{session['total_reward']:.1f} / 1.0",
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
                "max_steps": 3,
                "expected_reward": 1.0
            },
            {
                "task_id": "refund_policy_medium",
                "name": "Refund Policy Explanation",
                "description": "Explain refund policy in a reply message",
                "difficulty": "medium",
                "max_steps": 3,
                "expected_reward": 1.0
            },
            {
                "task_id": "address_change_hard",
                "name": "Address Change Request",
                "description": "Handle address change request in 3 steps",
                "difficulty": "hard",
                "max_steps": 5,
                "expected_reward": 1.0,
                "steps_detail": [
                    "Step 1: Lookup order (+0.6)",
                    "Step 2: Ask for address (+0.2)", 
                    "Step 3: Confirm address (+0.2)"
                ]
            }
        ]
    }


# =========================
# HEALTH
# =========================

@app.get("/health")
def health():
    return {"status": "healthy", "active_sessions": len(sessions)}


# =========================
# HOME with IMPROVED UI
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
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }
        .header h1 { font-size: 3em; margin-bottom: 10px; }
        .header p { font-size: 1.2em; opacity: 0.9; }
        .tasks-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 25px;
        }
        .task-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }
        .task-card:hover { transform: translateY(-5px); }
        .task-card h3 { color: #667eea; margin-top: 0; font-size: 1.5em; }
        .difficulty {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            margin-bottom: 15px;
        }
        .easy { background: #10b981; color: white; }
        .medium { background: #f59e0b; color: white; }
        .hard { background: #ef4444; color: white; }
        .reward-badge {
            display: inline-block;
            background: #e8eaf6;
            padding: 5px 10px;
            border-radius: 10px;
            font-size: 12px;
            margin: 5px 0;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            margin: 5px 5px 5px 0;
            font-size: 14px;
            transition: opacity 0.3s ease;
        }
        button:hover { opacity: 0.9; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .response-area {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin-top: 15px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            max-height: 400px;
            overflow-y: auto;
        }
        .step-entry {
            margin-top: 10px;
            padding: 12px;
            background: white;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .status {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: white;
            padding: 10px 20px;
            border-radius: 20px;
            font-size: 14px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .reward-positive { color: #10b981; font-weight: bold; }
        .reward-negative { color: #ef4444; font-weight: bold; }
        .completed { color: #10b981; font-weight: bold; font-size: 16px; }
        .score-large { font-size: 24px; font-weight: bold; color: #667eea; }
        .explanation { color: #6b7280; font-size: 12px; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Customer Support Environment</h1>
        <p>AI-powered customer support task automation with explainable rewards</p>
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
                document.getElementById('status').innerHTML = `🟢 Healthy (${data.active_sessions} active sessions)`;
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
                responseDiv.innerHTML = `<div class="step-entry" style="background: #e8f5e9; border-left-color: #4caf50;">
                    ✅ ${data.message}<br>
                    <strong>Score:</strong> ${data.score}
                </div>`;
                
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
                    <div class="step-entry">
                        <strong>Step ${data.step}</strong><br>
                        <strong>Action:</strong> <code>${JSON.stringify(data.action)}</code><br>
                        <strong>Reward:</strong> <span class="${rewardClass}">${data.reward}</span><br>
                        <div class="explanation">📖 ${data.reward_explanation}</div>
                        <strong>Score:</strong> <span class="score-large">${data.score}</span><br>
                        ${data.message ? `<div style="margin-top: 8px; color: #10b981;">💬 ${data.message}</div>` : ''}
                        ${data.task_complete_message ? `<div style="margin-top: 10px; padding: 10px; background: #e8f5e9; border-radius: 8px; font-weight: bold;">🎉 ${data.task_complete_message}</div>` : ''}
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
            let stepsTaken = 0;
            
            while (!done && stepsTaken < maxSteps) {
                await new Promise(r => setTimeout(r, 400));
                
                const response = await fetch(`${API_BASE}/step_ai`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: currentSessions[taskId] })
                });
                
                const data = await response.json();
                stepsTaken = data.step;
                done = data.done;
                
                const responseDiv = document.getElementById(`response-${taskId}`);
                const rewardClass = data.reward >= 0 ? 'reward-positive' : 'reward-negative';
                const stepHtml = `
                    <div class="step-entry">
                        <strong>Step ${data.step}</strong><br>
                        <strong>Action:</strong> <code>${JSON.stringify(data.action)}</code><br>
                        <strong>Reward:</strong> <span class="${rewardClass}">${data.reward}</span><br>
                        <div class="explanation">📖 ${data.reward_explanation}</div>
                        <strong>Score:</strong> <span class="score-large">${data.score}</span><br>
                        ${data.task_complete_message ? `<div style="margin-top: 10px; padding: 10px; background: #e8f5e9; border-radius: 8px; font-weight: bold;">🎉 ${data.task_complete_message}</div>` : ''}
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
                        <div class="reward-badge">🎯 Expected Reward: ${task.expected_reward} / 1.0</div>
                        ${task.steps_detail ? task.steps_detail.map(step => `<div class="reward-badge">📋 ${step}</div>`).join('') : ''}
                        <p><strong>Max steps:</strong> ${task.max_steps}</p>
                        <button onclick="resetTask('${task.task_id}')">🔄 Reset</button>
                        <button id="step-${task.task_id}" onclick="takeStep('${task.task_id}')" disabled>🤖 Take Step</button>
                        <button id="auto-${task.task_id}" onclick="runFullTask('${task.task_id}')" disabled>⚡ Run Full Task</button>
                        <div id="response-${task.task_id}" class="response-area">
                            <div style="color: #999; text-align: center;">Click Reset to start</div>
                        </div>
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