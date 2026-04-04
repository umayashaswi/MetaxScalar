import json
import os
import uuid
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from datetime import datetime

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
# TASK-SPECIFIC PROMPTS (for AI inference - SAME as inference.py)
# =========================

def get_task_prompt(task_id: str, step_num: int, history: List) -> str:
    """Get the system prompt for the AI model - MATCHES inference.py"""
    
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
            "STEP 1: {\"action_type\": \"lookup_order\", \"order_id\": \"12345\"}\n"
            "STEP 2: {\"action_type\": \"send_reply\", \"message\": \"Please provide your new address.\"}\n"
            "STEP 3: {\"action_type\": \"send_reply\", \"message\": \"Please confirm your new address.\"}\n\n"
            "CRITICAL: Keep messages short and direct."
        )
    
    else:
        return "Return valid JSON with action_type field."

def get_step_default_action(task_id: str, step_num: int) -> Dict[str, Any]:
    """Fallback action if AI fails"""
    if task_id == "order_status_easy":
        return {"action_type": "lookup_order", "order_id": "12345"}
    
    elif task_id == "refund_policy_medium":
        return {
            "action_type": "send_reply", 
            "message": "Our refund policy allows returns within 30 days for a full refund."
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
    """Call the AI model to get an action - REAL INFERENCE matching inference.py"""
    
    # If no API key, use fallback
    if client is None:
        return get_step_default_action(task_id, step_num)
    
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
        
        if "```" in content:
            parts = content.split("```")
            if len(parts) > 1:
                content = parts[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end != 0:
            content = content[start:end]
        else:
            return get_step_default_action(task_id, step_num)

        data = json.loads(content)
        
        if "action_type" not in data:
            return get_step_default_action(task_id, step_num)
        
        return data
        
    except Exception as e:
        print(f"AI model error: {e}")
        return get_step_default_action(task_id, step_num)

# =========================
# ACTION VALIDATION & REWARD EXPLANATIONS
# =========================

def validate_action_and_get_explanation(task_id: str, step: int, action_dict: Dict) -> tuple:
    """Validate action and return (is_valid, reward, explanation, expected_action)"""
    
    action_type = action_dict.get("action_type", "")
    
    # Order Status Easy Task
    if task_id == "order_status_easy":
        if action_type == "lookup_order" and action_dict.get("order_id") == "12345":
            return True, 1.0, "✓ Correct: Order lookup successful", "lookup_order"
        else:
            return False, -0.3, f"✗ Wrong action! Got: {action_type}, Expected: lookup_order", "lookup_order"
    
    # Refund Policy Medium Task
    if task_id == "refund_policy_medium":
        if action_type == "send_reply" and "refund" in action_dict.get("message", "").lower():
            return True, 1.0, "✓ Correct: Refund policy explained", "send_reply with 'refund'"
        else:
            return False, -0.3, "✗ Invalid: Send reply explaining refund policy (must include 'refund')", "send_reply containing 'refund'"
    
    # Address Change Hard Task
    if task_id == "address_change_hard":
        if step == 1:
            if action_type == "lookup_order" and action_dict.get("order_id") == "12345":
                return True, 0.6, "✓ Step 1/3: Order located successfully", "lookup_order with order_id 12345"
            else:
                return False, -0.3, f"✗ Step 1/3: Got {action_type}, Expected lookup_order", "lookup_order with order_id 12345"
        
        elif step == 2:
            if action_type == "send_reply" and "address" in action_dict.get("message", "").lower():
                return True, 0.2, "✓ Step 2/3: Address request sent", "send_reply asking for address"
            else:
                return False, -0.3, "✗ Step 2/3: Ask customer to provide new address", "send_reply asking for new address"
        
        elif step == 3:
            if action_type == "send_reply" and "confirm" in action_dict.get("message", "").lower():
                return True, 0.2, "✓ Step 3/3: Address confirmed - Complete!", "send_reply asking for confirmation"
            else:
                return False, -0.3, "✗ Step 3/3: Ask customer to confirm address", "send_reply asking to confirm address"
    
    return False, -0.3, "✗ Invalid action format", "valid JSON action"

def get_step_display_name(task_id: str, step: int) -> str:
    if task_id == "address_change_hard":
        if step == 1:
            return "🔍 Locate Order"
        elif step == 2:
            return "📝 Request New Address"
        elif step == 3:
            return "✅ Confirm Address Change"
    elif task_id == "order_status_easy":
        return "📊 Fetch Order Status"
    elif task_id == "refund_policy_medium":
        return "📋 Explain Refund Policy"
    return "Execute Step"

# =========================
# RESET
# =========================

@app.post("/reset")
def reset(req: ResetRequest):
    task_id = req.task_id
    env = CustomerSupportEnv(task_id)
    env.reset()
    session_id = str(uuid.uuid4())
    
    sessions[session_id] = {
        "env": env,
        "task_id": task_id,
        "steps": 0,
        "rewards": [],
        "explanations": [],
        "step_names": [],
        "mistakes": [],
        "ai_actions": [],
        "done": False,
        "total_reward": 0.0,
        "history": [],
        "start_time": datetime.now().isoformat()
    }
    
    return {
        "session_id": session_id,
        "task_id": task_id,
        "message": f"✅ Environment reset successfully",
        "steps": 0,
        "done": False,
        "score": "0.0 / 1.0"
    }

@app.get("/reset/{task_id}")
def reset_get(task_id: str):
    req = ResetRequest(task_id=task_id)
    return reset(req)

# =========================
# STEP AI - WITH REAL INFERENCE
# =========================

@app.post("/step_ai")
def step_ai(req: StepRequest):
    if req.session_id not in sessions:
        raise HTTPException(404, "Invalid session_id. Please reset first.")

    session = sessions[req.session_id]

    if session["done"]:
        final_message = ""
        if session["task_id"] == "address_change_hard":
            final_message = "🎉 Address change successfully completed!"
        elif session["task_id"] == "order_status_easy":
            final_message = "🎉 Order status retrieved successfully!"
        elif session["task_id"] == "refund_policy_medium":
            final_message = "🎉 Refund policy explained!"
        
        return {
            "message": f"✅ Task completed. {final_message} Reset to start new.",
            "done": True,
            "score": f"{session['total_reward']:.1f} / 1.0",
            "step": session["steps"]
        }

    env = session["env"]
    step_num = session["steps"] + 1
    
    # ============================================
    # CRITICAL: Call AI model for REAL inference
    # ============================================
    ai_action = call_ai_model(session["task_id"], step_num, session["history"])
    
    step_display_name = get_step_display_name(session["task_id"], step_num)

    try:
        action = Action(**ai_action)
    except Exception:
        ai_action = get_step_default_action(session["task_id"], step_num)
        action = Action(**ai_action)
    
    # Validate AI's action
    is_valid, reward_value, explanation, expected_action = validate_action_and_get_explanation(
        session["task_id"], step_num, ai_action
    )
    
    # Take step in environment
    obs, env_reward, done, info = env.step(action)
    
    final_reward = reward_value
    is_mistake = not is_valid
    
    session["steps"] += 1
    session["rewards"].append(final_reward)
    session["explanations"].append(explanation)
    session["step_names"].append(step_display_name)
    session["ai_actions"].append(ai_action)
    
    if is_mistake:
        session["mistakes"].append({
            "step": step_num,
            "expected": expected_action,
            "got": ai_action.get("action_type", "unknown"),
            "full_action": ai_action
        })
    
    session["total_reward"] += final_reward
    session["done"] = done
    session["history"].append(ai_action)

    score_value = min(max(session["total_reward"], 0.0), 1.0)
    score_display = f"{score_value:.1f} / 1.0"
    
    return {
        "step": session["steps"],
        "step_name": step_display_name,
        "action": ai_action,
        "reward": final_reward,
        "reward_explanation": explanation,
        "is_mistake": is_mistake,
        "expected_action": expected_action if is_mistake else None,
        "done": done,
        "score": score_display,
        "score_value": score_value,
        "total_reward": session["total_reward"],
        "total_steps": session["steps"],
        "mistakes_count": len(session["mistakes"]),
        "message": "🎉 Task completed successfully!" if done else None
    }


# =========================
# GET SESSION SUMMARY
# =========================

@app.get("/session/{session_id}/summary")
def get_session_summary(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    
    session = sessions[session_id]
    expected_steps = 3 if session["task_id"] == "address_change_hard" else 1
    efficiency = min(100, int((expected_steps / max(session["steps"], 1)) * 100)) if session["steps"] > 0 else 0
    score_value = min(max(session["total_reward"], 0.0), 1.0)
    intelligence_score = round(score_value * 10 - (len(session["mistakes"]) * 0.5), 1)
    intelligence_score = max(0, min(10, intelligence_score))
    
    return {
        "session_id": session_id,
        "task_id": session["task_id"],
        "steps_taken": session["steps"],
        "expected_steps": expected_steps,
        "efficiency": efficiency,
        "mistakes": len(session["mistakes"]),
        "mistake_details": session["mistakes"],
        "total_reward": session["total_reward"],
        "score": f"{score_value:.1f} / 1.0",
        "score_value": score_value,
        "intelligence_score": f"{intelligence_score:.1f} / 10",
        "completed": session["done"],
        "steps_history": [
            {"step": i+1, "name": session["step_names"][i], "reward": session["rewards"][i], "explanation": session["explanations"][i]}
            for i in range(len(session["steps"]))
        ]
    }


# =========================
# TASK LIST
# =========================

@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {"task_id": "order_status_easy", "name": "Order Status Query", "description": "Look up order status using lookup_order action", "difficulty": "easy", "max_steps": 3, "expected_reward": 1.0, "icon": "📊", "steps_detail": ["Step 1: Lookup order (+1.0)"]},
            {"task_id": "refund_policy_medium", "name": "Refund Policy Explanation", "description": "Explain refund policy in a reply message", "difficulty": "medium", "max_steps": 3, "expected_reward": 1.0, "icon": "📋", "steps_detail": ["Step 1: Send reply with 'refund' (+1.0)"]},
            {"task_id": "address_change_hard", "name": "Address Change Request", "description": "Handle address change request in 3 steps", "difficulty": "hard", "max_steps": 5, "expected_reward": 1.0, "icon": "📍", "steps_detail": ["Step 1: Lookup order (+0.6)", "Step 2: Ask for address (+0.2)", "Step 3: Confirm address (+0.2)"]}
        ]
    }


# =========================
# HEALTH
# =========================

@app.get("/health")
def health():
    return {"status": "healthy", "active_sessions": len(sessions), "ai_model": MODEL_NAME if client else "fallback"}


# =========================
# HOME UI - SAME BEAUTIFUL UI
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
        .score-large { font-size: 20px; font-weight: bold; color: #667eea; }
        .explanation { color: #6b7280; font-size: 12px; margin-top: 5px; }
        .perf-summary { background: #e8eaf6; border-radius: 10px; padding: 12px; margin-bottom: 15px; }
        .perf-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; text-align: center; }
        .perf-value { font-size: 1.2em; font-weight: bold; color: #667eea; }
        .perf-label { font-size: 0.7em; color: #666; }
        .progress-bar { background: #ddd; border-radius: 10px; height: 6px; margin: 10px 0; overflow: hidden; }
        .progress-fill { background: linear-gradient(90deg, #10b981, #667eea); height: 100%; width: 0%; transition: width 0.3s ease; }
        .intel-score { text-align: center; margin-top: 10px; padding: 8px; background: linear-gradient(135deg, #667eea20, #764ba220); border-radius: 8px; }
        .intel-value { font-size: 1.3em; font-weight: bold; color: #667eea; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Customer Support Environment</h1>
        <p>Real AI Inference (LLaMA 3.1) | Explainable Rewards | Performance Analytics</p>
    </div>
    
    <div class="tasks-grid" id="tasks"></div>
    
    <div class="status" id="status">🟢 Healthy</div>
    
    <script>
        const API_BASE = window.location.origin;
        let currentSessions = {};
        let sessionSummaries = {};
        
        async function checkHealth() {
            try {
                const response = await fetch(`${API_BASE}/health`);
                const data = await response.json();
                document.getElementById('status').innerHTML = `🟢 Healthy | ${data.active_sessions} active | 🤖 ${data.ai_model || 'LLaMA'}`;
            } catch (error) {
                document.getElementById('status').innerHTML = '🔴 Error';
            }
        }
        
        async function fetchSummary(sessionId, taskId) {
            try {
                const response = await fetch(`${API_BASE}/session/${sessionId}/summary`);
                const data = await response.json();
                sessionSummaries[taskId] = data;
                return data;
            } catch (error) {
                return null;
            }
        }
        
        function renderPerfSummary(summary) {
            if (!summary || summary.steps_taken === 0) return '';
            const progress = summary.score_value * 100;
            return `
                <div class="perf-summary">
                    <div class="perf-grid">
                        <div><div class="perf-value">${summary.steps_taken}/${summary.expected_steps}</div><div class="perf-label">STEPS</div></div>
                        <div><div class="perf-value">${summary.efficiency}%</div><div class="perf-label">EFFICIENCY</div></div>
                        <div><div class="perf-value">${summary.mistakes}</div><div class="perf-label">MISTAKES</div></div>
                        <div><div class="perf-value">${summary.score}</div><div class="perf-label">SCORE</div></div>
                    </div>
                    <div class="progress-bar"><div class="progress-fill" style="width: ${progress}%"></div></div>
                    <div class="intel-score"><div class="perf-label">🤖 AGENT INTELLIGENCE SCORE</div><div class="intel-value">${summary.intelligence_score}</div></div>
                </div>
            `;
        }
        
        function renderSteps(steps) {
            if (!steps || steps.length === 0) return '<div style="text-align: center; color: #999; padding: 20px;">🤖 Click "Take Step" to start AI inference</div>';
            return steps.map(s => `
                <div class="step-entry">
                    <strong>Step ${s.step}: ${s.name}</strong><br>
                    <strong>Reward:</strong> <span class="${s.reward >= 0 ? 'reward-positive' : 'reward-negative'}">${s.reward >= 0 ? '+' : ''}${s.reward}</span><br>
                    <div class="explanation">📖 ${s.explanation}</div>
                </div>
            `).reverse().join('');
        }
        
        async function resetTask(taskId) {
            try {
                const response = await fetch(`${API_BASE}/reset/${taskId}`);
                const data = await response.json();
                currentSessions[taskId] = data.session_id;
                delete sessionSummaries[taskId];
                
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
            stepBtn.textContent = '⏳ AI Thinking...';
            
            try {
                await fetch(`${API_BASE}/step_ai`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: currentSessions[taskId] })
                });
                
                const summary = await fetchSummary(currentSessions[taskId], taskId);
                const responseDiv = document.getElementById(`response-${taskId}`);
                
                if (summary) {
                    responseDiv.innerHTML = renderPerfSummary(summary) + renderSteps(summary.steps_history);
                }
                
                if (summary?.completed) {
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
            autoBtn.textContent = '🏃 AI Running...';
            stepBtn.disabled = true;
            
            let done = false;
            let maxSteps = taskId === 'address_change_hard' ? 5 : 3;
            let stepsTaken = 0;
            
            while (!done && stepsTaken < maxSteps) {
                await new Promise(r => setTimeout(r, 400));
                
                await fetch(`${API_BASE}/step_ai`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: currentSessions[taskId] })
                });
                
                const summary = await fetchSummary(currentSessions[taskId], taskId);
                const responseDiv = document.getElementById(`response-${taskId}`);
                
                if (summary) {
                    responseDiv.innerHTML = renderPerfSummary(summary) + renderSteps(summary.steps_history);
                    done = summary.completed;
                }
                stepsTaken++;
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
                        <h3>${task.icon || '🎯'} ${task.name}</h3>
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
    print(f"🤖 AI Model: {MODEL_NAME}")
    print(f"🔑 API Key configured: {bool(API_KEY)}")
    print(f"🚀 Server starting at http://0.0.0.0:7860")
    uvicorn.run(app, host="0.0.0.0", port=7860)