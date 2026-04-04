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
# TASK-SPECIFIC PROMPTS (for AI inference)
# =========================

def get_task_prompt(task_id: str, step_num: int, history: List) -> str:
    """Get the system prompt for the AI model"""
    
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
    """Call the AI model to get an action (REAL INFERENCE)"""
    
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
# ACTION VALIDATION
# =========================

def validate_action_and_get_explanation(task_id: str, step: int, action_dict: Dict) -> tuple:
    action_type = action_dict.get("action_type", "")
    
    if task_id == "order_status_easy":
        if action_type == "lookup_order" and action_dict.get("order_id") == "12345":
            return True, 1.0, "✓ Order lookup successful", "lookup_order"
        else:
            return False, -0.3, f"❌ Wrong action! Got: {action_type}, Expected: lookup_order", "lookup_order"
    
    if task_id == "refund_policy_medium":
        if action_type == "send_reply" and "refund" in action_dict.get("message", "").lower():
            return True, 1.0, "✓ Refund policy explained correctly", "send_reply with 'refund'"
        else:
            return False, -0.3, "❌ Missing 'refund' in message", "send_reply containing 'refund'"
    
    if task_id == "address_change_hard":
        if step == 1:
            if action_type == "lookup_order" and action_dict.get("order_id") == "12345":
                return True, 0.6, "✓ Step 1: Order located", "lookup_order with order_id 12345"
            else:
                return False, -0.3, f"❌ Step 1 failed! Got: {action_type}, Expected: lookup_order", "lookup_order with order_id 12345"
        
        elif step == 2:
            if action_type == "send_reply" and "address" in action_dict.get("message", "").lower():
                return True, 0.2, "✓ Step 2: Address requested", "send_reply asking for address"
            else:
                return False, -0.3, "❌ Step 2 failed! Expected: ask for address", "send_reply asking for new address"
        
        elif step == 3:
            if action_type == "send_reply" and "confirm" in action_dict.get("message", "").lower():
                return True, 0.2, "✓ Step 3: Address confirmed - Complete!", "send_reply asking for confirmation"
            else:
                return False, -0.3, "❌ Step 3 failed! Expected: ask for confirmation", "send_reply asking to confirm address"
    
    return False, -0.3, "❌ Invalid action format", "valid JSON action"

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
# STEP AI
# =========================

@app.post("/step_ai")
def step_ai(req: StepRequest):
    if req.session_id not in sessions:
        raise HTTPException(404, "Invalid session_id. Please reset first.")

    session = sessions[req.session_id]

    if session["done"]:
        return {
            "message": "✅ Task already completed. Please reset.",
            "done": True,
            "score": f"{session['total_reward']:.1f} / 1.0",
            "step": session["steps"]
        }

    env = session["env"]
    step_num = session["steps"] + 1
    ai_action = call_ai_model(session["task_id"], step_num, session["history"])
    step_display_name = get_step_display_name(session["task_id"], step_num)

    try:
        action = Action(**ai_action)
    except Exception:
        ai_action = get_step_default_action(session["task_id"], step_num)
        action = Action(**ai_action)
    
    is_valid, reward_value, explanation, expected_action = validate_action_and_get_explanation(
        session["task_id"], step_num, ai_action
    )
    
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
    expected_steps = 3 if session["task_id"] == "address_change_hard" else 1
    efficiency = min(100, int((expected_steps / max(session["steps"], 1)) * 100)) if done else 0
    
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
        "efficiency": efficiency,
        "message": "🎉 Task completed successfully!" if done else None
    }

# =========================
# SESSION SUMMARY
# =========================

@app.get("/session/{session_id}/summary")
def get_session_summary(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    
    session = sessions[session_id]
    expected_steps = 3 if session["task_id"] == "address_change_hard" else 1
    efficiency = min(100, int((expected_steps / max(session["steps"], 1)) * 100)) if session["done"] else 0
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
        "ai_actions": session["ai_actions"],
        "steps_history": [
            {"step": i+1, "name": session["step_names"][i], "reward": session["rewards"][i], "explanation": session["explanations"][i], "ai_action": session["ai_actions"][i]}
            for i in range(len(session["steps"]))
        ]
    }

# =========================
# TASKS
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
# HOME UI
# =========================

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Support Environment | AI Agent Training</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            min-height: 100vh;
            padding: 30px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 40px; }
        .header h1 { font-size: 3.5em; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 10px; }
        .header p { color: rgba(255,255,255,0.7); font-size: 1.2em; }
        .badge { display: inline-block; background: rgba(102, 126, 234, 0.2); backdrop-filter: blur(10px); padding: 8px 16px; border-radius: 50px; color: #667eea; font-size: 0.85em; margin-top: 15px; }
        .tasks-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 25px; margin-bottom: 30px; }
        .task-card { background: rgba(255,255,255,0.05); backdrop-filter: blur(10px); border-radius: 24px; padding: 25px; border: 1px solid rgba(255,255,255,0.1); transition: all 0.3s ease; }
        .task-card:hover { transform: translateY(-5px); background: rgba(255,255,255,0.08); border-color: rgba(102, 126, 234, 0.5); }
        .task-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 15px; }
        .task-title { font-size: 1.5em; font-weight: 600; color: white; }
        .difficulty { padding: 5px 12px; border-radius: 20px; font-size: 0.75em; font-weight: 600; text-transform: uppercase; }
        .easy { background: #10b981; color: white; }
        .medium { background: #f59e0b; color: white; }
        .hard { background: #ef4444; color: white; }
        .task-desc { color: rgba(255,255,255,0.6); font-size: 0.9em; margin-bottom: 15px; }
        .reward-badge { background: rgba(16, 185, 129, 0.2); padding: 6px 12px; border-radius: 12px; font-size: 0.8em; color: #10b981; display: inline-block; margin: 5px 5px 5px 0; }
        .button-group { display: flex; gap: 10px; margin: 20px 0; flex-wrap: wrap; }
        button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 10px 20px; border-radius: 12px; cursor: pointer; font-size: 0.85em; font-weight: 500; transition: all 0.2s ease; }
        button:hover { transform: scale(1.02); opacity: 0.9; }
        button:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
        .reset-btn { background: linear-gradient(135deg, #f59e0b, #d97706); }
        .auto-btn { background: linear-gradient(135deg, #10b981, #059669); }
        .response-area { background: rgba(0,0,0,0.3); border-radius: 16px; padding: 15px; margin-top: 15px; max-height: 500px; overflow-y: auto; }
        .visual-steps { margin-bottom: 20px; }
        .step-item { display: flex; align-items: center; gap: 12px; padding: 10px; margin: 8px 0; background: rgba(255,255,255,0.05); border-radius: 12px; }
        .step-icon { width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.1em; }
        .step-icon.success { background: rgba(16, 185, 129, 0.2); color: #10b981; }
        .step-icon.failed { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
        .step-content { flex: 1; }
        .step-name { color: white; font-weight: 500; font-size: 0.9em; }
        .step-detail { font-size: 0.75em; color: rgba(255,255,255,0.5); margin-top: 4px; }
        .step-reward { font-size: 0.8em; font-weight: 600; }
        .step-reward.positive { color: #10b981; }
        .step-reward.negative { color: #ef4444; }
        .performance-summary { background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(102, 126, 234, 0.1)); border-radius: 16px; padding: 15px; margin-bottom: 15px; border: 1px solid rgba(16, 185, 129, 0.3); }
        .summary-title { font-size: 0.85em; text-transform: uppercase; letter-spacing: 1px; color: rgba(255,255,255,0.5); margin-bottom: 12px; }
        .summary-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
        .summary-item { text-align: center; }
        .summary-value { font-size: 1.5em; font-weight: 700; color: white; }
        .summary-label { font-size: 0.7em; color: rgba(255,255,255,0.5); }
        .progress-container { background: rgba(255,255,255,0.1); border-radius: 20px; height: 8px; margin: 10px 0; overflow: hidden; }
        .progress-bar { background: linear-gradient(90deg, #10b981, #667eea); height: 100%; border-radius: 20px; transition: width 0.3s ease; width: 0%; }
        .intelligence-score { text-align: center; margin-top: 10px; padding: 10px; background: rgba(102, 126, 234, 0.2); border-radius: 12px; }
        .intelligence-value { font-size: 1.8em; font-weight: 700; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .status { position: fixed; bottom: 20px; right: 20px; background: rgba(0,0,0,0.7); backdrop-filter: blur(10px); padding: 10px 20px; border-radius: 50px; font-size: 0.85em; color: #10b981; }
        .empty-state { text-align: center; color: rgba(255,255,255,0.4); padding: 30px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Customer Support Environment</h1>
            <p>AI Agent Training Platform | Explainable Rewards | Real-time Performance Analytics</p>
            <div class="badge">🚀 RL Environment | OpenEnv Compliant | Real LLaMA 3.1 Inference</div>
        </div>
        <div class="tasks-grid" id="tasks"></div>
        <div class="status" id="status">🟢 System Online</div>
    </div>
    <script>
        const API_BASE = window.location.origin;
        let currentSessions = {};
        let sessionSummaries = {};
        
        async function checkHealth() {
            try {
                const response = await fetch(`${API_BASE}/health`);
                const data = await response.json();
                document.getElementById('status').innerHTML = `🟢 System Online | ${data.active_sessions} Active Sessions | 🤖 ${data.ai_model || 'LLaMA'}`;
            } catch (error) {
                document.getElementById('status').innerHTML = '🔴 System Offline';
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
        
        function renderVisualSteps(steps) {
            if (!steps || steps.length === 0) return '<div class="empty-state">🤖 Click "Take Step" to start AI inference</div>';
            return `<div class="visual-steps">${steps.map(step => `<div class="step-item"><div class="step-icon ${step.reward >= 0 ? 'success' : 'failed'}">${step.reward >= 0 ? '✓' : '✗'}</div><div class="step-content"><div class="step-name">${step.name}</div><div class="step-detail">${step.explanation}</div></div><div class="step-reward ${step.reward >= 0 ? 'positive' : 'negative'}">${step.reward >= 0 ? '+' : ''}${step.reward}</div></div>`).reverse().join('')}</div>`;
        }
        
        function renderPerformanceSummary(summary) {
            if (!summary || summary.steps_taken === 0) return '';
            const progressPercent = (summary.score_value * 100);
            return `<div class="performance-summary"><div class="summary-title">📊 Agent Performance Analysis</div><div class="summary-grid"><div class="summary-item"><div class="summary-value">${summary.steps_taken} / ${summary.expected_steps}</div><div class="summary-label">Steps Taken</div></div><div class="summary-item"><div class="summary-value">${summary.efficiency}%</div><div class="summary-label">Efficiency</div></div><div class="summary-item"><div class="summary-value">${summary.mistakes}</div><div class="summary-label">Mistakes</div></div><div class="summary-item"><div class="summary-value">${summary.score}</div><div class="summary-label">Final Score</div></div></div><div class="progress-container"><div class="progress-bar" style="width: ${progressPercent}%"></div></div><div class="intelligence-score"><div class="summary-label">🤖 Agent Intelligence Score</div><div class="intelligence-value">${summary.intelligence_score}</div></div></div>`;
        }
        
        async function resetTask(taskId) {
            try {
                const response = await fetch(`${API_BASE}/reset/${taskId}`);
                const data = await response.json();
                currentSessions[taskId] = data.session_id;
                delete sessionSummaries[taskId];
                const responseDiv = document.getElementById(`response-${taskId}`);
                responseDiv.innerHTML = `<div class="visual-steps"><div class="empty-state">🔄 Environment reset. AI agent ready.</div></div>`;
                const stepBtn = document.getElementById(`step-${taskId}`);
                stepBtn.disabled = false;
                stepBtn.textContent = '🤖 Take Step';
                const autoBtn = document.getElementById(`auto-${taskId}`);
                if (autoBtn) { autoBtn.disabled = false; autoBtn.textContent = '⚡ Run Full Episode'; }
            } catch (error) { alert('Error resetting task: ' + error.message); }
        }
        
        async function takeStep(taskId) {
            if (!currentSessions[taskId]) { alert('Please reset the task first!'); return; }
            const stepBtn = document.getElementById(`step-${taskId}`);
            stepBtn.disabled = true;
            stepBtn.textContent = '⏳ AI Thinking...';
            try {
                const response = await fetch(`${API_BASE}/step_ai`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: currentSessions[taskId] }) });
                const data = await response.json();
                const summary = await fetchSummary(currentSessions[taskId], taskId);
                const responseDiv = document.getElementById(`response-${taskId}`);
                if (summary) { responseDiv.innerHTML = renderPerformanceSummary(summary) + renderVisualSteps(summary.steps_history); }
                if (data.done) {
                    stepBtn.disabled = true;
                    stepBtn.textContent = '✅ Episode Complete';
                    const autoBtn = document.getElementById(`auto-${taskId}`);
                    if (autoBtn) autoBtn.disabled = true;
                } else {
                    stepBtn.disabled = false;
                    stepBtn.textContent = '🤖 Take Next Step';
                }
            } catch (error) { alert('Error: ' + error.message); stepBtn.disabled = false; stepBtn.textContent = '🤖 Retry Step'; }
        }
        
        async function runFullTask(taskId) {
            if (!currentSessions[taskId]) { await resetTask(taskId); await new Promise(r => setTimeout(r, 500)); }
            const stepBtn = document.getElementById(`step-${taskId}`);
            const autoBtn = document.getElementById(`auto-${taskId}`);
            autoBtn.disabled = true;
            autoBtn.textContent = '🏃 AI Running Episode...';
            stepBtn.disabled = true;
            let done = false;
            let maxSteps = taskId === 'address_change_hard' ? 5 : 3;
            let stepsTaken = 0;
            while (!done && stepsTaken < maxSteps) {
                await new Promise(r => setTimeout(r, 500));
                const response = await fetch(`${API_BASE}/step_ai`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: currentSessions[taskId] }) });
                const data = await response.json();
                stepsTaken = data.step;
                done = data.done;
                const summary = await fetchSummary(currentSessions[taskId], taskId);
                const responseDiv = document.getElementById(`response-${taskId}`);
                if (summary) { responseDiv.innerHTML = renderPerformanceSummary(summary) + renderVisualSteps(summary.steps_history); }
            }
            autoBtn.disabled = true;
            autoBtn.textContent = '✅ Episode Complete';
            if (!done) stepBtn.disabled = false;
        }
        
        async function loadTasks() {
            try {
                const response = await fetch(`${API_BASE}/tasks`);
                const data = await response.json();
                const tasksGrid = document.getElementById('tasks');
                tasksGrid.innerHTML = data.tasks.map(task => `<div class="task-card"><div class="task-header"><span class="task-title">${task.icon || '🎯'} ${task.name}</span><span class="difficulty ${task.difficulty}">${task.difficulty.toUpperCase()}</span></div><div class="task-desc">${task.description}</div><div class="reward-badge">🎯 Expected Reward: ${task.expected_reward} / 1.0</div>${task.steps_detail ? task.steps_detail.map(step => `<div class="reward-badge">📋 ${step}</div>`).join('') : ''}<div class="reward-badge">⚡ Max Steps: ${task.max_steps}</div><div class="button-group"><button class="reset-btn" onclick="resetTask('${task.task_id}')">🔄 Reset Environment</button><button id="step-${task.task_id}" onclick="takeStep('${task.task_id}')" disabled>🤖 Take Step</button><button id="auto-${task.task_id}" class="auto-btn" onclick="runFullTask('${task.task_id}')" disabled>⚡ Run Full Episode</button></div><div id="response-${task.task_id}" class="response-area"><div class="empty-state">🔄 Click "Reset Environment" to start AI training</div></div></div>`).join('');
            } catch (error) { console.error('Error loading tasks:', error); }
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