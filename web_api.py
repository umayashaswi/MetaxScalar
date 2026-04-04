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
# TASK-SPECIFIC PROMPTS
# =========================

def get_task_prompt(task_id: str, step_num: int, history: List) -> str:
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
# HOME UI - CLEAN & ORGANIZED
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
        
        /* Header */
        .header { text-align: center; margin-bottom: 40px; }
        .header h1 { font-size: 3em; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 10px; }
        .header p { color: rgba(255,255,255,0.7); font-size: 1.1em; }
        .badge { display: inline-block; background: rgba(102, 126, 234, 0.2); backdrop-filter: blur(10px); padding: 6px 14px; border-radius: 50px; color: #667eea; font-size: 0.8em; margin-top: 12px; }
        
        /* Tasks Grid */
        .tasks-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 25px; margin-bottom: 30px; }
        
        /* Task Card */
        .task-card { background: rgba(255,255,255,0.06); backdrop-filter: blur(10px); border-radius: 20px; padding: 20px; border: 1px solid rgba(255,255,255,0.1); transition: all 0.3s ease; }
        .task-card:hover { transform: translateY(-3px); background: rgba(255,255,255,0.08); border-color: rgba(102, 126, 234, 0.4); }
        .task-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
        .task-title { font-size: 1.3em; font-weight: 600; color: white; }
        .difficulty { padding: 4px 10px; border-radius: 20px; font-size: 0.7em; font-weight: 600; text-transform: uppercase; }
        .easy { background: #10b981; color: white; }
        .medium { background: #f59e0b; color: white; }
        .hard { background: #ef4444; color: white; }
        .task-desc { color: rgba(255,255,255,0.5); font-size: 0.85em; margin-bottom: 12px; }
        .reward-badge { background: rgba(16, 185, 129, 0.15); padding: 4px 10px; border-radius: 10px; font-size: 0.75em; color: #10b981; display: inline-block; margin: 3px 3px 3px 0; }
        
        /* Buttons */
        .button-group { display: flex; gap: 10px; margin: 15px 0; flex-wrap: wrap; }
        button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 8px 18px; border-radius: 10px; cursor: pointer; font-size: 0.8em; font-weight: 500; transition: all 0.2s ease; }
        button:hover { transform: scale(1.02); opacity: 0.9; }
        button:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
        .reset-btn { background: linear-gradient(135deg, #f59e0b, #d97706); }
        .auto-btn { background: linear-gradient(135deg, #10b981, #059669); }
        
        /* Response Area */
        .response-area { background: rgba(0,0,0,0.3); border-radius: 14px; padding: 12px; margin-top: 12px; max-height: 450px; overflow-y: auto; }
        
        /* Performance Summary Card */
        .perf-summary { background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(102, 126, 234, 0.12)); border-radius: 14px; padding: 14px; margin-bottom: 15px; border: 1px solid rgba(16, 185, 129, 0.25); }
        .perf-title { font-size: 0.75em; text-transform: uppercase; letter-spacing: 1px; color: rgba(255,255,255,0.5); margin-bottom: 10px; }
        .perf-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; text-align: center; }
        .perf-value { font-size: 1.2em; font-weight: 700; color: white; }
        .perf-label { font-size: 0.65em; color: rgba(255,255,255,0.5); }
        .progress-bar-container { background: rgba(255,255,255,0.1); border-radius: 20px; height: 6px; margin: 10px 0; overflow: hidden; }
        .progress-fill { background: linear-gradient(90deg, #10b981, #667eea); height: 100%; border-radius: 20px; transition: width 0.3s ease; width: 0%; }
        .intel-score { text-align: center; margin-top: 10px; padding: 8px; background: rgba(102, 126, 234, 0.2); border-radius: 10px; }
        .intel-value { font-size: 1.3em; font-weight: 700; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        
        /* Step Item */
        .step-item { display: flex; align-items: center; gap: 10px; padding: 8px; margin: 6px 0; background: rgba(255,255,255,0.04); border-radius: 10px; }
        .step-icon { width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.9em; }
        .step-icon.success { background: rgba(16, 185, 129, 0.2); color: #10b981; }
        .step-icon.failed { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
        .step-content { flex: 1; }
        .step-name { color: white; font-weight: 500; font-size: 0.85em; }
        .step-detail { font-size: 0.7em; color: rgba(255,255,255,0.45); margin-top: 2px; }
        .step-reward { font-size: 0.75em; font-weight: 600; }
        .step-reward.positive { color: #10b981; }
        .step-reward.negative { color: #ef4444; }
        
        .empty-state { text-align: center; color: rgba(255,255,255,0.35); padding: 25px; font-size: 0.85em; }
        .status { position: fixed; bottom: 15px; right: 15px; background: rgba(0,0,0,0.6); backdrop-filter: blur(8px); padding: 6px 14px; border-radius: 30px; font-size: 0.75em; color: #10b981; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Customer Support Environment</h1>
            <p>AI Agent Training | Explainable Rewards | Real-time Performance Analytics</p>
            <div class="badge">🚀 OpenEnv Compliant | Real LLaMA 3.1 Inference</div>
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
                const res = await fetch(`${API_BASE}/health`);
                const data = await res.json();
                document.getElementById('status').innerHTML = `🟢 Online | ${data.active_sessions} Sessions | 🤖 ${data.ai_model || 'LLaMA'}`;
            } catch(e) { document.getElementById('status').innerHTML = '🔴 Offline'; }
        }
        
        async function fetchSummary(sessionId, taskId) {
            try {
                const res = await fetch(`${API_BASE}/session/${sessionId}/summary`);
                const data = await res.json();
                sessionSummaries[taskId] = data;
                return data;
            } catch(e) { return null; }
        }
        
        function renderPerfSummary(summary) {
            if (!summary || summary.steps_taken === 0) return '';
            const progress = summary.score_value * 100;
            return `<div class="perf-summary">
                <div class="perf-title">📊 PERFORMANCE SUMMARY</div>
                <div class="perf-grid">
                    <div><div class="perf-value">${summary.steps_taken}/${summary.expected_steps}</div><div class="perf-label">STEPS</div></div>
                    <div><div class="perf-value">${summary.efficiency}%</div><div class="perf-label">EFFICIENCY</div></div>
                    <div><div class="perf-value">${summary.mistakes}</div><div class="perf-label">MISTAKES</div></div>
                    <div><div class="perf-value">${summary.score}</div><div class="perf-label">SCORE</div></div>
                </div>
                <div class="progress-bar-container"><div class="progress-fill" style="width: ${progress}%"></div></div>
                <div class="intel-score"><div class="perf-label">🤖 AGENT INTELLIGENCE SCORE</div><div class="intel-value">${summary.intelligence_score}</div></div>
            </div>`;
        }
        
        function renderSteps(steps) {
            if (!steps || steps.length === 0) return '<div class="empty-state">🤖 Click "Take Step" to start AI inference</div>';
            return steps.map(s => `<div class="step-item">
                <div class="step-icon ${s.reward >= 0 ? 'success' : 'failed'}">${s.reward >= 0 ? '✓' : '✗'}</div>
                <div class="step-content"><div class="step-name">${s.name}</div><div class="step-detail">${s.explanation}</div></div>
                <div class="step-reward ${s.reward >= 0 ? 'positive' : 'negative'}">${s.reward >= 0 ? '+' : ''}${s.reward}</div>
            </div>`).reverse().join('');
        }
        
        async function resetTask(taskId) {
            try {
                const res = await fetch(`${API_BASE}/reset/${taskId}`);
                const data = await res.json();
                currentSessions[taskId] = data.session_id;
                delete sessionSummaries[taskId];
                document.getElementById(`response-${taskId}`).innerHTML = '<div class="empty-state">🔄 Environment reset. AI agent ready.</div>';
                const stepBtn = document.getElementById(`step-${taskId}`);
                stepBtn.disabled = false;
                stepBtn.textContent = '🤖 Take Step';
                const autoBtn = document.getElementById(`auto-${taskId}`);
                if (autoBtn) { autoBtn.disabled = false; autoBtn.textContent = '⚡ Run Full Episode'; }
            } catch(e) { alert('Reset error: ' + e.message); }
        }
        
        async function takeStep(taskId) {
            if (!currentSessions[taskId]) { alert('Please reset first!'); return; }
            const stepBtn = document.getElementById(`step-${taskId}`);
            stepBtn.disabled = true;
            stepBtn.textContent = '⏳ Thinking...';
            try {
                await fetch(`${API_BASE}/step_ai`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: currentSessions[taskId] }) });
                const summary = await fetchSummary(currentSessions[taskId], taskId);
                const div = document.getElementById(`response-${taskId}`);
                if (summary) div.innerHTML = renderPerfSummary(summary) + renderSteps(summary.steps_history);
                if (summary?.completed) {
                    stepBtn.disabled = true;
                    stepBtn.textContent = '✅ Complete';
                    const autoBtn = document.getElementById(`auto-${taskId}`);
                    if (autoBtn) autoBtn.disabled = true;
                } else {
                    stepBtn.disabled = false;
                    stepBtn.textContent = '🤖 Next Step';
                }
            } catch(e) { alert('Step error: ' + e.message); stepBtn.disabled = false; stepBtn.textContent = '🤖 Retry'; }
        }
        
        async function runFullTask(taskId) {
            if (!currentSessions[taskId]) { await resetTask(taskId); await new Promise(r => setTimeout(r, 400)); }
            const stepBtn = document.getElementById(`step-${taskId}`);
            const autoBtn = document.getElementById(`auto-${taskId}`);
            autoBtn.disabled = true;
            autoBtn.textContent = '🏃 Running...';
            stepBtn.disabled = true;
            let done = false, maxSteps = taskId === 'address_change_hard' ? 5 : 3, iter = 0;
            while (!done && iter < maxSteps) {
                await new Promise(r => setTimeout(r, 450));
                await fetch(`${API_BASE}/step_ai`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: currentSessions[taskId] }) });
                const summary = await fetchSummary(currentSessions[taskId], taskId);
                const div = document.getElementById(`response-${taskId}`);
                if (summary) div.innerHTML = renderPerfSummary(summary) + renderSteps(summary.steps_history);
                done = summary?.completed || false;
                iter++;
            }
            autoBtn.disabled = true;
            autoBtn.textContent = '✅ Complete';
            if (!done) stepBtn.disabled = false;
        }
        
        async function loadTasks() {
            try {
                const res = await fetch(`${API_BASE}/tasks`);
                const data = await res.json();
                const grid = document.getElementById('tasks');
                grid.innerHTML = data.tasks.map(t => `<div class="task-card">
                    <div class="task-header"><span class="task-title">${t.icon} ${t.name}</span><span class="difficulty ${t.difficulty}">${t.difficulty.toUpperCase()}</span></div>
                    <div class="task-desc">${t.description}</div>
                    <div class="reward-badge">🎯 Expected Reward: ${t.expected_reward} / 1.0</div>
                    ${t.steps_detail ? t.steps_detail.map(s => `<div class="reward-badge">📋 ${s}</div>`).join('') : ''}
                    <div class="reward-badge">⚡ Max Steps: ${t.max_steps}</div>
                    <div class="button-group"><button class="reset-btn" onclick="resetTask('${t.task_id}')">🔄 Reset</button><button id="step-${t.task_id}" onclick="takeStep('${t.task_id}')" disabled>🤖 Take Step</button><button id="auto-${t.task_id}" class="auto-btn" onclick="runFullTask('${t.task_id}')" disabled>⚡ Run Full Episode</button></div>
                    <div id="response-${t.task_id}" class="response-area"><div class="empty-state">🔄 Click "Reset" to start</div></div>
                </div>`).join('');
            } catch(e) { console.error(e); }
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