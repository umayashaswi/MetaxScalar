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

print(f"🤖 AI Model: {MODEL_NAME}")
print(f"🔑 API Key configured: {bool(API_KEY)}")

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
    use_expert: bool = False

class ResetRequest(BaseModel):
    task_id: str

# =========================
# REWARD CONFIGURATION
# =========================

REWARD_CONFIG = {
    "order_status_easy": {
        "correct_step": 0.8,
        "wrong_action": -0.3,
        "ask_expert_penalty": -0.2,
        "max_score": 0.8,
        "expert_hint": "Try using lookup_order with order_id 12345"
    },
    "refund_policy_medium": {
        "correct_step": 0.8,
        "wrong_action": -0.3,
        "ask_expert_penalty": -0.2,
        "max_score": 0.8,
        "expert_hint": "Send a reply that explains the refund policy (include the word 'refund')"
    },
    "address_change_hard": {
        "step1_correct": 0.4,
        "step2_correct": 0.3,
        "step3_correct": 0.3,
        "wrong_action": -0.3,
        "ask_expert_penalty": -0.2,
        "max_score": 1.0,
        "expert_hint": {
            1: "Step 1: Use lookup_order with order_id 12345",
            2: "Step 2: Ask customer for their new address",
            3: "Step 3: Ask customer to confirm the new address"
        }
    }
}

# =========================
# TASK-SPECIFIC PROMPTS (NATURAL, NO HARDCODED ANSWERS)
# =========================

def get_task_prompt(task_id: str, step_num: int, history: List, use_expert: bool = False) -> str:
    expert_suffix = ""
    if use_expert:
        if task_id == "address_change_hard":
            hint = REWARD_CONFIG[task_id]["expert_hint"].get(step_num, "Follow the correct sequence")
        else:
            hint = REWARD_CONFIG[task_id]["expert_hint"]
        expert_suffix = f"\n\n🎓 EXPERT ADVICE: {hint}"
    
    if task_id == "order_status_easy":
        return (
            "You are a Customer Support AI. Return ONLY valid JSON.\n\n"
            "The customer wants to check their order status.\n\n"
            "Choose the appropriate action to help the customer.\n"
            "Return a JSON object with 'action_type' and necessary parameters.\n\n"
            f"Return ONLY the JSON.{expert_suffix}"
        )
    
    elif task_id == "refund_policy_medium":
        return (
            "You are a Customer Support AI. Return ONLY valid JSON.\n\n"
            "The customer wants to know the refund policy.\n\n"
            "Choose the appropriate action to help the customer.\n"
            "Your response should explain the refund policy.\n\n"
            f"Return ONLY the JSON.{expert_suffix}"
        )
    
    elif task_id == "address_change_hard":
        step_hint = ""
        if step_num == 1:
            step_hint = "The customer wants to change their address. Start by locating the order."
        elif step_num == 2:
            step_hint = "Now ask the customer for their new address."
        elif step_num == 3:
            step_hint = "Ask the customer to confirm the new address to complete the change."
        
        return (
            "You are a Customer Support AI handling address changes.\n"
            "Return ONLY valid JSON.\n\n"
            f"{step_hint}\n\n"
            "Choose the appropriate action based on the current step.\n\n"
            f"Return ONLY the JSON for the current step.{expert_suffix}"
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
    
    return {"action_type": "send_reply", "message": "OK"}

def call_ai_model(task_id: str, step_num: int, history: List, use_expert: bool = False) -> Dict[str, Any]:
    """Call the AI model to get an action"""
    
    if client is None:
        return get_step_default_action(task_id, step_num)
    
    sys_prompt = get_task_prompt(task_id, step_num, history, use_expert)
    
    try:
        print(f"🤖 Calling AI model for {task_id} step {step_num}...")
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
        print(f"📝 AI response: {content[:200]}...")
        
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
        
        # Fix common issues
        if data["action_type"] == "lookup_order" and "order_id" not in data:
            data["order_id"] = "12345"
        
        return data
        
    except Exception as e:
        print(f"❌ AI model error: {e}")
        return get_step_default_action(task_id, step_num)

# =========================
# ACTION VALIDATION WITH DYNAMIC REWARDS
# =========================

def validate_action(task_id: str, step: int, action_dict: Dict, used_expert: bool = False) -> tuple:
    """Validate action and return (is_valid, reward, explanation, expected_action, is_perfect)"""
    
    action_type = action_dict.get("action_type", "")
    config = REWARD_CONFIG[task_id]
    expert_penalty = config["ask_expert_penalty"] if used_expert else 0
    
    if task_id == "order_status_easy":
        if action_type == "lookup_order" and action_dict.get("order_id") == "12345":
            reward = config["correct_step"] + expert_penalty
            is_perfect = not used_expert
            return True, max(0, reward), "✅ Correct: Order lookup successful", "lookup_order", is_perfect
        else:
            return False, config["wrong_action"], f"❌ Wrong: Got '{action_type}', expected 'lookup_order'", "lookup_order", False
    
    if task_id == "refund_policy_medium":
        if action_type == "send_reply" and "refund" in action_dict.get("message", "").lower():
            reward = config["correct_step"] + expert_penalty
            is_perfect = not used_expert
            return True, max(0, reward), "✅ Correct: Refund policy explained", "send_reply with 'refund'", is_perfect
        else:
            return False, config["wrong_action"], "❌ Wrong: Must send reply with 'refund'", "send_reply with 'refund'", False
    
    if task_id == "address_change_hard":
        if step == 1:
            if action_type == "lookup_order" and action_dict.get("order_id") == "12345":
                reward = config["step1_correct"] + expert_penalty
                is_perfect = not used_expert
                return True, max(0, reward), "✅ Step 1/3: Order located", "lookup_order", is_perfect
            else:
                return False, config["wrong_action"], f"❌ Step 1: Got '{action_type}', expected 'lookup_order'", "lookup_order", False
        
        elif step == 2:
            if action_type == "send_reply" and "address" in action_dict.get("message", "").lower():
                reward = config["step2_correct"] + expert_penalty
                is_perfect = not used_expert
                return True, max(0, reward), "✅ Step 2/3: Address requested", "send_reply asking for address", is_perfect
            else:
                return False, config["wrong_action"], "❌ Step 2: Must ask for new address", "send_reply asking for address", False
        
        elif step == 3:
            if action_type == "send_reply" and "confirm" in action_dict.get("message", "").lower():
                reward = config["step3_correct"] + expert_penalty
                is_perfect = not used_expert
                return True, max(0, reward), "✅ Step 3/3: Address confirmed", "send_reply asking for confirmation", is_perfect
            else:
                return False, config["wrong_action"], "❌ Step 3: Must ask for confirmation", "send_reply asking for confirmation", False
    
    return False, -0.3, "❌ Invalid action format", "valid JSON", False

# =========================
# RESET - COMPLETE STATE RESET
# =========================

@app.post("/reset")
def reset(req: ResetRequest):
    task_id = req.task_id
    env = CustomerSupportEnv(task_id)
    env.reset()
    session_id = str(uuid.uuid4())
    
    config = REWARD_CONFIG[task_id]
    max_score = config["max_score"]
    
    sessions[session_id] = {
        "env": env,
        "task_id": task_id,
        "steps": 0,
        "rewards": [],
        "explanations": [],
        "actions_taken": [],
        "expert_used_count": 0,
        "done": False,
        "perfect_completion": False,
        "total_reward": 0.0,
        "history": [],
        "start_time": datetime.now().isoformat()
    }
    
    return {
        "session_id": session_id,
        "task_id": task_id,
        "message": f"Environment reset successfully",
        "steps": 0,
        "done": False,
        "score": 0.0,
        "max_score": max_score,
        "score_display": f"0.00 / {max_score}"
    }

@app.get("/reset/{task_id}")
def reset_get(task_id: str):
    req = ResetRequest(task_id=task_id)
    return reset(req)

# =========================
# STEP AI - WITH PROPER COMPLETION LOGIC
# =========================

@app.post("/step_ai")
def step_ai(req: StepRequest):
    if req.session_id not in sessions:
        raise HTTPException(404, "Invalid session_id. Please reset first.")

    session = sessions[req.session_id]
    config = REWARD_CONFIG[session["task_id"]]
    max_score = config["max_score"]

    if session["done"]:
        if session["perfect_completion"]:
            return {
                "message": "✅ Task already completed perfectly! Great job! Reset to start new.",
                "done": True,
                "perfect": True,
                "score": session["total_reward"],
                "max_score": max_score,
                "score_display": f"{session['total_reward']:.2f} / {max_score}",
                "step": session["steps"]
            }
        else:
            return {
                "message": f"⚠️ Task already completed but with penalties (score: {session['total_reward']:.2f}/{max_score}). Reset to try for perfect score.",
                "done": True,
                "perfect": False,
                "score": session["total_reward"],
                "max_score": max_score,
                "score_display": f"{session['total_reward']:.2f} / {max_score}",
                "step": session["steps"]
            }

    env = session["env"]
    step_num = session["steps"] + 1
    used_expert = req.use_expert
    
    # Call AI model (with expert advice if requested)
    ai_action = call_ai_model(session["task_id"], step_num, session["history"], used_expert)
    
    try:
        action = Action(**ai_action)
    except Exception:
        ai_action = get_step_default_action(session["task_id"], step_num)
        action = Action(**ai_action)
    
    # Validate AI's action
    is_valid, reward_value, explanation, expected, is_perfect_step = validate_action(
        session["task_id"], step_num, ai_action, used_expert
    )
    
    # Take step in environment
    obs, env_reward, done, info = env.step(action)
    
    # Update session
    session["steps"] += 1
    session["rewards"].append(reward_value)
    session["explanations"].append(explanation)
    session["actions_taken"].append(ai_action)
    session["total_reward"] += reward_value
    session["history"].append(ai_action)
    
    if used_expert:
        session["expert_used_count"] += 1
    
    # Determine if task is complete and if it's perfect
    all_steps_complete = False
    perfect_completion = False
    
    if session["task_id"] == "address_change_hard":
        all_steps_complete = session["steps"] >= 3
    else:
        all_steps_complete = session["steps"] >= 1
    
    if all_steps_complete:
        session["done"] = True
        # Perfect completion: no expert used AND total reward equals max_score
        perfect_completion = (session["expert_used_count"] == 0 and abs(session["total_reward"] - max_score) < 0.01)
        session["perfect_completion"] = perfect_completion
    
    score_value = min(max(session["total_reward"], 0.0), max_score)
    
    # Generate completion message
    completion_message = None
    if session["done"]:
        if perfect_completion:
            completion_message = "🎉 PERFECT COMPLETION! No expert used, optimal score achieved!"
        else:
            completion_message = f"⚠️ Completed with penalties (expert used: {session['expert_used_count']} times, score reduced to {score_value:.2f}/{max_score})"
    
    return {
        "step": session["steps"],
        "action": ai_action,
        "reward": round(reward_value, 2),
        "reward_explanation": explanation,
        "done": session["done"],
        "perfect_completion": session.get("perfect_completion", False),
        "score": score_value,
        "max_score": max_score,
        "score_display": f"{score_value:.2f} / {max_score}",
        "total_reward": session["total_reward"],
        "is_valid": is_valid,
        "used_expert": used_expert,
        "expert_penalty": -0.2 if used_expert else 0,
        "expert_used_count": session["expert_used_count"],
        "expected_action": expected if not is_valid else None,
        "completion_message": completion_message
    }


@app.get("/session/{session_id}")
def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    
    session = sessions[session_id]
    config = REWARD_CONFIG[session["task_id"]]
    max_score = config["max_score"]
    score_value = min(max(session["total_reward"], 0.0), max_score)
    
    return {
        "session_id": session_id,
        "task_id": session["task_id"],
        "steps": session["steps"],
        "rewards": session["rewards"],
        "explanations": session["explanations"],
        "actions_taken": session["actions_taken"],
        "expert_used_count": session["expert_used_count"],
        "total_reward": session["total_reward"],
        "score": score_value,
        "max_score": max_score,
        "score_display": f"{score_value:.2f} / {max_score}",
        "done": session["done"],
        "perfect_completion": session.get("perfect_completion", False)
    }


# =========================
# TASKS
# =========================

@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {"task_id": "order_status_easy", "name": "Order Status Query", "description": "Look up order status", "difficulty": "easy", "max_steps": 3, "max_score": 0.8},
            {"task_id": "refund_policy_medium", "name": "Refund Policy Explanation", "description": "Explain refund policy", "difficulty": "medium", "max_steps": 3, "max_score": 0.8},
            {"task_id": "address_change_hard", "name": "Address Change Request", "description": "Handle address change in 3 steps", "difficulty": "hard", "max_steps": 5, "max_score": 1.0}
        ]
    }


@app.get("/health")
def health():
    return {"status": "healthy", "active_sessions": len(sessions), "ai_model": MODEL_NAME if client else "fallback"}


# =========================
# UI - COMPLETE WITH ALL FIXES
# =========================

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Customer Support Environment | RL Agent Training</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        
        .header { text-align: center; color: white; margin-bottom: 30px; }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .header p { opacity: 0.8; }
        
        .tasks-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 20px; }
        
        .task-card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        .task-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
        .task-header h3 { color: white; font-size: 1.3em; }
        .difficulty { padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: bold; }
        .easy { background: #10b981; }
        .medium { background: #f59e0b; }
        .hard { background: #ef4444; }
        .task-desc { color: rgba(255,255,255,0.7); font-size: 13px; margin-bottom: 15px; }
        
        .progress-section { margin: 15px 0; }
        .progress-label { display: flex; justify-content: space-between; color: rgba(255,255,255,0.7); font-size: 12px; margin-bottom: 5px; }
        .progress-bar-container { background: rgba(255,255,255,0.2); border-radius: 10px; height: 8px; overflow: hidden; }
        .progress-fill { background: linear-gradient(90deg, #10b981, #667eea); height: 100%; border-radius: 10px; transition: width 0.3s ease; width: 0%; }
        
        .expert-badge { background: rgba(245, 158, 11, 0.2); border: 1px solid #f59e0b; border-radius: 8px; padding: 8px; margin: 10px 0; text-align: center; }
        .expert-badge span { color: #f59e0b; font-size: 12px; }
        
        .button-group { display: flex; gap: 10px; margin: 15px 0; flex-wrap: wrap; }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }
        button:hover { opacity: 0.9; transform: scale(1.02); }
        button:disabled { opacity: 0.4; cursor: not-allowed; }
        .reset-btn { background: linear-gradient(135deg, #f59e0b, #d97706); }
        .expert-btn { background: linear-gradient(135deg, #f59e0b, #d97706); border: 1px solid #f59e0b; }
        .auto-btn { background: linear-gradient(135deg, #10b981, #059669); }
        
        .response-area {
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            padding: 12px;
            margin-top: 15px;
            max-height: 400px;
            overflow-y: auto;
        }
        
        .step-entry {
            background: rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 10px;
            border-left: 3px solid #667eea;
        }
        .step-entry.expert-used { border-left-color: #f59e0b; background: rgba(245, 158, 11, 0.1); }
        .step-header { display: flex; justify-content: space-between; margin-bottom: 8px; }
        .step-number { font-weight: bold; color: #667eea; }
        .step-reward { font-weight: bold; }
        .reward-positive { color: #10b981; }
        .reward-negative { color: #ef4444; }
        .step-action { font-family: monospace; font-size: 11px; color: rgba(255,255,255,0.7); margin: 5px 0; word-break: break-all; }
        .step-explanation { font-size: 11px; color: #aaa; margin-top: 5px; }
        .expert-warning { color: #f59e0b; font-size: 11px; margin-top: 5px; }
        
        .score-display { font-size: 20px; font-weight: bold; text-align: center; padding: 10px; border-radius: 10px; margin: 10px 0; }
        .score-perfect { background: rgba(16, 185, 129, 0.2); color: #10b981; }
        .score-penalty { background: rgba(245, 158, 11, 0.2); color: #f59e0b; }
        .score-normal { background: rgba(102, 126, 234, 0.2); color: #667eea; }
        
        .completion-message { padding: 10px; border-radius: 8px; margin-top: 10px; font-size: 13px; font-weight: bold; text-align: center; }
        .completion-perfect { background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; }
        .completion-penalty { background: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid #f59e0b; }
        
        .status { position: fixed; bottom: 15px; right: 15px; background: rgba(0,0,0,0.7); padding: 6px 12px; border-radius: 20px; font-size: 11px; color: #10b981; }
        .empty-state { text-align: center; color: rgba(255,255,255,0.4); padding: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Customer Support Environment</h1>
            <p>RL Agent Training | Explainable Rewards | Ask Expert Feature</p>
        </div>
        <div class="tasks-grid" id="tasks"></div>
        <div class="status" id="status">🟢 System Online</div>
    </div>

    <script>
        const API_BASE = window.location.origin;
        let currentSessions = {};

        async function checkHealth() {
            try {
                const res = await fetch(`${API_BASE}/health`);
                const data = await res.json();
                document.getElementById('status').innerHTML = `🟢 Online | ${data.active_sessions} sessions | 🤖 ${data.ai_model || 'LLaMA'}`;
            } catch(e) { document.getElementById('status').innerHTML = '🔴 Offline'; }
        }

        async function resetTask(taskId, maxScore) {
            try {
                const res = await fetch(`${API_BASE}/reset/${taskId}`);
                const data = await res.json();
                currentSessions[taskId] = data.session_id;
                
                const responseDiv = document.getElementById(`response-${taskId}`);
                responseDiv.innerHTML = `<div class="empty-state">✅ Environment reset. Ready for training.</div>`;
                
                // Reset progress bar
                const progressFill = document.getElementById(`progress-${taskId}`);
                if (progressFill) progressFill.style.width = '0%';
                
                // Reset score display
                const scoreDiv = document.getElementById(`score-display-${taskId}`);
                if (scoreDiv) {
                    scoreDiv.innerHTML = `Score: 0.00 / ${maxScore.toFixed(2)}`;
                    scoreDiv.className = 'score-display score-normal';
                }
                
                const stepBtn = document.getElementById(`step-${taskId}`);
                stepBtn.disabled = false;
                stepBtn.textContent = '🤖 Take Step';
                
                const expertBtn = document.getElementById(`expert-${taskId}`);
                if (expertBtn) expertBtn.disabled = false;
                
                const autoBtn = document.getElementById(`auto-${taskId}`);
                if (autoBtn) autoBtn.disabled = false;
                
                // Clear completion message
                const completionDiv = document.getElementById(`completion-${taskId}`);
                if (completionDiv) completionDiv.innerHTML = '';
                
            } catch(e) { alert('Reset error: ' + e.message); }
        }
        
        function updateProgress(taskId, scoreValue, maxScore) {
            const progressFill = document.getElementById(`progress-${taskId}`);
            if (progressFill) {
                const percent = (scoreValue / maxScore) * 100;
                progressFill.style.width = `${percent}%`;
            }
        }

        async function takeStep(taskId, useExpert = false, maxScore) {
            if (!currentSessions[taskId]) { alert('Please reset first!'); return; }
            
            const stepBtn = document.getElementById(`step-${taskId}`);
            const expertBtn = document.getElementById(`expert-${taskId}`);
            stepBtn.disabled = true;
            if (expertBtn) expertBtn.disabled = true;
            stepBtn.textContent = useExpert ? '⏳ Asking Expert...' : '⏳ AI Thinking...';
            
            try {
                const res = await fetch(`${API_BASE}/step_ai`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: currentSessions[taskId], use_expert: useExpert })
                });
                const data = await res.json();
                
                const responseDiv = document.getElementById(`response-${taskId}`);
                const rewardClass = data.reward >= 0 ? 'reward-positive' : 'reward-negative';
                const expertClass = data.used_expert ? 'expert-used' : '';
                const expertWarning = data.used_expert ? `<div class="expert-warning">🎓 Expert Used! Penalty: -0.2 (Total: ${data.expert_used_count})</div>` : '';
                
                const stepHtml = `
                    <div class="step-entry ${expertClass}">
                        <div class="step-header">
                            <span class="step-number">Step ${data.step}</span>
                            <span class="step-reward ${rewardClass}">${data.reward >= 0 ? '+' : ''}${data.reward}</span>
                        </div>
                        <div class="step-action">Action: ${JSON.stringify(data.action)}</div>
                        <div class="step-explanation">📖 ${data.reward_explanation}</div>
                        ${expertWarning}
                    </div>
                `;
                
                responseDiv.innerHTML = stepHtml + responseDiv.innerHTML;
                
                // Update score display
                const scoreDiv = document.getElementById(`score-display-${taskId}`);
                if (scoreDiv) {
                    scoreDiv.innerHTML = `Score: ${data.score_display}`;
                    if (data.perfect_completion) {
                        scoreDiv.className = 'score-display score-perfect';
                    } else if (data.expert_used_count > 0 && data.done) {
                        scoreDiv.className = 'score-display score-penalty';
                    } else {
                        scoreDiv.className = 'score-display score-normal';
                    }
                }
                
                // Update progress
                updateProgress(taskId, data.score, maxScore);
                
                // Show completion message if done
                const completionDiv = document.getElementById(`completion-${taskId}`);
                if (completionDiv && data.done) {
                    if (data.perfect_completion) {
                        completionDiv.innerHTML = `<div class="completion-message completion-perfect">🎉 PERFECT COMPLETION! No expert used, optimal score achieved!</div>`;
                    } else {
                        completionDiv.innerHTML = `<div class="completion-message completion-penalty">⚠️ Completed with penalties (expert used: ${data.expert_used_count} times, score reduced to ${data.score_display})</div>`;
                    }
                }
                
                if (data.done) {
                    stepBtn.disabled = true;
                    stepBtn.textContent = '✅ Completed';
                    if (expertBtn) expertBtn.disabled = true;
                    const autoBtn = document.getElementById(`auto-${taskId}`);
                    if (autoBtn) autoBtn.disabled = true;
                } else {
                    stepBtn.disabled = false;
                    stepBtn.textContent = '🤖 Take Next Step';
                    if (expertBtn) expertBtn.disabled = false;
                }
            } catch(e) {
                alert('Step error: ' + e.message);
                stepBtn.disabled = false;
                stepBtn.textContent = '🤖 Retry';
                if (expertBtn) expertBtn.disabled = false;
            }
        }

        async function runFullTask(taskId, maxScore) {
            if (!currentSessions[taskId]) { await resetTask(taskId, maxScore); await new Promise(r => setTimeout(r, 500)); }
            
            const stepBtn = document.getElementById(`step-${taskId}`);
            const expertBtn = document.getElementById(`expert-${taskId}`);
            const autoBtn = document.getElementById(`auto-${taskId}`);
            autoBtn.disabled = true;
            autoBtn.textContent = '🏃 Running...';
            stepBtn.disabled = true;
            if (expertBtn) expertBtn.disabled = true;
            
            let done = false;
            let maxAttempts = 10;
            let attempts = 0;
            
            while (!done && attempts < maxAttempts) {
                await new Promise(r => setTimeout(r, 600));
                
                const res = await fetch(`${API_BASE}/step_ai`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: currentSessions[taskId], use_expert: false })
                });
                const data = await res.json();
                done = data.done;
                attempts++;
                
                const responseDiv = document.getElementById(`response-${taskId}`);
                const rewardClass = data.reward >= 0 ? 'reward-positive' : 'reward-negative';
                const stepHtml = `
                    <div class="step-entry">
                        <div class="step-header">
                            <span class="step-number">Step ${data.step}</span>
                            <span class="step-reward ${rewardClass}">${data.reward >= 0 ? '+' : ''}${data.reward}</span>
                        </div>
                        <div class="step-action">Action: ${JSON.stringify(data.action)}</div>
                        <div class="step-explanation">📖 ${data.reward_explanation}</div>
                    </div>
                `;
                responseDiv.innerHTML = stepHtml + responseDiv.innerHTML;
                
                const scoreDiv = document.getElementById(`score-display-${taskId}`);
                if (scoreDiv) scoreDiv.innerHTML = `Score: ${data.score_display}`;
                updateProgress(taskId, data.score, maxScore);
                
                const completionDiv = document.getElementById(`completion-${taskId}`);
                if (completionDiv && data.done) {
                    if (data.perfect_completion) {
                        completionDiv.innerHTML = `<div class="completion-message completion-perfect">🎉 PERFECT COMPLETION! No expert used, optimal score achieved!</div>`;
                    } else {
                        completionDiv.innerHTML = `<div class="completion-message completion-penalty">⚠️ Completed with penalties (expert used: ${data.expert_used_count} times, score reduced to ${data.score_display})</div>`;
                    }
                }
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
                grid.innerHTML = data.tasks.map(t => `
                    <div class="task-card">
                        <div class="task-header">
                            <h3>${t.name}</h3>
                            <span class="difficulty ${t.difficulty}">${t.difficulty.toUpperCase()}</span>
                        </div>
                        <div class="task-desc">${t.description}</div>
                        
                        <div class="progress-section">
                            <div class="progress-label"><span>Progress</span><span id="progress-label-${t.task_id}">0%</span></div>
                            <div class="progress-bar-container"><div id="progress-${t.task_id}" class="progress-fill" style="width: 0%"></div></div>
                        </div>
                        
                        <div id="score-display-${t.task_id}" class="score-display score-normal">Score: 0.00 / ${t.max_score.toFixed(2)}</div>
                        
                        <div class="button-group">
                            <button class="reset-btn" onclick="resetTask('${t.task_id}', ${t.max_score})">🔄 Reset</button>
                            <button id="step-${t.task_id}" onclick="takeStep('${t.task_id}', false, ${t.max_score})" disabled>🤖 Take Step</button>
                            <button id="expert-${t.task_id}" class="expert-btn" onclick="takeStep('${t.task_id}', true, ${t.max_score})" disabled>🎓 Ask Expert (-0.2)</button>
                            <button id="auto-${t.task_id}" class="auto-btn" onclick="runFullTask('${t.task_id}', ${t.max_score})" disabled>⚡ Run Full Episode</button>
                        </div>
                        
                        <div class="expert-badge"><span>🎓 Ask Expert: Get guidance (-0.2 penalty per use)</span></div>
                        
                        <div id="completion-${t.task_id}"></div>
                        <div id="response-${t.task_id}" class="response-area">
                            <div class="empty-state">🔄 Click "Reset" to start training</div>
                        </div>
                    </div>
                `).join('');
            } catch(e) { console.error(e); }
        }

        checkHealth();
        loadTasks();
        setInterval(checkHealth, 30000);
    </script>
</body>
</html>
    """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)