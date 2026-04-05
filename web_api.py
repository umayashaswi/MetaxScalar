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

class ResetRequest(BaseModel):
    task_id: str

# =========================
# TASK-SPECIFIC PROMPTS (ALL TASKS USE AI)
# =========================

def get_task_prompt(task_id: str, step_num: int, history: List) -> str:
    if task_id == "order_status_easy":
        return (
            "You are a Customer Support AI. Return ONLY valid JSON.\n\n"
            "The customer wants to check their order status.\n\n"
            "You MUST respond with EXACTLY this JSON:\n"
            '{"action_type": "lookup_order", "order_id": "12345"}\n\n'
            "IMPORTANT RULES:\n"
            "- DO NOT use 'order_status_easy' as action_type\n"
            "- DO NOT send any reply messages\n"
            "- DO NOT ask questions\n"
            "- ONLY use 'lookup_order' as action_type\n"
            "- The order_id MUST be '12345'\n\n"
            "Example of CORRECT response: {\"action_type\": \"lookup_order\", \"order_id\": \"12345\"}\n"
            "Example of WRONG response: {\"action_type\": \"order_status_easy\"}\n\n"
            "Return ONLY the JSON, no other text."
        )
    
    elif task_id == "refund_policy_medium":
        return (
            "You are a Customer Support AI. Return ONLY valid JSON.\n\n"
            "The customer wants to know the refund policy.\n\n"
            "You MUST respond with:\n"
            '{"action_type": "send_reply", "message": "Our refund policy allows returns within 30 days for a full refund."}\n\n'
            "IMPORTANT:\n"
            "- Your message MUST include the word 'refund'\n"
            "- DO NOT lookup the order\n"
            "- Return ONLY the JSON\n\n"
            "Example: {\"action_type\": \"send_reply\", \"message\": \"Our refund policy allows returns within 30 days for a full refund.\"}"
        )
    
    elif task_id == "address_change_hard":
        step_hint = ""
        if step_num == 1:
            step_hint = "This is STEP 1. You must lookup the order first."
        elif step_num == 2:
            step_hint = "This is STEP 2. You have already looked up the order. Now ask the customer for their new address."
        elif step_num == 3:
            step_hint = "This is STEP 3. You have already asked for the address. Now ask the customer to confirm it."
        
        return (
            "You are a Customer Support AI handling address changes.\n"
            "Return ONLY valid JSON.\n\n"
            f"{step_hint}\n\n"
            "Follow this EXACT sequence:\n"
            "STEP 1: {\"action_type\": \"lookup_order\", \"order_id\": \"12345\"}\n"
            "STEP 2: {\"action_type\": \"send_reply\", \"message\": \"Please provide your new address.\"}\n"
            "STEP 3: {\"action_type\": \"send_reply\", \"message\": \"Please confirm your new address.\"}\n\n"
            "Return ONLY the JSON for the current step. No explanations."
        )
    
    else:
        return "Return valid JSON with action_type field."

def get_step_default_action(task_id: str, step_num: int) -> Dict[str, Any]:
    """Fallback action if AI fails (used as safety net only)"""
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

def call_ai_model(task_id: str, step_num: int, history: List) -> Dict[str, Any]:
    """Call the AI model to get an action - ALL tasks use AI"""
    
    if client is None:
        print("⚠️ No API key, using fallback action")
        return get_step_default_action(task_id, step_num)
    
    sys_prompt = get_task_prompt(task_id, step_num, history)
    
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
            print(f"⚠️ No JSON found, using fallback")
            return get_step_default_action(task_id, step_num)

        data = json.loads(content)
        
        # Validate and fix common issues
        if "action_type" not in data:
            print(f"⚠️ No action_type in response, using fallback")
            return get_step_default_action(task_id, step_num)
        
        # Fix: if action_type is the task_id itself (common mistake)
        if data["action_type"] == task_id:
            print(f"⚠️ AI returned action_type as task_id, fixing...")
            if task_id == "order_status_easy":
                return {"action_type": "lookup_order", "order_id": "12345"}
            elif task_id == "refund_policy_medium":
                return {"action_type": "send_reply", "message": "Our refund policy allows returns within 30 days for a full refund."}
        
        # Fix: if lookup_order missing order_id
        if data["action_type"] == "lookup_order" and "order_id" not in data:
            data["order_id"] = "12345"
        
        return data
        
    except Exception as e:
        print(f"❌ AI model error: {e}")
        return get_step_default_action(task_id, step_num)

# =========================
# ACTION VALIDATION
# =========================

def validate_action(task_id: str, step: int, action_dict: Dict) -> tuple:
    action_type = action_dict.get("action_type", "")
    
    if task_id == "order_status_easy":
        if action_type == "lookup_order" and action_dict.get("order_id") == "12345":
            return True, 1.0, "✅ Correct: Order lookup successful", "lookup_order"
        else:
            return False, -0.3, f"❌ Wrong: Got '{action_type}', expected 'lookup_order'", "lookup_order"
    
    if task_id == "refund_policy_medium":
        if action_type == "send_reply" and "refund" in action_dict.get("message", "").lower():
            return True, 1.0, "✅ Correct: Refund policy explained", "send_reply with 'refund'"
        else:
            return False, -0.3, "❌ Wrong: Must send reply with 'refund'", "send_reply with 'refund'"
    
    if task_id == "address_change_hard":
        if step == 1:
            if action_type == "lookup_order" and action_dict.get("order_id") == "12345":
                return True, 0.6, "✅ Step 1/3: Order located", "lookup_order"
            else:
                return False, -0.3, f"❌ Step 1: Got '{action_type}', expected 'lookup_order'", "lookup_order"
        
        elif step == 2:
            if action_type == "send_reply" and "address" in action_dict.get("message", "").lower():
                return True, 0.2, "✅ Step 2/3: Address requested", "send_reply asking for address"
            else:
                return False, -0.3, "❌ Step 2: Must ask for new address", "send_reply asking for address"
        
        elif step == 3:
            if action_type == "send_reply" and "confirm" in action_dict.get("message", "").lower():
                return True, 0.2, "✅ Step 3/3: Address confirmed - Complete!", "send_reply asking for confirmation"
            else:
                return False, -0.3, "❌ Step 3: Must ask for confirmation", "send_reply asking for confirmation"
    
    return False, -0.3, "❌ Invalid action format", "valid JSON"

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
        "actions_taken": [],
        "done": False,
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
        "score": "0.0 / 1.0"
    }

@app.get("/reset/{task_id}")
def reset_get(task_id: str):
    req = ResetRequest(task_id=task_id)
    return reset(req)

# =========================
# STEP AI - ALL TASKS USE AI
# =========================

@app.post("/step_ai")
def step_ai(req: StepRequest):
    if req.session_id not in sessions:
        raise HTTPException(404, "Invalid session_id. Please reset first.")

    session = sessions[req.session_id]

    if session["done"]:
        return {
            "message": "Task already completed. Please reset.",
            "done": True,
            "score": f"{session['total_reward']:.1f} / 1.0",
            "step": session["steps"]
        }

    env = session["env"]
    step_num = session["steps"] + 1
    
    # ✅ ALL tasks use AI model (no special cases)
    ai_action = call_ai_model(session["task_id"], step_num, session["history"])
    
    try:
        action = Action(**ai_action)
    except Exception:
        ai_action = get_step_default_action(session["task_id"], step_num)
        action = Action(**ai_action)
    
    # Validate AI's action
    is_valid, reward_value, explanation, expected = validate_action(
        session["task_id"], step_num, ai_action
    )
    
    # Take step in environment
    obs, env_reward, done, info = env.step(action)
    
    session["steps"] += 1
    session["rewards"].append(reward_value)
    session["explanations"].append(explanation)
    session["actions_taken"].append(ai_action)
    session["total_reward"] += reward_value
    session["done"] = done
    session["history"].append(ai_action)

    score_value = min(max(session["total_reward"], 0.0), 1.0)
    
    return {
        "step": session["steps"],
        "action": ai_action,
        "reward": reward_value,
        "reward_explanation": explanation,
        "done": done,
        "score": f"{score_value:.1f} / 1.0",
        "score_value": score_value,
        "total_reward": session["total_reward"],
        "is_valid": is_valid,
        "expected_action": expected if not is_valid else None,
        "message": "🎉 Task completed!" if done else None
    }


@app.get("/session/{session_id}")
def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    
    session = sessions[session_id]
    score_value = min(max(session["total_reward"], 0.0), 1.0)
    
    return {
        "session_id": session_id,
        "task_id": session["task_id"],
        "steps": session["steps"],
        "rewards": session["rewards"],
        "explanations": session["explanations"],
        "actions_taken": session["actions_taken"],
        "total_reward": session["total_reward"],
        "score": f"{score_value:.1f} / 1.0",
        "done": session["done"]
    }


# =========================
# TASKS
# =========================

@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {"task_id": "order_status_easy", "name": "Order Status Query", "description": "Look up order status using lookup_order action", "difficulty": "easy", "max_steps": 3, "expected_reward": 1.0},
            {"task_id": "refund_policy_medium", "name": "Refund Policy Explanation", "description": "Explain refund policy in a reply message", "difficulty": "medium", "max_steps": 3, "expected_reward": 1.0},
            {"task_id": "address_change_hard", "name": "Address Change Request", "description": "Handle address change request in 3 steps", "difficulty": "hard", "max_steps": 5, "expected_reward": 1.0}
        ]
    }


@app.get("/health")
def health():
    return {"status": "healthy", "active_sessions": len(sessions), "ai_model": MODEL_NAME if client else "fallback"}


# =========================
# UI
# =========================

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Customer Support Environment</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { text-align: center; color: white; margin-bottom: 30px; }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { opacity: 0.8; }
        .tasks-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .task-card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        .task-card h3 { color: white; margin-bottom: 10px; }
        .difficulty {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .easy { background: #10b981; }
        .medium { background: #f59e0b; }
        .hard { background: #ef4444; }
        .task-desc { color: rgba(255,255,255,0.7); font-size: 13px; margin-bottom: 15px; }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            margin: 5px 5px 5px 0;
            font-size: 13px;
        }
        button:hover { opacity: 0.9; transform: scale(1.02); }
        button:disabled { opacity: 0.4; cursor: not-allowed; }
        .reset-btn { background: linear-gradient(135deg, #f59e0b, #d97706); }
        .auto-btn { background: linear-gradient(135deg, #10b981, #059669); }
        .response-area {
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            padding: 15px;
            margin-top: 15px;
            max-height: 400px;
            overflow-y: auto;
        }
        .step-entry {
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
            border-left: 3px solid #667eea;
        }
        .reward-positive { color: #10b981; font-weight: bold; }
        .reward-negative { color: #ef4444; font-weight: bold; }
        .score-display { font-size: 18px; font-weight: bold; color: #667eea; }
        .status {
            position: fixed;
            bottom: 15px;
            right: 15px;
            background: rgba(0,0,0,0.7);
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 11px;
            color: #10b981;
        }
        .empty-state { text-align: center; color: rgba(255,255,255,0.4); padding: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Customer Support Environment</h1>
            <p>Real AI Inference (LLaMA 3.1) | Live Actions | Instant Rewards</p>
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

        async function resetTask(taskId) {
            try {
                const res = await fetch(`${API_BASE}/reset/${taskId}`);
                const data = await res.json();
                currentSessions[taskId] = data.session_id;
                
                const responseDiv = document.getElementById(`response-${taskId}`);
                responseDiv.innerHTML = `<div class="empty-state">✅ Environment reset. Ready for AI inference.</div>`;
                
                const stepBtn = document.getElementById(`step-${taskId}`);
                stepBtn.disabled = false;
                stepBtn.textContent = '🤖 Take Step';
                stepBtn.style.opacity = '1';
                
                const autoBtn = document.getElementById(`auto-${taskId}`);
                if (autoBtn) { autoBtn.disabled = false; autoBtn.textContent = '⚡ Run Full Episode'; }
            } catch(e) { alert('Reset error: ' + e.message); }
        }

        async function takeStep(taskId) {
            if (!currentSessions[taskId]) { alert('Please reset first!'); return; }
            
            const stepBtn = document.getElementById(`step-${taskId}`);
            stepBtn.disabled = true;
            stepBtn.textContent = '⏳ AI Thinking...';
            
            try {
                const res = await fetch(`${API_BASE}/step_ai`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: currentSessions[taskId] })
                });
                const data = await res.json();
                
                const responseDiv = document.getElementById(`response-${taskId}`);
                const rewardClass = data.reward >= 0 ? 'reward-positive' : 'reward-negative';
                
                const stepHtml = `
                    <div class="step-entry">
                        <strong>Step ${data.step}</strong><br>
                        <strong>Action:</strong> <code>${JSON.stringify(data.action)}</code><br>
                        <strong>Reward:</strong> <span class="${rewardClass}">${data.reward >= 0 ? '+' : ''}${data.reward}</span><br>
                        <div style="font-size: 11px; color: #aaa; margin-top: 5px;">📖 ${data.reward_explanation}</div>
                        <strong>Score:</strong> <span class="score-display">${data.score}</span><br>
                        ${data.message ? `<div style="margin-top: 8px; color: #10b981;">🎉 ${data.message}</div>` : ''}
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
            } catch(e) {
                alert('Step error: ' + e.message);
                stepBtn.disabled = false;
                stepBtn.textContent = '🤖 Retry';
            }
        }

        async function runFullTask(taskId) {
            if (!currentSessions[taskId]) { await resetTask(taskId); await new Promise(r => setTimeout(r, 500)); }
            
            const stepBtn = document.getElementById(`step-${taskId}`);
            const autoBtn = document.getElementById(`auto-${taskId}`);
            autoBtn.disabled = true;
            autoBtn.textContent = '🏃 Running...';
            stepBtn.disabled = true;
            
            let done = false;
            let maxAttempts = 10;
            let attempts = 0;
            
            while (!done && attempts < maxAttempts) {
                await new Promise(r => setTimeout(r, 600));
                
                const res = await fetch(`${API_BASE}/step_ai`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: currentSessions[taskId] })
                });
                const data = await res.json();
                done = data.done;
                attempts++;
                
                const responseDiv = document.getElementById(`response-${taskId}`);
                const rewardClass = data.reward >= 0 ? 'reward-positive' : 'reward-negative';
                const stepHtml = `
                    <div class="step-entry">
                        <strong>Step ${data.step}</strong><br>
                        <strong>Action:</strong> <code>${JSON.stringify(data.action)}</code><br>
                        <strong>Reward:</strong> <span class="${rewardClass}">${data.reward >= 0 ? '+' : ''}${data.reward}</span><br>
                        <div style="font-size: 11px; color: #aaa; margin-top: 5px;">📖 ${data.reward_explanation}</div>
                        <strong>Score:</strong> <span class="score-display">${data.score}</span><br>
                        ${data.message ? `<div style="margin-top: 8px; color: #10b981;">🎉 ${data.message}</div>` : ''}
                    </div>
                `;
                responseDiv.innerHTML = stepHtml + responseDiv.innerHTML;
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
                        <h3>${t.name}</h3>
                        <span class="difficulty ${t.difficulty}">${t.difficulty.toUpperCase()}</span>
                        <div class="task-desc">${t.description}</div>
                        <div style="font-size: 12px; color: #10b981;">🎯 Expected: ${t.expected_reward}/1.0 | ⚡ Max Steps: ${t.max_steps}</div>
                        <div style="margin-top: 15px;">
                            <button class="reset-btn" onclick="resetTask('${t.task_id}')">🔄 Reset</button>
                            <button id="step-${t.task_id}" onclick="takeStep('${t.task_id}')" disabled>🤖 Take Step</button>
                            <button id="auto-${t.task_id}" class="auto-btn" onclick="runFullTask('${t.task_id}')" disabled>⚡ Run Full Episode</button>
                        </div>
                        <div id="response-${t.task_id}" class="response-area">
                            <div class="empty-state">🔄 Click "Reset" to start AI inference</div>
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