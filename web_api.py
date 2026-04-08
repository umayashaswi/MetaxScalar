import json
import os
import uuid
import random
from typing import Dict, Any, List, Optional
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
openenv_session: Optional[CustomerSupportEnv] = None

# =========================
# REQUEST MODELS
# =========================

class StepRequest(BaseModel):
    session_id: str
    use_expert: bool = False

class ResetRequest(BaseModel):
    task_id: str

# =========================
# MAX STEPS CONFIGURATION
# =========================

MAX_STEPS = {
    "order_status_easy": 5,
    "refund_policy_medium": 5,
    "address_change_hard": 8,
    "ambiguous_request": 10
}

# =========================
# AMBIGUOUS QUERIES (Randomized)
# =========================

AMBIGUOUS_QUERIES = [
    "I moved and didn't get my package",
    "My order didn't arrive after I changed address",
    "Package missing after relocation",
    "I think it was delivered to my old address",
    "I changed my address but my order went to the old one",
    "Moved houses last month, order still hasn't arrived"
]

# =========================
# REWARD CONFIGURATION
# =========================

REWARD_CONFIG = {
    "order_status_easy": {
        "max_score": 1.0,
        "expert_hint": "Use exactly: {'action_type': 'lookup_order', 'order_id': '12345'}"
    },
    "refund_policy_medium": {
        "max_score": 1.0,
        "expert_hint": "Use 'send_reply' and explain that we offer a 30-day full refund policy. Include the word 'refund'."
    },
    "address_change_hard": {
        "max_score": 1.0,
        "expert_hint": [
            "Step 1: Use {'action_type': 'lookup_order', 'order_id': '12345'}. DO NOT reply yet.",
            "Step 2: Order is FOUND. DO NOT use lookup_order again. Use {'action_type': 'send_reply', 'message': 'Please provide your new address.'}",
            "Step 3: Address received. Use {'action_type': 'send_reply', 'message': 'Please confirm this address.'}"
        ]
    },
    "ambiguous_request": {
        "max_score": 1.0,
        "expert_hint": [
            "Step 1: You must check the package location first. Use {'action_type': 'lookup_order', 'order_id': '12345'}.",
            "Step 2: Order shows 'Returned to Sender'. Use {'action_type': 'send_reply', 'message': 'I see you moved. Can you confirm your new address?'}",
            "Step 3: Address confirmed. Resolve the issue: {'action_type': 'send_reply', 'message': 'I have updated your address and a replacement is on the way.'}"
        ]
    }
}

# =========================
# OPENENV COMPATIBILITY ENDPOINTS (FIXED - NO BODY REQUIRED)
# =========================

@app.post("/openenv/reset")
def openenv_reset():
    """OpenEnv compatible reset endpoint - NO body required"""
    global openenv_session
    openenv_session = CustomerSupportEnv(task_id="order_status_easy")
    obs = openenv_session.reset()
    
    return {
        "task_id": obs.task_id,
        "history": obs.history,
        "done": obs.done,
        "observation_text": None
    }

@app.post("/openenv/step")
def openenv_step(action: Action):
    """OpenEnv compatible step endpoint"""
    global openenv_session
    
    if openenv_session is None:
        raise HTTPException(400, "Environment not initialized. Call /openenv/reset first.")
    
    obs, reward, done, info = openenv_session.step(action)
    
    return {
        "observation": {
            "task_id": obs.task_id,
            "history": obs.history,
            "done": obs.done,
            "observation_text": None
        },
        "reward": reward.value,
        "done": done,
        "info": info
    }

@app.get("/openenv/validate")
def openenv_validate():
    """OpenEnv validation endpoint"""
    return {"status": "ok", "ready": True}

# =========================
# TASKS ENDPOINT
# =========================

@app.get("/openenv/tasks")
def openenv_tasks():
    """List available tasks"""
    return {
        "tasks": [
            {
                "task_id": "order_status_easy",
                "name": "Order Status Query",
                "description": "Customer wants to check their order status",
                "difficulty": "easy",
                "max_steps": 5,
                "max_score": 1.0
            },
            {
                "task_id": "refund_policy_medium",
                "name": "Refund Policy Explanation",
                "description": "Customer wants to know the refund policy",
                "difficulty": "medium",
                "max_steps": 5,
                "max_score": 1.0
            },
            {
                "task_id": "address_change_hard",
                "name": "Address Change Request",
                "description": "Customer wants to change their shipping address",
                "difficulty": "hard",
                "max_steps": 8,
                "max_score": 1.0
            },
            {
                "task_id": "ambiguous_request",
                "name": "Moved & Missing Package",
                "description": "Customer issue with moved address and missing package",
                "difficulty": "hard+",
                "max_steps": 10,
                "max_score": 1.0
            }
        ]
    }

# =========================
# STATE ENDPOINT
# =========================

@app.get("/openenv/state")
def openenv_state():
    """Get current environment state"""
    global openenv_session
    if openenv_session is None:
        raise HTTPException(400, "Environment not initialized. Call /openenv/reset first.")
    
    return {
        "task_id": openenv_session.task_id,
        "history": openenv_session.history,
        "step_count": openenv_session.step_count,
        "total_reward": openenv_session.total_reward,
        "done": openenv_session.done
    }

# =========================
# LEGACY ENDPOINTS (for web UI)
# =========================

@app.post("/reset")
def reset(req: ResetRequest):
    task_id = req.task_id
    env = CustomerSupportEnv(task_id)
    env.reset()
    session_id = str(uuid.uuid4())
    
    config = REWARD_CONFIG[task_id]
    max_score = config["max_score"]
    
    random_order_id = str(random.randint(10000, 99999))
    user_prompt = random.choice(AMBIGUOUS_QUERIES) if task_id == "ambiguous_request" else None
    
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
        "start_time": datetime.now().isoformat(),
        "order_id": random_order_id,
        "order_checked": False,
        "address_collected": False,
        "address_confirmed": False,
        "policy_explained": False,
        "resolved": False,
        "user_prompt": user_prompt,
        "last_action_type": None
    }
    
    return {
        "session_id": session_id,
        "task_id": task_id,
        "message": "Environment reset successfully",
        "steps": 0,
        "done": False,
        "score": 0.0,
        "max_score": max_score,
        "score_display": f"0.00 / {max_score}",
        "order_id": random_order_id,
        "user_prompt": user_prompt,
        "state": {
            "order_checked": False,
            "address_collected": False,
            "address_confirmed": False,
            "policy_explained": False,
            "resolved": False,
            "steps": 0
        }
    }

@app.get("/reset/{task_id}")
def reset_get(task_id: str):
    req = ResetRequest(task_id=task_id)
    return reset(req)

@app.get("/state/{session_id}")
def get_state(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    
    session = sessions[session_id]
    
    return {
        "session_id": session_id,
        "task_id": session["task_id"],
        "state": {
            "steps": session["steps"],
            "order_checked": session.get("order_checked", False),
            "address_collected": session.get("address_collected", False),
            "address_confirmed": session.get("address_confirmed", False),
            "policy_explained": session.get("policy_explained", False),
            "resolved": session.get("resolved", False),
            "expert_used_count": session.get("expert_used_count", 0),
            "total_reward": session.get("total_reward", 0.0),
            "done": session.get("done", False)
        }
    }

# =========================
# AI MODEL CALL
# =========================

def call_ai_model(task_id: str, session_state: Dict, use_expert: bool = False) -> Dict[str, Any]:
    """Call the AI model to get an action"""
    
    if client is None:
        return {"action_type": "send_reply", "message": "I apologize, but I'm having trouble processing your request."}
    
    sys_prompt = get_task_prompt(task_id, session_state, use_expert)
    
    state_summary = {k: v for k, v in session_state.items() 
                     if k in ['order_checked', 'address_collected', 'address_confirmed', 'policy_explained', 'resolved']}
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": f"Task: {task_id}\nCurrent state: {json.dumps(state_summary)}\nReturn ONLY valid JSON with action_type field."
                }
            ],
            temperature=0.2,
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
            return {"action_type": "send_reply", "message": "I'm not sure how to help. Could you clarify?"}

        data = json.loads(content)
        
        if "action_type" not in data:
            return {"action_type": "send_reply", "message": "Let me try to help you with that."}
        
        if data["action_type"] not in ["lookup_order", "send_reply"]:
            data["action_type"] = "send_reply"
            data["message"] = "I understand your concern. Let me help you with that."
        
        return data
        
    except Exception as e:
        print(f"❌ AI model error: {e}")
        return {"action_type": "send_reply", "message": "I encountered an error. Could you please try again?"}

def get_task_prompt(task_id: str, session_state: Dict, use_expert: bool = False) -> str:
    """Natural language prompts with strict action restrictions"""
    
    base_instructions = (
        "### STRICT RULES ###\n"
        "1. You have ONLY TWO tools: 'lookup_order' and 'send_reply'.\n"
        "2. If you need to talk to the user, you MUST use 'send_reply'.\n"
        "3. NEVER invent action_types like 'ask_address' or 'confirm_order'.\n"
        "4. Your output must be PURE JSON.\n\n"
    )
    
    expert_section = ""
    if use_expert:
        expert_hint = REWARD_CONFIG[task_id]["expert_hint"]
        if isinstance(expert_hint, list):
            order_checked = session_state.get("order_checked", False)
            address_collected = session_state.get("address_collected", False)
            if not order_checked:
                hint = expert_hint[0]
            elif not address_collected:
                hint = expert_hint[1]
            else:
                hint = expert_hint[2]
        else:
            hint = expert_hint
        expert_section = f"\n\n🚨 EXPERT COMMAND: {hint}\n"
    
    if task_id == "order_status_easy":
        return base_instructions + "Task: Check order status. Use lookup_order first.\nRespond in JSON."
    elif task_id == "refund_policy_medium":
        return base_instructions + "Task: Explain refund policy (30-day refund). Must include 'refund'.\nRespond in JSON."
    elif task_id == "address_change_hard":
        return base_instructions + "Task: Change address. Step1: lookup_order, Step2: ask for address, Step3: confirm.\nRespond in JSON."
    elif task_id == "ambiguous_request":
        return base_instructions + "Task: Handle moved address and missing package. Check order first, then address, then resolve.\nRespond in JSON."
    else:
        return base_instructions + "Return valid JSON with action_type field."

# =========================
# VALIDATION FUNCTION
# =========================

def validate_and_update_state(task_id: str, action_dict: Dict, session: Dict, used_expert: bool = False) -> tuple:
    """Validate action and update state"""
    action_type = action_dict.get("action_type", "")
    expert_penalty = -0.2 if used_expert else 0
    message = action_dict.get("message", "").lower()
    
    reward = 0.0
    explanation = ""
    is_valid = False
    
    if task_id == "order_status_easy":
        if not session.get("order_checked", False):
            if action_type == "lookup_order":
                reward = 0.8 + expert_penalty
                explanation = "✅ Order lookup successful"
                is_valid = True
                session["order_checked"] = True
            else:
                reward = -0.5 + expert_penalty
                explanation = f"❌ Need lookup_order first"
                is_valid = False
        else:
            if action_type == "send_reply":
                reward = 0.2 + expert_penalty
                explanation = "✅ Task complete"
                is_valid = True
            else:
                reward = -0.3 + expert_penalty
                explanation = "❌ Should send reply"
                is_valid = False
    
    elif task_id == "refund_policy_medium":
        if not session.get("policy_explained", False):
            if action_type == "send_reply" and "refund" in message:
                reward = 0.8 + expert_penalty
                explanation = "✅ Refund policy explained"
                is_valid = True
                session["policy_explained"] = True
            else:
                reward = -0.4 + expert_penalty
                explanation = "❌ Need send_reply with 'refund'"
                is_valid = False
        else:
            reward = 0.1 + expert_penalty
            explanation = "✅ Already complete"
            is_valid = True
    
    elif task_id == "address_change_hard":
        order_checked = session.get("order_checked", False)
        address_collected = session.get("address_collected", False)
        address_confirmed = session.get("address_confirmed", False)
        
        if not order_checked:
            if action_type == "lookup_order":
                reward = 0.5 + expert_penalty
                explanation = "✅ Order located"
                is_valid = True
                session["order_checked"] = True
            else:
                reward = -0.5 + expert_penalty
                explanation = "❌ Need lookup_order first"
                is_valid = False
        elif not address_collected:
            if action_type == "send_reply" and "address" in message:
                reward = 0.4 + expert_penalty
                explanation = "✅ Address requested"
                is_valid = True
                session["address_collected"] = True
            else:
                reward = -0.4 + expert_penalty
                explanation = "❌ Need ask for address"
                is_valid = False
        elif not address_confirmed:
            if action_type == "send_reply" and "confirm" in message:
                reward = 0.4 + expert_penalty
                explanation = "✅ Address confirmed"
                is_valid = True
                session["address_confirmed"] = True
            else:
                reward = -0.4 + expert_penalty
                explanation = "❌ Need confirmation"
                is_valid = False
        else:
            reward = 0.1 + expert_penalty
            explanation = "✅ Complete"
            is_valid = True
    
    elif task_id == "ambiguous_request":
        order_checked = session.get("order_checked", False)
        address_collected = session.get("address_collected", False)
        resolved = session.get("resolved", False)
        
        if not order_checked:
            if action_type == "lookup_order":
                reward = 0.4 + expert_penalty
                explanation = "✅ Order located"
                is_valid = True
                session["order_checked"] = True
            else:
                reward = -0.5 + expert_penalty
                explanation = "❌ Need lookup_order first"
                is_valid = False
        elif not address_collected:
            if action_type == "send_reply" and "address" in message:
                reward = 0.4 + expert_penalty
                explanation = "✅ Address requested"
                is_valid = True
                session["address_collected"] = True
            else:
                reward = -0.4 + expert_penalty
                explanation = "❌ Need ask for address"
                is_valid = False
        elif not resolved:
            if action_type == "send_reply":
                reward = 0.4 + expert_penalty
                explanation = "✅ Issue resolved"
                is_valid = True
                session["resolved"] = True
            else:
                reward = -0.4 + expert_penalty
                explanation = "❌ Need resolution"
                is_valid = False
        else:
            reward = 0.1 + expert_penalty
            explanation = "✅ Complete"
            is_valid = True
    
    return is_valid, round(max(reward, -0.8), 2), explanation, is_valid

# =========================
# STEP AI ENDPOINT
# =========================

@app.post("/step_ai")
def step_ai(req: StepRequest):
    if req.session_id not in sessions:
        raise HTTPException(404, "Invalid session_id. Please reset first.")

    session = sessions[req.session_id]
    config = REWARD_CONFIG[session["task_id"]]
    max_score = config["max_score"]
    max_steps_limit = MAX_STEPS[session["task_id"]]

    if session["done"]:
        return {
            "message": "Task already completed. Please reset.",
            "done": True,
            "score": min(session["total_reward"], max_score),
            "max_score": max_score,
            "score_display": f"{min(session['total_reward'], max_score):.2f} / {max_score}",
            "step": session["steps"],
            "state": session.get("state", {})
        }

    env = session["env"]
    used_expert = req.use_expert
    
    ai_action = call_ai_model(session["task_id"], session, used_expert)
    
    try:
        action = Action(**ai_action)
        parse_success = True
    except Exception:
        ai_action = {"action_type": "send_reply", "message": "I'm not sure how to help."}
        action = Action(**ai_action)
        parse_success = False
    
    is_valid, reward_value, explanation, is_perfect = validate_and_update_state(
        session["task_id"], ai_action, session, used_expert
    )
    
    if not parse_success:
        reward_value = -0.5
        explanation = "❌ Invalid JSON format"
    
    obs, env_reward, done, info = env.step(action)
    
    env_r = env_reward.value if hasattr(env_reward, "value") else float(env_reward)
    reward_value = 0.7 * reward_value + 0.3 * env_r
    
    session["steps"] += 1
    session["rewards"].append(reward_value)
    session["explanations"].append(explanation)
    session["actions_taken"].append(ai_action)
    session["total_reward"] += reward_value
    session["total_reward"] = min(session["total_reward"], max_score)
    session["total_reward"] = max(session["total_reward"], 0.0)
    
    if used_expert:
        session["expert_used_count"] += 1
    
    if session["steps"] >= max_steps_limit:
        session["done"] = True
        session["total_reward"] = max(0, session["total_reward"] - 0.3)
    
    all_steps_complete = False
    if session["task_id"] == "order_status_easy":
        all_steps_complete = session.get("order_checked", False)
    elif session["task_id"] == "refund_policy_medium":
        all_steps_complete = session.get("policy_explained", False)
    elif session["task_id"] == "address_change_hard":
        all_steps_complete = session.get("order_checked", False) and session.get("address_collected", False) and session.get("address_confirmed", False)
    elif session["task_id"] == "ambiguous_request":
        all_steps_complete = session.get("order_checked", False) and session.get("address_collected", False) and session.get("resolved", False)
    
    if all_steps_complete and not session["done"]:
        session["done"] = True
        session["perfect_completion"] = session["expert_used_count"] == 0 and session["total_reward"] >= max_score * 0.8
    
    score_value = min(session["total_reward"], max_score)
    
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
        "expert_used_count": session["expert_used_count"],
        "state": {
            "order_checked": session.get("order_checked", False),
            "address_collected": session.get("address_collected", False),
            "address_confirmed": session.get("address_confirmed", False),
            "steps": session["steps"],
            "done": session["done"]
        }
    }

@app.get("/session/{session_id}")
def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    
    session = sessions[session_id]
    config = REWARD_CONFIG[session["task_id"]]
    max_score = config["max_score"]
    score_value = min(session["total_reward"], max_score)
    
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

@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {"task_id": "order_status_easy", "name": "Order Status Query", "description": "Customer wants to check their order status", "difficulty": "easy", "max_steps": 5, "max_score": 1.0},
            {"task_id": "refund_policy_medium", "name": "Refund Policy Explanation", "description": "Customer wants to know the refund policy", "difficulty": "medium", "max_steps": 5, "max_score": 1.0},
            {"task_id": "address_change_hard", "name": "Address Change Request", "description": "Customer wants to change their shipping address", "difficulty": "hard", "max_steps": 8, "max_score": 1.0},
            {"task_id": "ambiguous_request", "name": "Moved & Missing Package", "description": "Customer issue with moved address and missing package", "difficulty": "hard+", "max_steps": 10, "max_score": 1.0}
        ]
    }

@app.get("/health")
def health():
    return {"status": "healthy", "active_sessions": len(sessions), "ai_model": MODEL_NAME if client else "fallback"}

# =========================
# ROOT ENDPOINT
# =========================

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Customer Support RL Environment</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #0a0a1a; color: #e2e8f0; }
            .container { max-width: 800px; margin: 0 auto; }
            .card { background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; margin: 20px 0; }
            h1 { color: #8b5cf6; }
            code { background: #1a1a2a; padding: 2px 5px; border-radius: 3px; }
            pre { background: #1a1a2a; padding: 10px; border-radius: 5px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Customer Support RL Environment</h1>
            <div class="card">
                <h2>OpenEnv Compliant Environment</h2>
                <p>A realistic customer support simulation for training AI agents.</p>
                <h3>Available Tasks:</h3>
                <ul>
                    <li><strong>order_status_easy</strong> - Check order status (Easy)</li>
                    <li><strong>refund_policy_medium</strong> - Explain refund policy (Medium)</li>
                    <li><strong>address_change_hard</strong> - Change shipping address (Hard)</li>
                    <li><strong>ambiguous_request</strong> - Handle moved address + missing package (Hard+)</li>
                </ul>
            </div>
            <div class="card">
                <h2>API Endpoints</h2>
                <pre>
POST   /openenv/reset     - Reset environment
POST   /openenv/step      - Take an action
GET    /openenv/state     - Get current state
GET    /openenv/tasks     - List tasks
GET    /openenv/validate  - Validate OpenEnv compliance
                </pre>
            </div>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
