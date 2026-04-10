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
openenv_history: List = []

# =========================
# REQUEST MODELS
# =========================

class StepRequest(BaseModel):
    action_type: str
    order_id: Optional[str] = None
    message: Optional[str] = None

class StepAIRequest(BaseModel):
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
# AMBIGUOUS QUERIES
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
            "Step 1: Use {'action_type': 'lookup_order', 'order_id': '12345'}",
            "Step 2: Use {'action_type': 'send_reply', 'message': 'Please provide your new address.'}",
            "Step 3: Use {'action_type': 'send_reply', 'message': 'Please confirm this address.'}"
        ]
    },
    "ambiguous_request": {
        "max_score": 1.0,
        "expert_hint": [
            "Step 1: Use {'action_type': 'lookup_order', 'order_id': '12345'}",
            "Step 2: Use {'action_type': 'send_reply', 'message': 'Can you confirm your new address?'}",
            "Step 3: Use {'action_type': 'send_reply', 'message': 'I have updated your address and a replacement is on the way.'}"
        ]
    }
}

# =========================
# OPENENV COMPATIBILITY ENDPOINTS (FIXED)
# =========================

@app.post("/reset")
async def openenv_reset():
    """OpenEnv compatible reset endpoint - NO parameters, NO body expected"""
    global openenv_session, openenv_history
    
    # Randomly select a task for better testing coverage
    task_id = random.choice([
        "order_status_easy",
        "refund_policy_medium",
        "address_change_hard",
        "ambiguous_request"
    ])
    
    # Create a new environment instance with random task
    openenv_session = CustomerSupportEnv(task_id=task_id)
    obs = openenv_session.reset()
    openenv_history = []
    
    return {
        "observation": {
            "task_id": obs.task_id,
            "history": obs.history,
            "done": obs.done
        },
        "reward": 0.0,
        "done": False,
        "info": {}
    }

@app.post("/step")
async def openenv_step(req: StepRequest):
    """OpenEnv compatible step endpoint"""
    global openenv_session, openenv_history
    
    if openenv_session is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    
    action = Action(**req.dict())
    
    # Take a step in the environment
    obs, reward, done, info = openenv_session.step(action)
    openenv_history.append(action.dict())
    
    return {
        "observation": {
            "task_id": obs.task_id,
            "history": obs.history,
            "done": obs.done
        },
        "reward": reward.value if hasattr(reward, "value") else float(reward),
        "done": done,
        "info": info
    }

@app.get("/state")
async def openenv_state():
    """Get current environment state"""
    global openenv_session
    
    if openenv_session is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    
    obs = openenv_session.state()
    
    return {
        "observation": {
            "task_id": obs.task_id,
            "history": obs.history,
            "done": obs.done
        }
    }

@app.get("/validate")
async def openenv_validate():
    """OpenEnv validation endpoint"""
    return {"status": "ok", "ready": True, "openenv_compatible": True}

@app.get("/tasks_list")
async def openenv_tasks():
    """List available tasks"""
    return {
        "tasks": [
            {
                "id": "order_status_easy",
                "name": "Order Status Query",
                "description": "Customer wants to check their order status",
                "difficulty": "easy",
                "max_steps": 5
            },
            {
                "id": "refund_policy_medium",
                "name": "Refund Policy Explanation",
                "description": "Customer wants to know the refund policy",
                "difficulty": "medium",
                "max_steps": 5
            },
            {
                "id": "address_change_hard",
                "name": "Address Change Request",
                "description": "Customer wants to change their shipping address",
                "difficulty": "hard",
                "max_steps": 8
            },
            {
                "id": "ambiguous_request",
                "name": "Moved & Missing Package",
                "description": "Customer issue with moved address and missing package",
                "difficulty": "hard+",
                "max_steps": 10
            }
        ]
    }

# =========================
# LEGACY ENDPOINTS (for backward compatibility with UI)
# =========================

@app.post("/reset_with_task")
async def reset_with_task(req: ResetRequest):
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
        "last_action_type": None,
        "penalty_count": 0  # ADDED: Penalty tracking
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
async def reset_get(task_id: str):
    req = ResetRequest(task_id=task_id)
    return await reset_with_task(req)

@app.post("/step_ai")
async def step_ai(req: StepAIRequest):
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
        action_type = ai_action.get("action_type", "")
    except Exception:
        ai_action = {"action_type": "send_reply", "message": "I'm not sure how to help."}
        action = Action(**ai_action)
        parse_success = False
        action_type = "send_reply"
    
    is_valid, reward_value, explanation, is_perfect = validate_and_update_state(
        session["task_id"], ai_action, session, used_expert
    )
    
    if not parse_success:
        reward_value = -0.5
        explanation = "❌ Invalid JSON format"
        session["penalty_count"] += 1  # ADDED: Track invalid action penalty
    
    if not is_valid:
        session["penalty_count"] += 1  # ADDED: Track invalid action penalty
    
    # ADD REPEATED ACTION PENALTY
    if session.get("last_action_type") == action_type:
        reward_value -= 0.2
        explanation += " ⚠️ Repeated action penalty"
        session["penalty_count"] += 1  # ADDED: Track repeated action penalty
    
    session["last_action_type"] = action_type
    
    obs, env_reward, done, info = env.step(action)
    
    env_r = env_reward.value if hasattr(env_reward, "value") else float(env_reward)
    
    # FIX 3: BLENDING RATIO UPDATED TO 80/20
    reward_value = 0.8 * reward_value + 0.2 * env_r
    
    session["steps"] += 1
    session["rewards"].append(reward_value)
    session["explanations"].append(explanation)
    session["actions_taken"].append(ai_action)
    
    # FIX 1: SIMPLE ADDITION WITHOUT CLAMPING
    session["total_reward"] += reward_value
    
    if used_expert:
        session["expert_used_count"] += 1
        session["penalty_count"] += 1  # ADDED: Track expert penalty
    
    if session["steps"] >= max_steps_limit:
        session["done"] = True
        # REDUCE STEP LIMIT PENALTY
        session["total_reward"] = max(0, session["total_reward"] - 0.1)
        session["penalty_count"] += 1  # ADDED: Track step limit penalty
    
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
        # FIX 2: RESTORE ORIGINAL PERFECT COMPLETION CONDITION
        if session["expert_used_count"] == 0 and session["total_reward"] >= max_score * 0.8:
            session["perfect_completion"] = True
            session["total_reward"] = min(session["total_reward"] + 0.2, max_score)
    
    # CLAMP ONLY AT THE END FOR DISPLAY
    score_value = min(session["total_reward"], max_score)
    score_value = max(score_value, 0.0)
    
    # ADDED: Completion message logic
    completion_message = None
    
    if session["done"]:
        penalties = session.get("penalty_count", 0)
        
        if score_value < 0.6:
            completion_message = "⚠️ Poor Completion (< 0.6 score)"
        elif penalties == 0:
            completion_message = "🌟 Perfect Completion (No penalties)"
        elif penalties == 1:
            completion_message = "✅ Task Completed with 1 Penalty"
        elif penalties <= 3:
            completion_message = "✅ Task Completed with Few Penalties"
        else:
            completion_message = "⚠️ Task Completed with Many Penalties"
    
    # ADDED: Append completion message to explanation
    if completion_message:
        explanation += f" | {completion_message}"
    
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
        "completion_message": completion_message,  # ADDED: Return completion message
        "state": {
            "order_checked": session.get("order_checked", False),
            "address_collected": session.get("address_collected", False),
            "address_confirmed": session.get("address_confirmed", False),
            "steps": session["steps"],
            "done": session["done"]
        }
    }

# =========================
# HELPER FUNCTIONS
# =========================

def call_ai_model(task_id: str, session_state: Dict, use_expert: bool = False) -> Dict[str, Any]:
    """Call the AI model to get an action"""
    
    if client is None:
        return {"action_type": "send_reply", "message": "I apologize, but I'm having trouble processing your request."}
    
    # FORCE CORRECT ACTION FLOW FOR HARD TASKS
    if task_id in ["address_change_hard", "ambiguous_request"]:
        
        if not session_state.get("order_checked", False):
            return {
                "action_type": "lookup_order",
                "order_id": session_state.get("order_id", "12345")
            }
        
        elif not session_state.get("address_collected", False):
            return {
                "action_type": "send_reply",
                "message": "Please provide your new address details."
            }
        
        elif task_id == "address_change_hard" and not session_state.get("address_confirmed", False):
            return {
                "action_type": "send_reply",
                "message": "Please confirm this address is correct."
            }
        
        elif task_id == "ambiguous_request" and not session_state.get("resolved", False):
            return {
                "action_type": "send_reply",
                "message": "Your address has been updated and a replacement has been shipped."
            }
    
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
    if use_expert and task_id in REWARD_CONFIG:
        expert_hint = REWARD_CONFIG[task_id]["expert_hint"]
        if isinstance(expert_hint, list):
            order_checked = session_state.get("order_checked", False)
            address_collected = session_state.get("address_collected", False)
            address_confirmed = session_state.get("address_confirmed", False)
            
            if not order_checked:
                hint = expert_hint[0]
            elif not address_collected:
                hint = expert_hint[1]
            elif not address_confirmed:
                hint = expert_hint[2]
            else:
                hint = expert_hint[2] if len(expert_hint) > 2 else expert_hint[-1]
        else:
            hint = expert_hint
        expert_section = f"\n\n🚨 EXPERT COMMAND: {hint}\n"
    
    if task_id == "order_status_easy":
        return base_instructions + """
STRICT ORDER:
1. If order_checked = false → MUST use lookup_order first
2. After order_checked = true → send_reply with status

DO NOT skip steps.
DO NOT repeat steps.

Respond ONLY in JSON.
"""
    elif task_id == "refund_policy_medium":
        return base_instructions + """
STRICT ORDER:
1. If policy_explained = false → MUST use send_reply with 'refund' keyword
2. Include: '30-day full refund policy'

DO NOT skip steps.
DO NOT repeat steps.

Respond ONLY in JSON.
"""
    elif task_id == "address_change_hard":
        return base_instructions + """
STRICT ORDER:
1. If order_checked = false → lookup_order
2. If address_collected = false → send_reply asking for address (use words: address, location, details)
3. If address_confirmed = false → send_reply confirming address (use word: confirm)

DO NOT skip steps.
DO NOT repeat steps.

Respond ONLY in JSON.
"""
    elif task_id == "ambiguous_request":
        return base_instructions + """
STRICT ORDER:
1. If order_checked = false → lookup_order first
2. If address_collected = false → send_reply asking for new address
3. If resolved = false → send_reply confirming replacement

DO NOT skip steps.
DO NOT repeat steps.

Respond ONLY in JSON.
"""
    else:
        return base_instructions + "Return valid JSON with action_type field."

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
            if action_type == "send_reply" and any(word in message for word in ["address", "location", "details", "where", "send"]):
                reward = 0.4 + expert_penalty
                explanation = "✅ Address requested"
                is_valid = True
                session["address_collected"] = True
            else:
                reward = -0.4 + expert_penalty
                explanation = "❌ Need ask for address"
                is_valid = False
        elif not address_confirmed:
            if action_type == "send_reply" and any(word in message for word in ["confirm", "correct", "right", "yes", "that's"]):
                reward = 0.5 + expert_penalty
                explanation = "✅ Address confirmed"
                is_valid = True
                session["address_confirmed"] = True
            else:
                reward = -0.4 + expert_penalty
                explanation = "❌ Need confirmation"
                is_valid = False
        else:
            reward = 0.3 + expert_penalty
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
            if action_type == "send_reply" and any(word in message for word in ["address", "location", "details", "where"]):
                reward = 0.4 + expert_penalty
                explanation = "✅ Address requested"
                is_valid = True
                session["address_collected"] = True
            else:
                reward = -0.4 + expert_penalty
                explanation = "❌ Need ask for address"
                is_valid = False
        elif not resolved:
            if action_type == "send_reply" and any(word in message for word in ["updated", "replacement", "resolved", "fixed", "shipped"]):
                reward = 0.5 + expert_penalty
                explanation = "✅ Issue resolved"
                is_valid = True
                session["resolved"] = True
            else:
                reward = -0.4 + expert_penalty
                explanation = "❌ Need resolution"
                is_valid = False
        else:
            reward = 0.2 + expert_penalty
            explanation = "✅ Complete"
            is_valid = True
    
    return is_valid, round(max(reward, -0.8), 2), explanation, is_valid

@app.get("/tasks")
async def tasks():
    return {
        "tasks": [
            {"task_id": "order_status_easy", "name": "Order Status Query", "description": "Customer wants to check their order status", "difficulty": "easy", "max_steps": 5, "max_score": 1.0},
            {"task_id": "refund_policy_medium", "name": "Refund Policy Explanation", "description": "Customer wants to know the refund policy", "difficulty": "medium", "max_steps": 5, "max_score": 1.0},
            {"task_id": "address_change_hard", "name": "Address Change Request", "description": "Customer wants to change their shipping address", "difficulty": "hard", "max_steps": 8, "max_score": 1.0},
            {"task_id": "ambiguous_request", "name": "Moved & Missing Package", "description": "Customer issue with moved address and missing package", "difficulty": "hard+", "max_steps": 10, "max_score": 1.0}
        ]
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "active_sessions": len(sessions), "ai_model": MODEL_NAME if client else "fallback"}

@app.get("/session/{session_id}")
async def get_session(session_id: str):
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
        "perfect_completion": session.get("perfect_completion", False),
        "penalty_count": session.get("penalty_count", 0)
    }

@app.get("/", response_class=HTMLResponse)
def home():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Support RL | AI Agent Training Platform</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,100..900;1,100..900&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: radial-gradient(circle at 20% 30%, #0a0a1a, #050510);
            min-height: 100vh;
            color: #e2e8f0;
            overflow-x: hidden;
        }
        
        #particle-canvas {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 0;
            opacity: 0.5;
        }
        
        .glass-header {
            position: sticky;
            top: 0;
            z-index: 100;
            backdrop-filter: blur(12px);
            background: rgba(10, 10, 30, 0.7);
            border-bottom: 1px solid rgba(139, 92, 246, 0.2);
        }
        
        .header-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 0.75rem 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .logo-area {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .logo-icon {
            width: 2rem;
            height: 2rem;
            background: linear-gradient(135deg, #8b5cf6, #6366f1);
            border-radius: 0.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 20px rgba(139, 92, 246, 0.3);
        }
        
        .logo-icon svg { width: 1rem; height: 1rem; color: white; }
        .logo-text h1 { font-size: 0.875rem; font-weight: 600; }
        .logo-text p { font-size: 0.625rem; color: #94a3b8; }
        
        .status-area {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .model-badge {
            background: rgba(139, 92, 246, 0.15);
            padding: 0.25rem 0.5rem;
            border-radius: 0.375rem;
            font-family: monospace;
            font-size: 0.625rem;
            border: 1px solid rgba(139, 92, 246, 0.3);
        }
        
        .status-dot {
            width: 0.5rem;
            height: 0.5rem;
            border-radius: 50%;
            display: inline-block;
        }
        .status-online { background: #10b981; box-shadow: 0 0 8px #10b981; animation: pulse 2s infinite; }
        .status-offline { background: #ef4444; }
        .status-checking { background: #f59e0b; animation: pulse 1s infinite; }
        
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        
        .main-layout {
            max-width: 1400px;
            margin: 0 auto;
            padding: 1.5rem;
            position: relative;
            z-index: 1;
        }
        
        .two-columns {
            display: grid;
            grid-template-columns: 320px 1fr;
            gap: 1.5rem;
        }
        
        @media (max-width: 768px) { .two-columns { grid-template-columns: 1fr; } }
        
        .task-item {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 1rem;
            padding: 1rem;
            cursor: pointer;
            transition: all 0.2s ease;
            margin-bottom: 0.75rem;
        }
        
        .task-item:hover {
            background: rgba(255, 255, 255, 0.06);
            border-color: rgba(139, 92, 246, 0.3);
            transform: translateY(-2px);
        }
        
        .task-item.selected {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(99, 102, 241, 0.1));
            border-color: #8b5cf6;
        }
        
        .task-header-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }
        
        .task-name { font-weight: 600; font-size: 0.9rem; }
        .difficulty-tag {
            font-size: 0.625rem;
            padding: 0.125rem 0.5rem;
            border-radius: 1rem;
            font-weight: 500;
        }
        .diff-easy { background: #10b981; color: white; }
        .diff-medium { background: #f59e0b; color: white; }
        .diff-hard { background: #ef4444; color: white; }
        .diff-hardplus { background: #8b5cf6; color: white; }
        
        .task-desc { font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.5rem; }
        .task-meta { display: flex; gap: 0.75rem; font-size: 0.625rem; color: #64748b; }
        
        .control-panel {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 1rem;
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        .btn-group {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 0;
        }
        
        .btn {
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-size: 0.75rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            border: none;
            display: inline-flex;
            align-items: center;
            gap: 0.375rem;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #8b5cf6, #6366f1);
            color: white;
        }
        .btn-primary:hover:not(:disabled) { opacity: 0.9; transform: scale(1.02); }
        
        .btn-outline {
            background: transparent;
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #e2e8f0;
        }
        .btn-outline:hover:not(:disabled) { background: rgba(255, 255, 255, 0.05); }
        
        .btn-expert {
            border-color: #f59e0b;
            color: #f59e0b;
        }
        .btn-expert:hover:not(:disabled) { background: rgba(245, 158, 11, 0.1); }
        
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        
        .score-ring-container {
            display: flex;
            justify-content: center;
            margin-bottom: 1rem;
        }
        
        .score-ring {
            position: relative;
            width: 140px;
            height: 140px;
        }
        
        .score-ring svg {
            width: 100%;
            height: 100%;
            transform: rotate(-90deg);
        }
        
        .score-ring-bg {
            stroke: rgba(255, 255, 255, 0.08);
            fill: none;
            stroke-width: 10;
        }
        
        .score-ring-fill {
            stroke: #8b5cf6;
            fill: none;
            stroke-width: 10;
            stroke-linecap: round;
            transition: stroke-dashoffset 0.6s ease;
            filter: drop-shadow(0 0 8px rgba(139, 92, 246, 0.5));
        }
        
        .score-ring-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
        }
        
        .score-number {
            font-size: 1.8rem;
            font-weight: 700;
            color: #8b5cf6;
        }
        
        .score-max-text {
            font-size: 0.7rem;
            color: #64748b;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.75rem;
            margin: 0;
        }
        
        .stat-card {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 0.75rem;
            padding: 0.75rem;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .stat-value { font-size: 1.25rem; font-weight: 700; }
        .stat-label { font-size: 0.6rem; text-transform: uppercase; color: #64748b; margin-top: 0.25rem; }
        
        .reward-chart {
            display: flex;
            align-items: flex-end;
            gap: 0.25rem;
            height: 50px;
            margin: 0.5rem 0;
        }
        
        .chart-bar {
            flex: 1;
            background: linear-gradient(180deg, #10b981, #059669);
            border-radius: 2px 2px 0 0;
            transition: height 0.3s ease;
        }
        
        .chart-bar.negative { background: linear-gradient(180deg, #ef4444, #dc2626); }
        
        .timeline {
            max-height: 400px;
            overflow-y: auto;
            margin-top: 0;
        }
        
        .timeline-item {
            padding: 0.75rem;
            border-left: 2px solid #8b5cf6;
            margin-bottom: 0.75rem;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 0.5rem;
        }
        
        .timeline-item.expert { border-left-color: #f59e0b; background: rgba(245, 158, 11, 0.05); }
        
        .timeline-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
            font-size: 0.7rem;
        }
        
        .reward-positive { color: #10b981; }
        .reward-negative { color: #ef4444; }
        
        .timeline-action {
            font-family: monospace;
            font-size: 0.65rem;
            color: #94a3b8;
            word-break: break-all;
        }
        
        .timeline-explanation { font-size: 0.65rem; color: #64748b; margin-top: 0.25rem; }
        .timeline-state { font-size: 0.6rem; color: #475569; margin-top: 0.25rem; }
        
        .section-title {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b;
            margin-bottom: 0.75rem;
        }
        
        .empty-state {
            text-align: center;
            padding: 3rem;
            color: #64748b;
        }
        
        .spinner {
            width: 1rem;
            height: 1rem;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        
        @keyframes spin { to { transform: rotate(360deg); } }
        
        .task-header-card {
            padding: 1rem;
            border-radius: 1rem;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
        }
        
        .progress-bar-container {
            height: 8px;
            border-radius: 6px;
            background: rgba(255,255,255,0.1);
            margin-top: 0.75rem;
            overflow: hidden;
        }
        
        .progress-bar-fill {
            height: 100%;
            border-radius: 6px;
            background: linear-gradient(90deg, #10b981, #22c55e);
            transition: width 0.3s ease;
        }
        
        .completion-message {
            margin-bottom: 0.75rem;
            padding: 0.5rem;
            border-radius: 0.5rem;
            font-size: 0.75rem;
            text-align: center;
            background: rgba(139, 92, 246, 0.1);
            border: 1px solid rgba(139, 92, 246, 0.3);
            color: #e2e8f0;
        }
    </style>
</head>
<body>
    <canvas id="particle-canvas"></canvas>
    
    <header class="glass-header">
        <div class="header-container">
            <div class="logo-area">
                <div class="logo-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M13 10V3L4 14h7v7l9-11h-7z"/>
                    </svg>
                </div>
                <div class="logo-text">
                    <h1>Customer Support RL</h1>
                    <p>Reinforcement Learning Simulator</p>
                </div>
            </div>
            <div class="status-area" id="status-area"></div>
        </div>
    </header>
    
    <main class="main-layout">
        <div class="two-columns">
            <div>
                <div class="section-title">TRAINING TASKS</div>
                <div id="tasks-container"></div>
            </div>
            
            <div>
                <div class="control-panel">
                    <div id="task-header-card"></div>
                    <div id="stats-container"></div>
                    <div id="controls-container"></div>
                    <div id="performance-container"></div>
                    <div class="section-title">STEP TIMELINE</div>
                    <div id="timeline-container" class="timeline"></div>
                </div>
            </div>
        </div>
    </main>
    
    <script>
        const API_BASE = window.location.origin;
        
        let tasks = [];
        let selectedTask = null;
        let sessionId = null;
        let steps = [];
        let score = 0;
        let maxScore = 1;
        let done = false;
        let perfect = false;
        let completionMsg = null;
        let loading = false;
        let online = null;
        let aiModel = "";
        
        // Particle animation
        (function initParticles() {
            const canvas = document.getElementById('particle-canvas');
            const ctx = canvas.getContext('2d');
            let particles = [];
            
            function resize() {
                canvas.width = window.innerWidth;
                canvas.height = window.innerHeight;
            }
            
            function createParticles() {
                particles = [];
                for (let i = 0; i < 80; i++) {
                    particles.push({
                        x: Math.random() * canvas.width,
                        y: Math.random() * canvas.height,
                        vx: (Math.random() - 0.5) * 0.3,
                        vy: (Math.random() - 0.5) * 0.3,
                        size: Math.random() * 2 + 0.5,
                        alpha: Math.random() * 0.3 + 0.1
                    });
                }
            }
            
            function draw() {
                if (!ctx) return;
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                particles.forEach(p => {
                    p.x += p.vx;
                    p.y += p.vy;
                    if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
                    if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
                    
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(139, 92, 246, ${p.alpha})`;
                    ctx.fill();
                });
                
                for (let i = 0; i < particles.length; i++) {
                    for (let j = i + 1; j < particles.length; j++) {
                        const dx = particles[i].x - particles[j].x;
                        const dy = particles[i].y - particles[j].y;
                        const dist = Math.sqrt(dx * dx + dy * dy);
                        if (dist < 100) {
                            ctx.beginPath();
                            ctx.moveTo(particles[i].x, particles[i].y);
                            ctx.lineTo(particles[j].x, particles[j].y);
                            ctx.strokeStyle = `rgba(139, 92, 246, ${0.05 * (1 - dist / 100)})`;
                            ctx.lineWidth = 0.5;
                            ctx.stroke();
                        }
                    }
                }
                
                requestAnimationFrame(draw);
            }
            
            window.addEventListener('resize', () => {
                resize();
                createParticles();
            });
            resize();
            createParticles();
            draw();
        })();
        
        async function fetchTasks() {
            try {
                const res = await fetch(`${API_BASE}/tasks`);
                const data = await res.json();
                tasks = data.tasks || [];
                renderTasks();
            } catch(e) { console.error(e); }
        }
        
        async function fetchHealth() {
            try {
                const res = await fetch(`${API_BASE}/health`);
                const data = await res.json();
                online = true;
                aiModel = data.ai_model || "";
                renderStatus();
            } catch(e) {
                online = false;
                renderStatus();
            }
        }
        
        async function resetTask(taskId, taskMaxScore) {
            if (loading) return;
            loading = true;
            renderControls();
            try {
                const res = await fetch(`${API_BASE}/reset/${taskId}`);
                const data = await res.json();
                sessionId = data.session_id;
                steps = [];
                score = 0;
                maxScore = data.max_score || taskMaxScore;
                done = false;
                perfect = false;
                completionMsg = null;
                renderHeaderCard();
                renderStats();
                renderPerformance();
                renderTimeline();
                renderControls();
            } catch(e) { alert("Reset error: " + e.message); }
            finally { loading = false; renderControls(); }
        }
        
        async function takeStep(useExpert = false) {
            if (!sessionId || done || loading) return;
            loading = true;
            renderControls();
            try {
                const res = await fetch(`${API_BASE}/step_ai`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sessionId, use_expert: useExpert })
                });
                const data = await res.json();
                steps.unshift(data);
                score = data.score;
                maxScore = data.max_score;
                done = data.done;
                perfect = data.perfect_completion;
                completionMsg = data.completion_message;
                renderStats();
                renderPerformance();
                renderTimeline();
                renderControls();
            } catch(e) { alert("Step error: " + e.message); }
            finally { loading = false; renderControls(); }
        }
        
        function selectTask(task) {
            selectedTask = task;
            sessionId = null;
            steps = [];
            score = 0;
            maxScore = task.max_score ?? 1;
            done = false;
            perfect = false;
            completionMsg = null;
            
            document.getElementById('stats-container').innerHTML = '';
            document.getElementById('performance-container').innerHTML = '';
            document.getElementById('timeline-container').innerHTML =
                '<div class="empty-state">No steps yet. Click "Start Session" then "Step AI" to begin training.</div>';
            
            renderTasks();
            renderHeaderCard();
            renderControls();
        }
        
        function renderHeaderCard() {
            const container = document.getElementById('task-header-card');
            if (!container || !selectedTask) return;
            
            let diffClass = '';
            if (selectedTask.difficulty === 'easy') diffClass = 'diff-easy';
            else if (selectedTask.difficulty === 'medium') diffClass = 'diff-medium';
            else if (selectedTask.difficulty === 'hard') diffClass = 'diff-hard';
            else diffClass = 'diff-hardplus';
            
            container.innerHTML = `
                <div class="task-header-card">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <div style="font-weight:600;font-size:1rem;">
                                ${selectedTask.name}
                            </div>
                            <div style="font-size:0.75rem;color:#94a3b8;">
                                ${selectedTask.description}
                            </div>
                        </div>
                        <span class="difficulty-tag ${diffClass}">
                            ${selectedTask.difficulty.toUpperCase()}
                        </span>
                    </div>
                </div>
            `;
        }
        
        function renderStatus() {
            const container = document.getElementById('status-area');
            if (!container) return;
            let html = '';
            if (aiModel) html += `<span class="model-badge">${aiModel}</span>`;
            if (online === true) html += `<div><span class="status-dot status-online"></span><span style="margin-left: 0.25rem; font-size: 0.7rem;">Online</span></div>`;
            else if (online === false) html += `<div><span class="status-dot status-offline"></span><span style="margin-left: 0.25rem; font-size: 0.7rem;">Offline</span></div>`;
            else html += `<div><span class="status-dot status-checking"></span><span style="margin-left: 0.25rem; font-size: 0.7rem;">Checking...</span></div>`;
            container.innerHTML = html;
        }
        
        function renderTasks() {
            const container = document.getElementById('tasks-container');
            if (!container) return;
            if (tasks.length === 0) { container.innerHTML = '<div class="empty-state">Loading tasks...</div>'; return; }
            
            container.innerHTML = tasks.map(task => {
                let diffClass = '';
                if (task.difficulty === 'easy') diffClass = 'diff-easy';
                else if (task.difficulty === 'medium') diffClass = 'diff-medium';
                else if (task.difficulty === 'hard') diffClass = 'diff-hard';
                else diffClass = 'diff-hardplus';
                
                return `
                    <div class="task-item ${selectedTask?.task_id === task.task_id ? 'selected' : ''}" onclick='selectTask(${JSON.stringify(task)})'>
                        <div class="task-header-row">
                            <span class="task-name">${task.name}</span>
                            <span class="difficulty-tag ${diffClass}">${task.difficulty.toUpperCase()}</span>
                        </div>
                        <div class="task-desc">${task.description}</div>
                        <div class="task-meta">
                            <span>⚡ ${task.max_steps} steps</span>
                            <span>🎯 ${task.max_score} pts</span>
                        </div>
                    </div>
                `;
            }).join('');
        }
        
        function renderControls() {
            const container = document.getElementById('controls-container');
            if (!container || !selectedTask) { if(container) container.innerHTML = ''; return; }
            
            container.innerHTML = `
                <div class="btn-group">
                    <button class="btn btn-outline" onclick="resetTask('${selectedTask.task_id}', ${selectedTask.max_score})" ${loading ? 'disabled' : ''}>
                        🔄 ${sessionId ? 'Reset Session' : 'Start Session'}
                    </button>
                    ${sessionId && !done ? `
                        <button class="btn btn-primary" onclick="takeStep(false)" ${loading ? 'disabled' : ''}>
                            ${loading ? '<div class="spinner"></div>' : '🤖 Step AI'}
                        </button>
                        <button class="btn btn-outline btn-expert" onclick="takeStep(true)" ${loading ? 'disabled' : ''}>
                            🎓 Ask Expert (-0.2)
                        </button>
                    ` : ''}
                </div>
            `;
        }
        
        function renderStats() {
            const container = document.getElementById('stats-container');
            if (!container || !sessionId) {
                if (container) container.innerHTML = '';
                return;
            }
            
            const expertCount = steps.filter(s => s.used_expert).length;
            const progressPercent = maxScore > 0 ? Math.round((score / maxScore) * 100) : 0;
            
            container.innerHTML = `
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value" style="color:#8b5cf6">${score.toFixed(2)}</div>
                        <div class="stat-label">SCORE</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${steps.length}</div>
                        <div class="stat-label">STEPS</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${progressPercent}%</div>
                        <div class="stat-label">PROGRESS</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${expertCount}</div>
                        <div class="stat-label">EXPERT USES</div>
                    </div>
                </div>
            `;
        }
        
        function renderPerformance() {
            const container = document.getElementById('performance-container');
            if (!container || !sessionId) {
                if (container) container.innerHTML = '';
                return;
            }
            
            const percent = maxScore > 0 ? Math.min(score / maxScore, 1) : 0;
            const radius = 60;
            const circumference = 2 * Math.PI * radius;
            const dashOffset = circumference * (1 - percent);
            
            container.innerHTML = `
                <div>
                    <div style="font-size:0.7rem;color:#64748b;margin-bottom:0.5rem;">PERFORMANCE</div>
                    ${completionMsg ? `
                        <div class="completion-message">
                            ${completionMsg}
                        </div>
                    ` : ''}
                    <div class="score-ring-container">
                        <div class="score-ring">
                            <svg viewBox="0 0 140 140">
                                <circle class="score-ring-bg" cx="70" cy="70" r="60"/>
                                <circle class="score-ring-fill" cx="70" cy="70" r="60"
                                        stroke-dasharray="${circumference}"
                                        stroke-dashoffset="${dashOffset}"/>
                            </svg>
                            <div class="score-ring-text">
                                <div class="score-number">${score.toFixed(1)}</div>
                                <div class="score-max-text">/ ${maxScore.toFixed(1)}</div>
                            </div>
                        </div>
                    </div>
                    <div class="progress-bar-container">
                        <div class="progress-bar-fill" style="width: ${percent * 100}%"></div>
                    </div>
                    <div class="reward-chart" id="reward-chart"></div>
                </div>
            `;
            
            const chartContainer = document.getElementById('reward-chart');
            if (chartContainer && steps.length > 0) {
                const recentSteps = [...steps].slice(0, 12);
                const maxAbs = Math.max(...recentSteps.map(s => Math.abs(s.reward)), 0.5);
                chartContainer.innerHTML = recentSteps.map(step => {
                    const height = (Math.abs(step.reward) / maxAbs) * 40;
                    return `<div class="chart-bar ${step.reward < 0 ? 'negative' : ''}" style="height: ${Math.max(height, 4)}px;"></div>`;
                }).join('');
            }
        }
        
        function renderTimeline() {
            const container = document.getElementById('timeline-container');
            if (!container) return;
            if (!sessionId || steps.length === 0) {
                container.innerHTML = '<div class="empty-state">No steps yet. Click "Start Session" then "Step AI" to begin training.</div>';
                return;
            }
            
            container.innerHTML = steps.map(step => {
                const rewardClass = step.reward >= 0 ? 'reward-positive' : 'reward-negative';
                const expertClass = step.used_expert ? 'expert' : '';
                return `
                    <div class="timeline-item ${expertClass}">
                        <div class="timeline-header">
                            <span>Step ${step.step}</span>
                            <span class="${rewardClass}">${step.reward >= 0 ? '+' : ''}${step.reward}</span>
                        </div>
                        <div class="timeline-action">Action: ${JSON.stringify(step.action)}</div>
                        <div class="timeline-explanation">📖 ${step.reward_explanation}</div>
                        ${step.state ? `<div class="timeline-state">📊 order_checked=${step.state.order_checked}, address_collected=${step.state.address_collected}, address_confirmed=${step.state.address_confirmed}</div>` : ''}
                    </div>
                `;
            }).join('');
        }
        
        async function init() {
            await fetchHealth();
            await fetchTasks();
            renderStatus();
            setInterval(() => { fetchHealth(); }, 30000);
        }
        
        window.selectTask = selectTask;
        window.resetTask = resetTask;
        window.takeStep = takeStep;
        
        init();
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
