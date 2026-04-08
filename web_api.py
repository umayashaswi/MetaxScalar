import json
import os
import uuid
import random
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from datetime import datetime

from app.env import CustomerSupportEnv
from app.models import Action  # Import Action from app.models

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
# REWARD CONFIGURATION (UPDATED WITH STRICT HINTS)
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
# IMPROVED TASK-SPECIFIC PROMPTS WITH STRICT RULES
# =========================

def get_task_prompt(task_id: str, session_state: Dict, use_expert: bool = False) -> str:
    """Natural language prompts with strict action restrictions"""
    
    base_instructions = (
        "### STRICT RULES ###\n"
        "1. You have ONLY TWO tools: 'lookup_order' and 'send_reply'.\n"
        "2. If you need to talk to the user, ask a question, or give info, you MUST use 'send_reply'.\n"
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
        return (
            base_instructions +
            "You are a customer support assistant helping a customer check their order status.\n\n"
            "Think before acting:\n"
            "- What is the quickest way to find the order status?\n"
            "- Avoid unnecessary conversation\n\n"
            "Respond ONLY in JSON with fields: action_type, order_id (if needed), message (if needed)."
            f"{expert_section}"
        )
    
    elif task_id == "refund_policy_medium":
        return (
            base_instructions +
            "You are a customer support assistant explaining a refund policy.\n\n"
            "Think before acting:\n"
            "- The user wants information, not an action on their order\n"
            "- Provide a clear and helpful explanation\n\n"
            "Respond ONLY in JSON with fields: action_type, order_id (if needed), message (if needed)."
            f"{expert_section}"
        )
    
    elif task_id == "address_change_hard":
        order_checked = session_state.get("order_checked", False)
        address_collected = session_state.get("address_collected", False)
        address_confirmed = session_state.get("address_confirmed", False)
        
        state_hint = ""
        if not order_checked:
            state_hint = "\nCurrent status: Need to locate the order first."
        elif not address_collected:
            state_hint = "\nCurrent status: Order located. Now need the new address."
        elif not address_confirmed:
            state_hint = "\nCurrent status: Address received. Need confirmation."
        else:
            state_hint = "\nCurrent status: All steps complete."
        
        if order_checked:
            base_instructions += "⚠️ ALERT: Order is already checked. 'lookup_order' is now FORBIDDEN. Use 'send_reply'.\n"
        
        return (
            base_instructions +
            "You are a customer support assistant helping a user change their shipping address.\n\n"
            "Think step-by-step:\n"
            "- Identify what information is needed before making changes\n"
            "- Gather required details before confirming anything\n"
            "- Ensure actions follow a logical sequence\n\n"
            "Important:\n"
            "- Do not confirm address before asking for it\n"
            "- Avoid repeating the same action\n\n"
            f"{state_hint}\n\n"
            "Respond ONLY in JSON with fields: action_type, order_id (if needed), message (if needed)."
            f"{expert_section}"
        )
    
    elif task_id == "ambiguous_request":
        user_prompt = session_state.get("user_prompt", "I moved recently and didn't receive my package.")
        order_checked = session_state.get("order_checked", False)
        address_collected = session_state.get("address_collected", False)
        resolved = session_state.get("resolved", False)
        
        state_hint = ""
        if not order_checked:
            state_hint = "\nCurrent status: Need to check the order status first."
        elif not address_collected:
            state_hint = "\nCurrent status: Order checked. Need to confirm new address."
        elif not resolved:
            state_hint = "\nCurrent status: Address confirmed. Need to resolve the issue."
        else:
            state_hint = "\nCurrent status: Issue resolved."
        
        if order_checked:
            base_instructions += "⚠️ ALERT: Order is already checked. 'lookup_order' is now FORBIDDEN. Use 'send_reply'.\n"
        
        return (
            base_instructions +
            f"You are a customer support assistant handling a complex issue:\n"
            f"The customer says: \"{user_prompt}\"\n\n"
            "Think carefully step-by-step:\n"
            "- Identify all issues in the request (address change and missing delivery)\n"
            "- Gather missing information before confirming anything\n"
            "- Resolve the issue logically and completely\n\n"
            "Important guidelines:\n"
            "- Do not confirm address before asking for it\n"
            "- Avoid repeating the same action multiple times\n"
            "- Ensure your response addresses both the address issue and delivery issue\n\n"
            f"{state_hint}\n\n"
            "Respond ONLY in JSON with fields: action_type, order_id (if needed), message (if needed)."
            f"{expert_section}"
        )
    
    else:
        return "Return valid JSON with action_type field."

# =========================
# AI MODEL CALL
# =========================

def call_ai_model(task_id: str, session_state: Dict, use_expert: bool = False) -> Dict[str, Any]:
    """Call the AI model to get an action"""
    
    if client is None:
        return {"action_type": "send_reply", "message": "I apologize, but I'm having trouble processing your request. Could you please rephrase?"}
    
    sys_prompt = get_task_prompt(task_id, session_state, use_expert)
    
    state_summary = {k: v for k, v in session_state.items() 
                     if k in ['order_checked', 'address_collected', 'address_confirmed', 'policy_explained', 'resolved']}
    
    try:
        print(f"🤖 Calling AI model for {task_id}...")
        print(f"📊 Current state: {state_summary}")
        
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

# =========================
# VALIDATION WITH STATE TRACKING AND STRONGER PENALTIES
# =========================

def validate_and_update_state(task_id: str, action_dict: Dict, session: Dict, used_expert: bool = False) -> tuple:
    """Validate action and update state with appropriate rewards and stronger penalties"""
    
    action_type = action_dict.get("action_type", "")
    expert_penalty = -0.2 if used_expert else 0
    message = action_dict.get("message", "").lower()
    
    reward = 0.0
    explanation = ""
    is_valid = False
    
    last_action = session.get("last_action_type", None)
    
    # Order Status Easy Task
    if task_id == "order_status_easy":
        order_checked = session.get("order_checked", False)
        
        if not order_checked:
            if action_type == "lookup_order":
                reward = 0.8 + expert_penalty
                explanation = "✅ Correct: Order lookup successful"
                is_valid = True
                session["order_checked"] = True
            else:
                reward = -0.5 + expert_penalty
                explanation = f"❌ CRITICAL: You cannot help the user without looking up the order first. Got {action_type}"
                is_valid = False
        else:
            if action_type == "send_reply":
                reward = 0.2 + expert_penalty
                explanation = "✅ Task complete"
                is_valid = True
            else:
                reward = -0.3 + expert_penalty
                explanation = "❌ Already checked order, should send reply"
                is_valid = False
    
    # Refund Policy Medium Task
    elif task_id == "refund_policy_medium":
        policy_explained = session.get("policy_explained", False)
        
        if not policy_explained:
            if action_type == "send_reply":
                if "refund" in message:
                    reward = 0.8 + expert_penalty
                    explanation = "✅ Correct: Refund policy explained"
                    is_valid = True
                    session["policy_explained"] = True
                else:
                    reward = 0.1 + expert_penalty
                    explanation = "⚠️ Sent reply but missing 'refund' keyword"
                    is_valid = False
            else:
                reward = -0.4 + expert_penalty
                explanation = f"❌ Wrong: Should send reply, got {action_type}"
                is_valid = False
        else:
            reward = 0.1 + expert_penalty
            explanation = "✅ Task already complete"
            is_valid = True
    
    # Address Change Hard Task
    elif task_id == "address_change_hard":
        order_checked = session.get("order_checked", False)
        address_collected = session.get("address_collected", False)
        address_confirmed = session.get("address_confirmed", False)
        
        if not address_collected and "confirm" in message:
            reward = -0.6 + expert_penalty
            explanation = "❌ INVALID: Cannot confirm address before asking for it!"
            is_valid = False
            session["last_action_type"] = action_type
            return is_valid, round(max(reward, -0.8), 2), explanation, is_valid
        
        if not order_checked and action_type == "send_reply" and "address" in message:
            reward = -0.5 + expert_penalty
            explanation = "❌ INVALID: Must lookup order before asking for address!"
            is_valid = False
            session["last_action_type"] = action_type
            return is_valid, round(max(reward, -0.8), 2), explanation, is_valid
        
        if not order_checked:
            if action_type == "lookup_order":
                reward = 0.5 + expert_penalty
                explanation = "✅ Step 1: Order located"
                is_valid = True
                session["order_checked"] = True
            else:
                reward = -0.5 + expert_penalty
                explanation = f"❌ CRITICAL: You cannot help the user without looking up the order first. Got {action_type}"
                is_valid = False
        elif not address_collected:
            if action_type == "send_reply" and "address" in message:
                reward = 0.4 + expert_penalty
                explanation = "✅ Step 2: Address requested"
                is_valid = True
                session["address_collected"] = True
            else:
                reward = -0.4 + expert_penalty
                explanation = "❌ Step 2: Should ask for new address"
                is_valid = False
        elif not address_confirmed:
            if action_type == "send_reply" and "confirm" in message:
                reward = 0.4 + expert_penalty
                explanation = "✅ Step 3: Address confirmed"
                is_valid = True
                session["address_confirmed"] = True
            else:
                reward = -0.4 + expert_penalty
                explanation = "❌ Step 3: Should ask for confirmation"
                is_valid = False
        else:
            reward = 0.1 + expert_penalty
            explanation = "✅ Task already complete"
            is_valid = True
        
        if last_action == action_type and reward < 0:
            reward -= 0.3
            explanation += " (Repeated same mistake - penalty increased!)"
    
    # Ambiguous Request Task
    elif task_id == "ambiguous_request":
        order_checked = session.get("order_checked", False)
        address_collected = session.get("address_collected", False)
        resolved = session.get("resolved", False)
        
        if not address_collected and "confirm" in message:
            reward = -0.6 + expert_penalty
            explanation = "❌ INVALID: Cannot confirm address before asking for it!"
            is_valid = False
            session["last_action_type"] = action_type
            return is_valid, round(max(reward, -0.8), 2), explanation, is_valid
        
        if not order_checked and action_type == "send_reply":
            reward = -0.5 + expert_penalty
            explanation = "❌ INVALID: Must check order status first!"
            is_valid = False
            session["last_action_type"] = action_type
            return is_valid, round(max(reward, -0.8), 2), explanation, is_valid
        
        if not order_checked:
            if action_type == "lookup_order":
                reward = 0.4 + expert_penalty
                explanation = "✅ Step 1: Order located"
                is_valid = True
                session["order_checked"] = True
            else:
                reward = -0.5 + expert_penalty
                explanation = f"❌ CRITICAL: You must check the order status first. Got {action_type}"
                is_valid = False
        elif not address_collected:
            if action_type == "send_reply" and "address" in message:
                reward = 0.4 + expert_penalty
                explanation = "✅ Step 2: Address confirmation requested"
                is_valid = True
                session["address_collected"] = True
            else:
                reward = -0.4 + expert_penalty
                explanation = "❌ Step 2: Should ask for address confirmation"
                is_valid = False
        elif not resolved:
            if action_type == "send_reply":
                reward = 0.4 + expert_penalty
                explanation = "✅ Step 3: Issue resolved"
                is_valid = True
                session["resolved"] = True
            else:
                reward = -0.4 + expert_penalty
                explanation = f"❌ Step 3: Should send resolution, got {action_type}"
                is_valid = False
        else:
            reward = 0.1 + expert_penalty
            explanation = "✅ Task already complete"
            is_valid = True
        
        if last_action == action_type and reward < 0:
            reward -= 0.3
            explanation += " (Repeated same mistake - penalty increased!)"
    
    session["last_action_type"] = action_type
    reward = max(reward, -0.8)
    return is_valid, round(reward, 2), explanation, is_valid

# =========================
# OPENENV COMPATIBILITY ENDPOINTS (FIXED)
# =========================

@app.post("/openenv/reset")
def openenv_reset():
    """OpenEnv compatible reset endpoint - no body required"""
    env = CustomerSupportEnv(task_id="order_status_easy")
    obs = env.reset()
    
    # Store in sessions for compatibility
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "env": env,
        "task_id": "order_status_easy",
        "steps": 0,
        "rewards": [],
        "done": False,
        "total_reward": 0.0,
        "history": []
    }
    
    return {
        "task_id": obs.task_id,
        "history": obs.history,
        "done": obs.done,
        "observation_text": None  # OpenEnv doesn't require this field
    }

@app.post("/openenv/step")
def openenv_step(action: Action):
    """OpenEnv compatible step endpoint"""
    # Find the most recent session
    if not sessions:
        return {"error": "Environment not initialized. Call /openenv/reset first."}
    
    session_id = list(sessions.keys())[-1]
    session = sessions[session_id]
    env = session["env"]
    
    obs, reward, done, info = env.step(action)
    
    # Update session
    session["steps"] += 1
    session["rewards"].append(reward.value)
    session["total_reward"] += reward.value
    session["done"] = done
    session["history"].append(action.dict())
    
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
# RESET - WITH DYNAMIC STATE
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

# =========================
# STATE ENDPOINT
# =========================

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
# STEP AI
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
            "state": {
                "order_checked": session.get("order_checked", False),
                "address_collected": session.get("address_collected", False),
                "address_confirmed": session.get("address_confirmed", False),
                "steps": session["steps"],
                "done": True
            }
        }

    env = session["env"]
    used_expert = req.use_expert
    
    ai_action = call_ai_model(session["task_id"], session, used_expert)
    
    try:
        action = Action(**ai_action)
        parse_success = True
    except Exception:
        ai_action = {"action_type": "send_reply", "message": "I'm not sure how to help. Could you clarify?"}
        action = Action(**ai_action)
        parse_success = False
    
    is_valid, reward_value, explanation, is_perfect = validate_and_update_state(
        session["task_id"], ai_action, session, used_expert
    )
    
    if not parse_success:
        reward_value = -0.5
        explanation = "❌ Invalid JSON format from model"
    
    obs, env_reward, done, info = env.step(action)
    
    env_r = env_reward.value if hasattr(env_reward, "value") else float(env_reward)
    reward_value = 0.7 * reward_value + 0.3 * env_r
    
    reward_value -= 0.05
    
    if len(session["actions_taken"]) >= 2:
        last_two_same = (
            session["actions_taken"][-1].get("action_type") == 
            session["actions_taken"][-2].get("action_type") == 
            ai_action.get("action_type")
        )
        if last_two_same:
            reward_value -= 0.25
    
    elapsed_seconds = 0
    try:
        start_time = datetime.fromisoformat(session["start_time"])
        elapsed_seconds = (datetime.now() - start_time).seconds
        if elapsed_seconds > 30:
            reward_value -= 0.05
    except:
        pass
    
    reward_value = max(reward_value, -0.8)
    
    session["steps"] += 1
    session["rewards"].append(reward_value)
    session["explanations"].append(explanation)
    session["actions_taken"].append(ai_action)
    session["total_reward"] += reward_value
    session["total_reward"] = min(session["total_reward"], max_score)
    session["total_reward"] = max(session["total_reward"], 0.0)
    session["history"].append(ai_action)
    
    if used_expert:
        session["expert_used_count"] += 1
    
    if session["steps"] >= max_steps_limit:
        session["done"] = True
        session["total_reward"] = max(0, session["total_reward"] - 0.3)
        return {
            "step": session["steps"],
            "action": ai_action,
            "reward": round(reward_value, 2),
            "reward_explanation": explanation,
            "done": True,
            "perfect_completion": False,
            "score": session["total_reward"],
            "max_score": max_score,
            "score_display": f"{session['total_reward']:.2f} / {max_score}",
            "total_reward": session["total_reward"],
            "is_valid": is_valid,
            "used_expert": used_expert,
            "expert_penalty": -0.2 if used_expert else 0,
            "expert_used_count": session["expert_used_count"],
            "completion_message": "❌ Max steps reached",
            "state": {
                "order_checked": session.get("order_checked", False),
                "address_collected": session.get("address_collected", False),
                "address_confirmed": session.get("address_confirmed", False),
                "steps": session["steps"],
                "done": True
            }
        }
    
    all_steps_complete = False
    
    if session["task_id"] == "order_status_easy":
        all_steps_complete = session.get("order_checked", False)
    elif session["task_id"] == "refund_policy_medium":
        all_steps_complete = session.get("policy_explained", False)
    elif session["task_id"] == "address_change_hard":
        all_steps_complete = (
            session.get("order_checked", False) and
            session.get("address_collected", False) and
            session.get("address_confirmed", False)
        )
    elif session["task_id"] == "ambiguous_request":
        all_steps_complete = (
            session.get("order_checked", False) and
            session.get("address_collected", False) and
            session.get("resolved", False)
        )
    
    if all_steps_complete and not session["done"]:
        session["done"] = True
        perfect_completion = (
            session["expert_used_count"] == 0 and
            session["steps"] <= max_steps_limit - 2 and
            session["total_reward"] >= max_score * 0.8
        )
        session["perfect_completion"] = perfect_completion
    
    score_value = min(session["total_reward"], max_score)
    
    completion_message = None
    if session["done"]:
        if session.get("perfect_completion"):
            completion_message = "🎉 Perfect completion!"
        elif session["expert_used_count"] > 0:
            completion_message = f"⚠️ Completed with expert assistance ({session['expert_used_count']} time(s))"
        elif session["total_reward"] < max_score * 0.6:
            completion_message = "❌ Poor completion"
        else:
            completion_message = "✅ Completed successfully"
    
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
        "completion_message": completion_message,
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
        "perfect_completion": session.get("perfect_completion", False),
        "session_state": {
            "order_checked": session.get("order_checked", False),
            "address_collected": session.get("address_collected", False),
            "address_confirmed": session.get("address_confirmed", False)
        }
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


# Keep the HTML UI but simplified for brevity
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
            button { background: #8b5cf6; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            pre { background: #1a1a2a; padding: 10px; border-radius: 5px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Customer Support RL Environment</h1>
            <div class="card">
                <h2>API Endpoints</h2>
                <ul>
                    <li><code>POST /openenv/reset</code> - Reset environment</li>
                    <li><code>POST /openenv/step</code> - Take an action</li>
                    <li><code>GET /openenv/validate</code> - Validate environment</li>
                    <li><code>GET /tasks</code> - List available tasks</li>
                    <li><code>POST /reset/{task_id}</code> - Reset with specific task</li>
                    <li><code>POST /step_ai</code> - AI-powered step</li>
                    <li><code>GET /health</code> - Health check</li>
                </ul>
            </div>
            <div class="card">
                <h2>Available Tasks</h2>
                <ul>
                    <li><strong>order_status_easy</strong> - Check order status</li>
                    <li><strong>refund_policy_medium</strong> - Explain refund policy</li>
                    <li><strong>address_change_hard</strong> - Change shipping address</li>
                    <li><strong>ambiguous_request</strong> - Handle moved address and missing package</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
