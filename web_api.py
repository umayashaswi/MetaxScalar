import json
import os
import uuid
import random
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

class Action(BaseModel):
    action_type: str
    order_id: str = None
    message: str = None

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
# OPENENV COMPATIBILITY ENDPOINTS (IMPORTANT)
# =========================

@app.post("/openenv/reset")
def openenv_reset():
    # Create env with default task
    env = CustomerSupportEnv(task_id="order_status_easy")
    obs = env.reset()

    return {
        "task_id": obs.task_id,
        "history": obs.history,
        "done": obs.done,
        "observation_text": obs.observation_text
    }


@app.post("/openenv/step")
def openenv_step(action: Action):
    env = CustomerSupportEnv(task_id="order_status_easy")
    obs = env.reset()

    obs, reward, done, info = env.step(action)

    return {
        "observation": {
            "task_id": obs.task_id,
            "history": obs.history,
            "done": obs.done,
            "observation_text": obs.observation_text
        },
        "reward": reward.value,
        "done": done,
        "info": info
    }


@app.get("/openenv/validate")
def openenv_validate():
    return {"status": "ok"}


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


# =========================
# REACT-STYLE PREMIUM UI WITH COMPLETION MESSAGE & TASK SWITCH RESET
# =========================

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
        
        /* FIXED SCORE RING - COMPLETE CIRCLE */
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
        
        // FIXED: Force FULL UI reset when switching tasks
        function selectTask(task) {
            selectedTask = task;
            sessionId = null;
            steps = [];
            score = 0;
            maxScore = task.max_score ?? 1;
            done = false;
            perfect = false;
            completionMsg = null;
            
            // 🔥 CLEAR UI manually
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
        
        // FIXED: Stop rendering stats when no session
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
        
        // FIXED: Stop rendering performance when no session
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
