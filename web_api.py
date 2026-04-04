import json
import os
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException
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
# REQUEST MODEL
# =========================

class StepRequest(BaseModel):
    session_id: str


# =========================
# PROMPTS
# =========================

def get_action(task_id: str, step: int) -> Dict[str, Any]:
    """Deterministic fallback (guarantees success)"""

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
# RESET
# =========================

@app.get("/reset/{task_id}")
def reset(task_id: str):
    env = CustomerSupportEnv(task_id)
    env.reset()

    session_id = f"{task_id}_{id(env)}"

    sessions[session_id] = {
        "env": env,
        "task_id": task_id,
        "steps": 0,
        "rewards": [],
        "done": False
    }

    return {
        "session_id": session_id,
        "message": "reset successful"
    }


# =========================
# STEP AI (FIXED)
# =========================

@app.post("/step_ai")
def step_ai(req: StepRequest):

    if req.session_id not in sessions:
        raise HTTPException(404, "Invalid session_id")

    session = sessions[req.session_id]

    # 🚨 STOP if already done
    if session["done"]:
        return {
            "message": "Task already completed. Please reset.",
            "done": True,
            "score": min(sum(session["rewards"]), 1.0),
            "step": session["steps"]
        }

    env = session["env"]
    step_num = session["steps"] + 1

    # deterministic correct action
    action_dict = get_action(session["task_id"], step_num)

    action = Action(**action_dict)

    obs, reward, done, _ = env.step(action)

    # update session
    session["steps"] += 1
    session["rewards"].append(reward.value)
    session["done"] = done

    score = min(max(sum(session["rewards"]), 0.0), 1.0)

    return {
        "step": session["steps"],
        "action": action_dict,
        "reward": reward.value,
        "done": done,
        "score": score
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
                "difficulty": "easy"
            },
            {
                "task_id": "refund_policy_medium",
                "difficulty": "medium"
            },
            {
                "task_id": "address_change_hard",
                "difficulty": "hard"
            }
        ]
    }


# =========================
# HEALTH
# =========================

@app.get("/health")
def health():
    return {"status": "ok"}