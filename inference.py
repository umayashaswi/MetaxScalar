import os
import asyncio
import json
from typing import Optional, List
from openai import OpenAI
from app.env import CustomerSupportEnv
from app.models import Action

# =========================
# CONFIG
# =========================

# Use environment variables for API credentials (required for HF Spaces)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("GROQ_API_KEY") or os.getenv("API_KEY")

if not API_KEY:
    API_KEY = "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"

if not API_KEY:
    raise ValueError("Set API_KEY, GROQ_API_KEY, or HF_TOKEN environment variable")

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL
)

# =========================
# LOGGING (Matches required format)
# =========================

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    if isinstance(action, dict):
        action = json.dumps(action)
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# =========================
# SIMPLIFIED MODEL CALL
# =========================

def call_model(task_id: str, history: List, sys_prompt: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": f"Task: {task_id}\nHistory: {json.dumps(history)}\nReturn JSON action."
            }
        ],
        temperature=0.3,
        max_tokens=150
    )

    content = response.choices[0].message.content.strip()

    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end != 0:
            data = json.loads(content[start:end])
        else:
            return {"action_type": "send_reply", "message": "I apologize, but I'm having trouble processing your request."}
    except:
        return {"action_type": "send_reply", "message": "I apologize, but I'm having trouble processing your request."}

    return data

# =========================
# NATURAL PROMPTS
# =========================

def get_system_prompt(task_id: str) -> str:
    core_rules = (
        "You are a Customer Support AI. You must follow these STRICT rules:\n"
        "1. ONLY use 'lookup_order' to find data or 'send_reply' to talk to the user.\n"
        "2. To ask a question (like asking for an address), you MUST use 'send_reply'.\n"
        "3. Your output must be valid JSON ONLY.\n\n"
    )

    if task_id == "order_status_easy":
        return core_rules + (
            "Task: Find the customer's order status.\n"
            "Logic: Use lookup_order first.\n"
            "Respond in JSON format with fields: action_type, order_id (if needed), message (if needed)."
        )
    
    elif task_id == "refund_policy_medium":
        return core_rules + (
            "Task: Explain that refunds are allowed within 30 days.\n"
            "Logic: Use send_reply to explain the policy. Must include the word 'refund'.\n"
            "Respond in JSON format with fields: action_type, order_id (if needed), message (if needed)."
        )
    
    elif task_id == "address_change_hard":
        return core_rules + (
            "Task: Change shipping address.\n"
            "Logic: 1. lookup_order -> 2. send_reply (ask for address) -> 3. send_reply (confirm).\n"
            "IMPORTANT: If the history shows you already looked up the order, DO NOT do it again. Move to asking for the address.\n"
            "Respond in JSON format with fields: action_type, order_id (if needed), message (if needed)."
        )
    
    else:
        return core_rules + "Return valid JSON with action_type field."

async def run_task(task_id):
    log_start(task=task_id, env="CustomerSupport-v1", model=MODEL_NAME)

    env = CustomerSupportEnv(task_id)
    obs = env.reset()

    rewards = []
    sys_prompt = get_system_prompt(task_id)

    for step in range(1, 11):
        try:
            action_dict = call_model(task_id, obs.history, sys_prompt)
            
            if "action_type" not in action_dict:
                action_dict = {"action_type": "send_reply", "message": "I apologize, but I'm having trouble processing your request."}
            
            try:
                action = Action(**action_dict)
            except Exception as e:
                action = Action(action_type="send_reply", message="I apologize, but I'm having trouble processing your request.")
                log_step(step, action_dict, -0.2, False, str(e))

            obs, reward, done, _ = env.step(action)
            rewards.append(reward.value)
            log_step(step, action_dict, reward.value, done, None)

            if done:
                break

        except Exception as e:
            error_msg = str(e)
            log_step(step, {}, 0.0, True, error_msg)
            break

    if rewards:
        total_score = sum(rewards)
        score = max(min(total_score, 1.0), 0.0)
        success = score >= 0.7
    else:
        score = 0.0
        success = False

    log_end(success, len(rewards), score, rewards)

async def main():
    tasks = [
        "order_status_easy",
        "refund_policy_medium",
        "address_change_hard"
    ]

    for t in tasks:
        await run_task(t)

if __name__ == "__main__":
    asyncio.run(main())
