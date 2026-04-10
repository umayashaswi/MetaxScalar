import json
import asyncio
import os
from typing import List, Optional

from openai import OpenAI
from app.env import CustomerSupportEnv
from app.models import Action

# =========================
# CONFIG
# =========================

API_KEY = (
    os.getenv("GROQ_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("HF_TOKEN")
)

MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")

if not API_KEY:
    raise ValueError("❌ Set API_KEY, GROQ_API_KEY, or HF_TOKEN")

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL
)

# =========================
# LOGGING (REQUIRED FORMAT)
# =========================

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    done_val = str(done).lower()

    if isinstance(action, dict):
        action = json.dumps(action)

    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True
    )

# =========================
# MODEL CALL
# =========================

def call_model(task_id: str, history: List, sys_prompt: str) -> dict:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": f"Task: {task_id}\nHistory: {json.dumps(history)}\nReturn JSON action."
                }
            ],
            temperature=0.2,
            max_tokens=150
        )

        content = response.choices[0].message.content.strip()

        # Extract JSON safely
        start = content.find("{")
        end = content.rfind("}") + 1

        if start != -1 and end != 0:
            return json.loads(content[start:end])

    except Exception:
        pass

    # fallback
    return {
        "action_type": "send_reply",
        "message": "I apologize, but I'm having trouble processing your request."
    }

# =========================
# PROMPTS
# =========================

def get_system_prompt(task_id: str) -> str:
    base = (
        "You are a Customer Support AI.\n"
        "Rules:\n"
        "1. Only use 'lookup_order' or 'send_reply'\n"
        "2. Use send_reply to talk\n"
        "3. Output ONLY JSON\n\n"
    )

    if task_id == "order_status_easy":
        return base + "First lookup order, then reply with status."

    elif task_id == "refund_policy_medium":
        return base + "Explain 30-day refund policy using 'refund' word."

    elif task_id == "address_change_hard":
        return base + (
            "Steps:\n"
            "1. lookup_order\n"
            "2. ask for address\n"
            "3. confirm address\n"
            "Do not repeat steps."
        )

    else:
        return base

# =========================
# RUN TASK
# =========================

async def run_task(task_id: str):
    log_start(task=task_id, env="CustomerSupport-v1", model=MODEL_NAME)

    env = CustomerSupportEnv(task_id)
    obs = env.reset()

    rewards = []
    sys_prompt = get_system_prompt(task_id)

    for step in range(1, 11):
        try:
            action_dict = call_model(task_id, obs.history, sys_prompt)

            if "action_type" not in action_dict:
                action_dict = {
                    "action_type": "send_reply",
                    "message": "Fallback response"
                }

            try:
                action = Action(**action_dict)
            except Exception as e:
                action = Action(
                    action_type="send_reply",
                    message="Invalid action fallback"
                )
                log_step(step, action_dict, -0.2, False, str(e))
                continue

            obs, reward, done, _ = env.step(action)

            r = reward.value if hasattr(reward, "value") else float(reward)
            rewards.append(r)

            log_step(step, action_dict, r, done, None)

            if done:
                break

        except Exception as e:
            log_step(step, {}, 0.0, True, str(e))
            break

    # =========================
    # FINAL SCORE
    # =========================

    if rewards:
        total = sum(rewards)
        score = max(0.0, min(total, 1.0))  # clamp
        success = score >= 0.7
    else:
        score = 0.0
        success = False

    log_end(success, len(rewards), score, rewards)

# =========================
# MAIN
# =========================

async def main():
    tasks = [
        "order_status_easy",
        "refund_policy_medium",
        "address_change_hard",
        "ambiguous_request"
    ]

    for t in tasks:
        await run_task(t)


if __name__ == "__main__":
    asyncio.run(main())
