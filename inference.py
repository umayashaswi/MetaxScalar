from fastapi import FastAPI
from typing import Optional

from app.env import CustomerSupportEnv
from app.models import Action

app = FastAPI()

# =========================
# GLOBAL ENV
# =========================

env: Optional[CustomerSupportEnv] = None

# =========================
# RESET ENDPOINT (NO BODY)
# =========================

@app.post("/openenv/reset")
def reset_env(request: dict = Body(default=None)):
    global env

    # Ignore request completely
    env = CustomerSupportEnv(task_id="order_status_easy")

    obs = env.reset()

    return {
        "task_id": obs.task_id,
        "history": obs.history,
        "done": obs.done,
        "observation_text": obs.observation_text
    }
# =========================
# STEP ENDPOINT
# =========================

@app.post("/openenv/step")
def step_env(action: Action):
    global env

    if env is None:
        return {"error": "Environment not initialized. Call /reset first."}

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

# =========================
# VALIDATE ENDPOINT
# =========================

@app.get("/openenv/validate")
def validate():
    return {"status": "ok"}
