from fastapi import FastAPI
from app.env import CustomerSupportEnv
from app.models import SupportAction

app = FastAPI()
env = CustomerSupportEnv()

@app.post("/reset")
async def reset(task_id: str = "order_status_easy"):
    obs = await env.reset(task_id=task_id)
    return {"observation": obs}

@app.post("/step")
async def step(action: SupportAction):
    obs, reward, done, info = await env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
async def state():
    return {"state": env.history}

@app.get("/")
def read_root():
    return {"status": "online", "message": "Customer Support Simulator is Running"}