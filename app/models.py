from pydantic import BaseModel
from typing import Optional, Dict, Any
class Action(BaseModel):
    action_type: str
    order_id: Optional[str] = None
    message: Optional[str] = None

class Observation(BaseModel):
    task_id: str
    history: list
    done: bool

class Reward(BaseModel):
    value: float
    info: Optional[Dict[str, Any]] = None