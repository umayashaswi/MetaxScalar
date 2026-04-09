from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Literal

class Action(BaseModel):
    """
    STRICT SCHEMA: Forces the AI to choose between valid actions.
    This prevents the 'ask_address' or 'confirm_order' errors seen in your logs.
    """
    action_type: Literal["lookup_order", "send_reply"]
    order_id: Optional[str] = Field(
        default=None, 
        description="The 5-digit order ID required for lookup_order"
    )
    message: Optional[str] = Field(
        default=None, 
        description="The text response required for send_reply"
    )

class Observation(BaseModel):
    """
    Standard OpenEnv Observation model.
    """
    task_id: str
    history: List[Dict[str, Any]] = Field(default_factory=list)
    done: bool = False
    observation_text: Optional[str] = None # Optional: helps the agent understand the current state

class Reward(BaseModel):
    """
    Standard OpenEnv Reward model.
    Value must be between 0.0 and 1.0 for the final score.
    """
    value: float = Field(ge=-1.0, le=1.0)
    explanation: Optional[str] = None
    info: Dict[str, Any] = Field(default_factory=dict)