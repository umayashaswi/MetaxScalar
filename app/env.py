from pydantic import BaseModel
from typing import Optional, Dict, Any, List

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

class CustomerSupportEnv:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.history = []
        self.done = False
        self.step_count = 0
        self.total_reward = 0.0
        
        # Task-specific state
        if task_id == "address_change_hard":
            self.expected_steps = 3
        elif task_id == "ambiguous_request":
            self.expected_steps = 3
        else:
            self.expected_steps = 1

    def reset(self):
        self.history = []
        self.done = False
        self.step_count = 0
        self.total_reward = 0.0
        return self.state()

    def state(self):
        return Observation(
            task_id=self.task_id,
            history=self.history,
            done=self.done
        )

    def step(self, action: Action):
        reward = self.evaluate(action)
        self.history.append(action.dict())
        self.step_count += 1
        self.total_reward += reward.value
        
        # Check if task is complete
        if self.step_count >= self.expected_steps:
            self.done = True

        return self.state(), Reward(value=reward.value), self.done, {}

    # =========================
    # TASK LOGIC
    # =========================

    def evaluate(self, action: Action):
        step = self.step_count
        
        # Order Status Easy Task
        if self.task_id == "order_status_easy":
            if action.action_type == "lookup_order" and action.order_id == "12345":
                return Reward(value=1.0)
            return Reward(value=-0.3)

        # Refund Policy Medium Task
        elif self.task_id == "refund_policy_medium":
            if action.action_type == "send_reply" and action.message and "refund" in action.message.lower():
                return Reward(value=1.0)
            return Reward(value=-0.3)

        # Address Change Hard Task
        elif self.task_id == "address_change_hard":
            if step == 0:  # Step 1
                if action.action_type == "lookup_order" and action.order_id == "12345":
                    return Reward(value=0.6)
                return Reward(value=-0.3)
            elif step == 1:  # Step 2
                if action.action_type == "send_reply" and action.message and "address" in action.message.lower():
                    return Reward(value=0.2)
                return Reward(value=-0.3)
            elif step == 2:  # Step 3
                if action.action_type == "send_reply" and action.message and "confirm" in action.message.lower():
                    return Reward(value=0.2)
                return Reward(value=-0.3)
            return Reward(value=0.0)

        # Ambiguous Request Task (NEW)
        elif self.task_id == "ambiguous_request":
            if step == 0:  # Step 1 - Lookup order
                if action.action_type == "lookup_order" and action.order_id == "12345":
                    return Reward(value=0.3)
                return Reward(value=-0.3)
            elif step == 1:  # Step 2 - Ask for address confirmation
                if action.action_type == "send_reply" and action.message and "address" in action.message.lower():
                    return Reward(value=0.35)
                return Reward(value=-0.3)
            elif step == 2:  # Step 3 - Resolve both issues
                if action.action_type == "send_reply" and action.message:
                    # Check if message addresses both issues
                    msg = action.message.lower()
                    if "address" in msg and ("order" in msg or "delivery" in msg or "ship" in msg):
                        return Reward(value=0.35)
                    elif "address" in msg:
                        return Reward(value=0.2)  # Partial credit
                    return Reward(value=-0.2)
                return Reward(value=-0.3)
            return Reward(value=0.0)

        return Reward(value=0.0)