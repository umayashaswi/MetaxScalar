from typing import Tuple, Dict, Any, List
from dataclasses import dataclass

# =========================
# ACTION MODEL
# =========================

@dataclass
class Action:
    action_type: str
    order_id: str = None
    message: str = None


# =========================
# OBSERVATION MODEL
# =========================

@dataclass
class Observation:
    task_id: str
    history: List[Dict]
    done: bool


# =========================
# ENVIRONMENT
# =========================

class CustomerSupportEnv:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.reset()

    def reset(self) -> Observation:
        self.history = []
        self.done = False

        self.state = {
            "order_checked": False,
            "address_collected": False,
            "address_confirmed": False,
            "policy_explained": False,
            "resolved": False
        }

        return Observation(self.task_id, self.history, self.done)

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        if self.done:
            return self._get_obs(), 0.0, True, {}

        reward = 0.0
        info = {}

        # =========================
        # TASK LOGIC
        # =========================

        if self.task_id == "order_status_easy":
            if not self.state["order_checked"]:
                if action.action_type == "lookup_order":
                    self.state["order_checked"] = True
                    reward = 1.0
                else:
                    reward = -0.5
            else:
                if action.action_type == "send_reply":
                    reward = 0.5
                    self.done = True

        elif self.task_id == "refund_policy_medium":
            if action.action_type == "send_reply" and action.message:
                if "refund" in action.message.lower():
                    reward = 1.0
                    self.state["policy_explained"] = True
                    self.done = True
                else:
                    reward = -0.5

        elif self.task_id == "address_change_hard":
            if not self.state["order_checked"]:
                if action.action_type == "lookup_order":
                    self.state["order_checked"] = True
                    reward = 0.5
                else:
                    reward = -0.5

            elif not self.state["address_collected"]:
                if "address" in (action.message or "").lower():
                    self.state["address_collected"] = True
                    reward = 0.5
                else:
                    reward = -0.5

            elif not self.state["address_confirmed"]:
                if "confirm" in (action.message or "").lower():
                    self.state["address_confirmed"] = True
                    reward = 0.5
                    self.done = True
                else:
                    reward = -0.5

        elif self.task_id == "ambiguous_request":
            if not self.state["order_checked"]:
                if action.action_type == "lookup_order":
                    self.state["order_checked"] = True
                    reward = 0.5

            elif not self.state["address_collected"]:
                if "address" in (action.message or "").lower():
                    self.state["address_collected"] = True
                    reward = 0.5

            elif not self.state["resolved"]:
                if any(word in (action.message or "").lower() for word in ["resolved", "replacement", "updated"]):
                    self.state["resolved"] = True
                    reward = 1.0
                    self.done = True

        self.history.append({
            "action": action.__dict__,
            "reward": reward
        })

        return self._get_obs(), reward, self.done, info

    def state(self) -> Observation:
        return self._get_obs()

    def _get_obs(self):
        return Observation(self.task_id, self.history, self.done)
