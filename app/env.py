from app.models import Action, Observation, Reward

class CustomerSupportEnv:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.history = []
        self.done = False

    def reset(self):
        self.history = []
        self.done = False
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

        self.done = self.check_done()

        return self.state(), Reward(value=reward), self.done, {}

    # =========================
    # TASK LOGIC
    # =========================

    def evaluate(self, action: Action):
        step = len(self.history)

        if self.task_id == "address_change_hard":

            if step == 0:
                return 0.6 if action.action_type == "lookup_order" else -0.3

            elif step == 1:
                if action.action_type == "send_reply" and "address" in (action.message or "").lower():
                    return 0.2
                return -0.2

            elif step == 2:
                if action.action_type == "send_reply" and "confirm" in (action.message or "").lower():
                    return 0.2
                return -0.2

        elif self.task_id == "order_status_easy":
            return 1.0 if action.action_type == "lookup_order" else -0.2

        elif self.task_id == "refund_policy_medium":
            if action.action_type == "send_reply" and "refund" in (action.message or "").lower():
                return 1.0
            return -0.2

        return 0.0

    def check_done(self):
        if self.task_id == "address_change_hard":
            if len(self.history) < 3:
                return False

            return (
                self.history[0]["action_type"] == "lookup_order"
                and "address" in (self.history[1].get("message") or "").lower()
                and "confirm" in (self.history[2].get("message") or "").lower()
            )

        elif self.task_id == "order_status_easy":
            return any(a["action_type"] == "lookup_order" for a in self.history)

        elif self.task_id == "refund_policy_medium":
            return any("refund" in (a.get("message") or "").lower() for a in self.history)

        return False