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
    # observation_text is good for LLM context, but keep it optional
    observation_text: Optional[str] = None

class Reward(BaseModel):
    value: float
    info: Optional[Dict[str, Any]] = None

class CustomerSupportEnv:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.reset()

    def reset(self):
        """Reset the environment state"""
        self.history = []
        self.done = False
        self.step_count = 0
        self.total_reward = 0.0
        
        # State flags for task progression
        self.flags = {
            "lookup_done": False,
            "address_asked": False,
            "address_confirmed": False,
            "policy_explained": False,
            "resolved": False
        }
        
        # Store last action to detect repetition
        self.last_action_type = None
        self.last_action_valid = True
        
        return self.state()

    def state(self) -> Observation:
        """Return current observation as per OpenEnv spec"""
        obs_text = f"Task: {self.task_id}. Steps taken: {self.step_count}/10."
        
        if self.task_id == "order_status_easy":
            if self.flags["lookup_done"]:
                obs_text += " Order status has been retrieved. You can now reply to the customer."
            else:
                obs_text += " You need to look up the order status first."
                
        elif self.task_id == "refund_policy_medium":
            if self.flags["policy_explained"]:
                obs_text += " Refund policy has been explained. Task complete."
            else:
                obs_text += " The customer wants to know the refund policy. Send a reply explaining it."
                
        elif self.task_id == "address_change_hard":
            if not self.flags["lookup_done"]:
                obs_text += " First, look up the customer's order."
            elif not self.flags["address_asked"]:
                obs_text += " Order found. Now ask the customer for their new address."
            elif not self.flags["address_confirmed"]:
                obs_text += " Address received. Ask the customer to confirm it."
            else:
                obs_text += " Address change completed successfully."
                
        elif self.task_id == "ambiguous_request":
            if not self.flags["lookup_done"]:
                obs_text += " First, check the order status to see what happened with the package."
            elif not self.flags["address_asked"]:
                obs_text += " Order checked. Ask the customer to confirm their new address."
            elif not self.flags["resolved"]:
                obs_text += " Address confirmed. Send a resolution message about the replacement package."
            else:
                obs_text += " Issue resolved successfully."
        
        return Observation(
            task_id=self.task_id,
            history=self.history,
            done=self.done,
            observation_text=obs_text
        )

    def step(self, action: Action):
        """Execute action and update completion status"""
        if self.done:
            return self.state(), 0.0, True, {"msg": "Episode already done"}

        self.step_count += 1
        
        # Evaluate action and get reward
        reward_obj = self.evaluate(action)
        reward_value = reward_obj.value
        
        # Add small penalty for repeating the same invalid action
        if not reward_obj.info.get("is_valid", True) and self.last_action_type == action.action_type:
            reward_value -= 0.1
            reward_obj.info["repeat_penalty"] = -0.1
        
        self.last_action_type = action.action_type
        self.last_action_valid = reward_obj.info.get("is_valid", True)
        
        # Store history using model_dump() for Pydantic v2
        self.history.append(action.model_dump())
        self.total_reward += reward_value
        
        # --- TASK COMPLETION LOGIC (based on flags) ---
        if self.task_id == "order_status_easy" and self.flags["lookup_done"]:
            self.done = True
        elif self.task_id == "refund_policy_medium" and self.flags["policy_explained"]:
            self.done = True
        elif self.task_id == "address_change_hard" and self.flags["address_confirmed"]:
            self.done = True
        elif self.task_id == "ambiguous_request" and self.flags["resolved"]:
            self.done = True
            
        # Safety limit
        if self.step_count >= 10:
            self.done = True

        return self.state(), reward_value, self.done, reward_obj.info

    def evaluate(self, action: Action) -> Reward:
        """State-based reward logic with fuzzy matching for robustness"""
        a_type = action.action_type
        msg = (action.message or "").lower()
        order_id = action.order_id
        
        # ============================================
        # ORDER STATUS EASY TASK
        # ============================================
        if self.task_id == "order_status_easy":
            if a_type == "lookup_order":
                if not self.flags["lookup_done"]:
                    self.flags["lookup_done"] = True
                    # Reward based on whether order_id is provided
                    reward = 0.8 if order_id else 0.5
                    return Reward(value=reward, info={"is_valid": True, "message": "Order lookup successful"})
                else:
                    return Reward(value=-0.1, info={"is_valid": False, "message": "Already looked up order"})
            elif a_type == "send_reply":
                if self.flags["lookup_done"]:
                    return Reward(value=0.2, info={"is_valid": True, "message": "Reply sent to customer"})
                else:
                    return Reward(value=-0.2, info={"is_valid": False, "message": "Lookup order before replying"})
            else:
                return Reward(value=-0.3, info={"is_valid": False, "message": f"Invalid action: {a_type}"})

        # ============================================
        # REFUND POLICY MEDIUM TASK
        # ============================================
        elif self.task_id == "refund_policy_medium":
            if a_type == "send_reply":
                # Fuzzy matching - check for keywords
                keywords = ["refund", "policy", "return", "30-day", "money back", "full refund"]
                if any(kw in msg for kw in keywords):
                    if not self.flags["policy_explained"]:
                        self.flags["policy_explained"] = True
                        return Reward(value=0.9, info={"is_valid": True, "message": "Refund policy explained correctly"})
                    else:
                        return Reward(value=-0.1, info={"is_valid": False, "message": "Policy already explained"})
                else:
                    return Reward(value=0.1, info={"is_valid": False, "message": "Reply missing refund-related keywords"})
            elif a_type == "lookup_order":
                return Reward(value=-0.2, info={"is_valid": False, "message": "Don't lookup order for policy question"})
            else:
                return Reward(value=-0.3, info={"is_valid": False, "message": f"Invalid action: {a_type}"})

        # ============================================
        # ADDRESS CHANGE HARD TASK
        # ============================================
        elif self.task_id == "address_change_hard":
            # Prevent confirming before asking
            if "confirm" in msg and not self.flags["address_asked"]:
                return Reward(value=-0.4, info={"is_valid": False, "message": "Cannot confirm address before asking for it!"})
            
            # Step 1: Lookup order
            if not self.flags["lookup_done"]:
                if a_type == "lookup_order":
                    self.flags["lookup_done"] = True
                    reward = 0.5 if order_id else 0.3
                    return Reward(value=reward, info={"is_valid": True, "message": "Order located"})
                else:
                    return Reward(value=-0.3, info={"is_valid": False, "message": f"Should lookup order first, got {a_type}"})
            
            # Step 2: Ask for address
            elif not self.flags["address_asked"]:
                if a_type == "send_reply" and ("address" in msg or "shipping" in msg):
                    self.flags["address_asked"] = True
                    return Reward(value=0.4, info={"is_valid": True, "message": "Address requested"})
                else:
                    return Reward(value=-0.3, info={"is_valid": False, "message": "Ask for new address (include 'address')"})
            
            # Step 3: Confirm address
            elif not self.flags["address_confirmed"]:
                if a_type == "send_reply" and ("confirm" in msg or "correct" in msg):
                    self.flags["address_confirmed"] = True
                    return Reward(value=0.4, info={"is_valid": True, "message": "Address confirmed"})
                else:
                    return Reward(value=-0.3, info={"is_valid": False, "message": "Ask for confirmation (include 'confirm')"})
            
            else:
                return Reward(value=0.1, info={"is_valid": True, "message": "Task already complete"})

        # ============================================
        # AMBIGUOUS REQUEST TASK
        # ============================================
        elif self.task_id == "ambiguous_request":
            # Prevent confirming before asking
            if "confirm" in msg and not self.flags["address_asked"]:
                return Reward(value=-0.4, info={"is_valid": False, "message": "Cannot confirm address before asking for it!"})
            
            # Step 1: Check order status
            if not self.flags["lookup_done"]:
                if a_type == "lookup_order":
                    self.flags["lookup_done"] = True
                    reward = 0.4 if order_id else 0.3
                    return Reward(value=reward, info={"is_valid": True, "message": "Order status checked"})
                else:
                    return Reward(value=-0.3, info={"is_valid": False, "message": f"Should check order status first, got {a_type}"})
            
            # Step 2: Ask for address confirmation
            elif not self.flags["address_asked"]:
                if a_type == "send_reply" and ("address" in msg or "shipping" in msg):
                    self.flags["address_asked"] = True
                    return Reward(value=0.4, info={"is_valid": True, "message": "Address confirmation requested"})
                else:
                    return Reward(value=-0.3, info={"is_valid": False, "message": "Ask for address confirmation"})
            
            # Step 3: Resolve the issue
            elif not self.flags["resolved"]:
                if a_type == "send_reply":
                    # Check if message addresses both issues
                    has_address = "address" in msg or "shipping" in msg
                    has_resolution = any(kw in msg for kw in ["replacement", "resend", "new package", "ship", "deliver"])
                    
                    if has_address and has_resolution:
                        self.flags["resolved"] = True
                        return Reward(value=0.45, info={"is_valid": True, "message": "Issue fully resolved - address updated and replacement arranged"})
                    elif has_address:
                        return Reward(value=0.2, info={"is_valid": True, "message": "Partial resolution: address confirmed but missing replacement"})
                    elif has_resolution:
                        return Reward(value=0.2, info={"is_valid": True, "message": "Partial resolution: replacement offered but address not confirmed"})
                    else:
                        return Reward(value=0.05, info={"is_valid": False, "message": "Reply lacks resolution details"})
                else:
                    return Reward(value=-0.3, info={"is_valid": False, "message": f"Should send resolution, got {a_type}"})
            
            else:
                return Reward(value=0.1, info={"is_valid": True, "message": "Task already complete"})

        return Reward(value=0.0, info={"is_valid": False, "message": "Unknown task"})