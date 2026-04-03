from pydantic import BaseModel
from typing import List, Dict, Any

class Task(BaseModel):
    id: str
    difficulty: str
    description: str
    initial_customer_message: str
    target_data: Dict[str, Any] # What the agent needs to find or do

TASKS = [
    Task(
        id="order_status_easy",
        difficulty="easy",
        description="Customer asks for the status of a single order.",
        initial_customer_message="Hi, can you tell me the status of order ORD-552?",
        target_data={"order_id": "ORD-552", "expected_status": "Shipped"}
    ),
    Task(
        id="refund_policy_medium",
        difficulty="medium",
        description="Customer wants a refund. Agent must check tier and order date.",
        initial_customer_message="I want a refund for ORD-990. It arrived broken.",
        target_data={"order_id": "ORD-990", "must_mention": "refund processed"}
    ),
    Task(
        id="address_change_hard",
        difficulty="hard",
        description="Customer wants to change address for an order that is already 'Out for Delivery'.",
        initial_customer_message="I moved! Can you change the address for ORD-111 to 742 Evergreen Terrace?",
        target_data={"order_id": "ORD-111", "expected_outcome": "denied"}
    )
]

def grade_task(task_id: str, history: List[Dict[str, str]], tool_usage: List[str]) -> float:
    score = 0.0
    # Convert history to a single string for easier checking
    transcript = " ".join([m['content'].lower() for m in history])
    
    if task_id == "order_status_easy":
        # Did they use the lookup tool?
        if "lookup_order" in tool_usage: score += 0.5
        # Did they tell the customer it is Shipped?
        if "shipped" in transcript: score += 0.5
            
    elif task_id == "refund_policy_medium":
        if "lookup_order" in tool_usage: score += 0.3
        if "refund" in transcript: score += 0.7
            
    elif task_id == "address_change_hard":
        # The agent must realize it's too late to change (Out for Delivery)
        if "cannot" in transcript or "too late" in transcript or "sorry" in transcript:
            score = 1.0
            
    return min(score, 1.0)