# OpenEnv: AI Customer Support Simulator

## Overview
A high-fidelity environment simulating a Customer Support Agent's workspace. Unlike "toy" environments, this requires the agent to:
1. **Query Databases**: Use `lookup_order` to find real-time shipment data.
2. **Apply Policy**: Navigate complex refund rules and delivery constraints.
3. **Multi-turn Dialogue**: Maintain context across several steps of a conversation.

## Tasks
- **Easy**: Track a single order status.
- **Medium**: Process a refund for a damaged item (requires verifying order state).
- **Hard**: Handle a delivery address change request for a package already "Out for Delivery" (requires denying the request per policy).

## Action Space
- `action_type`: "lookup_order", "lookup_customer", or "send_reply".
- `query`: The specific ID or search term.
- `message`: The text sent to the human customer.

## Deployment
This environment is containerized and ready for Hugging Face Spaces.