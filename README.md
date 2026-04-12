# Customer Support Simulator - OpenEnv Environment with True Q-Learning

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/uma0729/customer-support-simulator)
[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-green)](https://github.com/openenv/openenv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A production-ready reinforcement learning environment for training AI agents in customer support scenarios. Built for the OpenEnv ecosystem, this simulator features a **true Q-Learning agent** that learns optimal policies through experience, combining RL-based action selection with LLM-powered message generation.

## 🎯 Why Customer Support?

Customer support automation is a **$50B+ industry** where AI agents are increasingly deployed. However, training these agents requires:
- **Safe sandboxes** that don't impact real customers
- **Granular feedback** beyond binary success/failure  
- **Progressive difficulty** from simple queries to ambiguous requests
- **True reinforcement learning** capabilities for policy optimization

This environment fills that gap by providing:
- **Q-Learning with ε-greedy exploration** (ε decays 0.3 → 0.01)
- **State-aware action selection** (RL chooses action type, LLM generates content)
- **Bellman equation updates** with γ=0.95 discount factor
- **Experience replay** for sample-efficient learning
- **Real-time Q-value tracking** and policy confidence metrics

## 🧠 RL Architecture

### Decision Making Pipeline
1. **State Abstraction**: Simplified state vector (stage + last action) to prevent state explosion
2. **Action Selection**: ε-greedy policy over valid actions only
3. **Message Generation**: LLM generates natural language for `send_reply` actions
4. **Q-Update**: Bellman equation: `Q(s,a) ← Q(s,a) + α[r + γ·max Q(s') - Q(s,a)]`
5. **Experience Replay**: Periodic replay of random trajectories for stable learning

### Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `EPSILON` | 0.3 → 0.01 | Exploration rate (decays 0.995 per step) |
| `GAMMA` | 0.95 | Discount factor for future rewards |
| `LEARNING_RATE` | 0.1 | Q-learning update rate |
| `REPLAY_LEARNING_RATE` | 0.05 | Experience replay update rate |
| `STEP_PENALTY` | -0.05 | Small penalty to encourage efficiency |

## 📋 Environment Overview

### Action Space

Agents interact via a structured action model with exactly **two primitive actions**:

| Field | Type | Description | Required For |
|-------|------|-------------|--------------|
| `action_type` | `"lookup_order" \| "send_reply"` | The type of action to perform | All actions |
| `order_id` | `string` (optional) | 5-digit order identifier | `lookup_order` |
| `message` | `string` (optional) | Text response to customer | `send_reply` |

**Example Actions:**
```json
// Look up an order (RL selects this action)
{"action_type": "lookup_order", "order_id": "12345"}

// Send a reply (RL selects action, LLM generates message)
{"action_type": "send_reply", "message": "Your order has shipped!"}
```
⚠️ Important: Agents must NOT invent new action types (e.g., ask_address, confirm_order). All communication with the customer happens through send_reply.

## Observation Space
| Field | Type | Description |
|-------|------|-------------|
| `task_id` |	`string` |	`Current task identifier` |
| `history` |	`list[dict]` |	`Complete action history for this episode` |
| `done` |	`boolean` |	`Whether episode has terminated` |
| `observation_text` |	`string` |	`Natural language hint about current state` |

## Reward Function
Rewards are dense and shaped to provide learning signal throughout the episode:

- Partial Progress Rewards: +0.3 to +0.8 for completing sub-goals
- Invalid Action Penalties: -0.2 to -0.5 for incorrect sequence
- Repeat Penalties: -0.1 for repeating the same invalid action
- Expert Penalty: -0.2 for using expert hints (training wheels)
  
All rewards are normalized to [-1.0, 1.0] with final scores clipped to [0.0, 1.0].

# 📚 Customer Support Agent Tasks

This environment includes 4 tasks with progressive difficulty levels, designed to evaluate agent decision-making and workflow execution.

---

## 🟢 Task 1: Order Status Query (Easy)

**Description:**  
Customer wants to check their order status.

- **Max Steps:** 5  
- **Expected Reward:** 1.0  

### ✅ Required Actions:
1. `lookup_order` with a valid `order_id`
2. `send_reply` with status information

### 🎯 Success Criteria:
- Both actions are completed in the correct sequence.

---

## 🟡 Task 2: Refund Policy Explanation (Medium)

**Description:**  
Customer wants to know the refund policy.

- **Max Steps:** 5  
- **Expected Reward:** 1.0  

### ✅ Required Actions:
- Single `send_reply` explaining the **30-day refund policy**

### ⚠️ Constraints:
- Must include the keyword **"refund"** in the message
- Should NOT perform unnecessary `lookup_order`

### 🎯 Success Criteria:
- Policy is correctly explained in one response.

---

## 🔴 Task 3: Address Change Request (Hard)

**Description:**  
Customer wants to change their shipping address before delivery.

- **Max Steps:** 8  
- **Expected Reward:** 1.0  

### ✅ Required Sequence:
1. `lookup_order` → Verify order exists  
2. `send_reply` → Ask for new **address** (must include "address")  
3. `send_reply` → Confirm change (must include "confirm")  

### 🎯 Success Criteria:
- All steps executed in the exact sequence with required keywords.

---

## 🟣 Task 4: Moved & Missing Package (Hard+)

**Description:**  
Customer moved to a new location and their package is missing (ambiguous scenario).

- **Max Steps:** 10  
- **Expected Reward:** 1.0  

### ✅ Required Sequence:
1. `lookup_order` → Check package status  
2. `send_reply` → Request address confirmation  
3. `send_reply` → Provide resolution (replacement + address update)  

### 🎯 Success Criteria:
- All steps completed with appropriate context and responses.

---

## 🚀 Goal

These tasks are designed to train and evaluate:
- Decision-making ability  
- Proper tool usage (`lookup_order`, `send_reply`)  
- Understanding of user intent  
- Handling ambiguity in real-world customer support scenarios

## 🚀 Quick Start

### ✅ Prerequisites
- Python 3.10+
- Docker (optional, for containerized deployment)
- API key for LLM provider:
- OpenAI / Groq / Hugging Face

---

## 💻 Local Installation

```bash
# Clone the repository
git clone https://huggingface.co/spaces/uma0729/customer-support-simulator
cd customer-support-simulator

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GROQ_API_KEY="your-api-key-here"  # or OPENAI_API_KEY / HF_TOKEN
export MODEL_NAME="llama-3.1-8b-instant"
export API_BASE_URL="https://api.groq.com/openai/v1"

# Run the web interface
python web_api.py
Python 3.10+
```
Visit http://localhost:7860 to access the interactive training interface.
## Docker Deployment
```bash
# Build the image
docker build -t customer-support-env .

# Run container
docker run -p 7860:7860 \
  -e GROQ_API_KEY="your-key" \
  -e MODEL_NAME="llama-3.1-8b-instant" \
  customer-support-env
```
## Baseline Inference
The environment includes inference.py for reproducible baseline evaluation:
```bash
# Run baseline on all tasks
python inference.py
```
## 📊 Baseline Performance

| Task | Difficulty | Baseline Score | Steps to Complete | Notes |
|------|-----------|---------------|------------------|-------|
| Order Status Query | Easy | 1.0 | 2 | Trivial - lookup then reply |
| Refund Policy | Medium | 0.9 | 1-2 | Sometimes misses "refund" keyword |
| Address Change | Hard | 0.8–0.9 | 3-4 | May repeat actions unnecessarily |
| Moved & Missing | Hard+ | 0.7–0.85 | 4-6 | Challenging for frontier models |

> Baseline tested with **Llama 3.1 8B on Groq**. Scores may vary slightly.

---

## 🧠 RL Learning Performance

With **Q-Learning enabled**, agents typically show the following progression:

### 🔄 Training Phases

- **Initial Exploration Phase**
  - Score: `0.3 – 0.5`
  - Behavior: Random actions due to high epsilon
  - Goal: Explore state-action space

- **Learning Phase**
  - Gradual improvement as Q-values converge
  - Reduced randomness
  - Better task sequencing

- **Converged Policy**
  - Score: `> 0.8`
  - Stable and consistent decisions
  - Epsilon near `0.01`

---

### ⚖️ Exploration vs Exploitation

- After ~50+ episodes:
  - **Exploration:** ~30%
  - **Exploitation:** ~70%

This balance enables efficient learning while maintaining adaptability.

---

## 🧾 Inference Output Format

The system follows strict **OpenEnv logging requirements**:

```text id="log-format-33211"
[START] task=order_status_easy env=CustomerSupport-v1 model=llama-3.1-8b-instant
[STEP] step=1 action={"action_type":"lookup_order","order_id":"12345"} reward=0.80 done=false error=null
[STEP] step=2 action={"action_type":"send_reply","message":"Your order has shipped!"} reward=0.20 done=true error=null
[END] success=true steps=2 score=1.000 rewards=0.80,0.20
```
## 📁 Project Structure
```bash
.
├── app/
│   ├── __init__.py
│   ├── env.py              # Core environment implementation
│   └── models.py           # Pydantic models (Action/Observation/Reward)
├── inference.py            # Baseline inference script
├── web_api.py              # FastAPI server with Q-Learning agent
├── validate_env.py         # OpenEnv compliance validation
├── openenv.yaml            # Environment metadata & task definitions
├── Dockerfile              # Container configuration
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Package configuration
└── README.md              # This file
```
## ✅ OpenEnv Compliance

This environment is fully compliant with the **OpenEnv specification**:

- ✅ Typed **Pydantic models** for Action, Observation, Reward  
- ✅ Standard endpoints: `/reset`, `/step`, `/state`, `/validate`  
- ✅ `openenv.yaml` with complete task metadata  
- ✅ Deterministic, reproducible task graders  
- ✅ Reward values bounded within `[0.0, 1.0]`  

### 🔍 Validate Compliance

```bash
python validate_env.py
```
## 🧪 Testing & Validation
```bash
# Run full validation suite
python validate_env.py

# Test a specific task
python -c "from app.env import CustomerSupportEnv; env = CustomerSupportEnv('order_status_easy'); print(env.reset())"
```
# 🎮 Interactive Training Interface

The web UI (`/`) provides a rich interface for training and debugging agents.

## 🧩 Features

- **Task Selection** → Choose from all 4 difficulty levels
- **Real-time RL Metrics** → Q-values, epsilon decay, exploration stats
- **Action Source Tracking** → Identify if action came from:
  - `q_value`
  - `exploration`
  - `smart_fallback`
- **Learning Status** → Track improvement 📈 or decline 📉
- **Live Score Ring** → Visual progress indicator
- **Step Timeline** → Full action history with Q-confidence
- **Expert Mode** → Get hints with -0.2 penalty
- **Performance Charts** → Reward distribution visualization

## 📊 RL Metrics Display

| Metric | Description |
|--------|-------------|
| **Action Source** | `q_value`, `exploration`, or `smart_fallback` |
| **Q-Confidence** | Maximum Q-value for current state |
| **Epsilon (ε)** | Current exploration rate |
| **Average Q-Value** | Mean Q-value across all states |
| **Exploration vs Exploitation** | Decision distribution |
| **Learning Status** | Improving 📈 or Declining 📉 |

## 🔧 Configuration

### 🌍 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key (preferred) | - |
| `OPENAI_API_KEY` | OpenAI API key (fallback) | - |
| `HF_TOKEN` | Hugging Face token (fallback) | - |
| `MODEL_NAME` | LLM model identifier | `llama-3.1-8b-instant` |
| `API_BASE_URL` | API endpoint URL | `https://api.groq.com/openai/v1` |

### ⚙️ RL Hyperparameters (in `web_api.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `EPSILON` | 0.3 | Initial exploration rate |
| `GAMMA` | 0.95 | Discount factor |
| `LEARNING_RATE` | 0.1 | Q-learning rate |
| `REPLAY_LEARNING_RATE` | 0.05 | Experience replay rate |
| `EPSILON_DECAY` | 0.995 | Decay per step |
| `MIN_EPSILON` | 0.01 | Minimum exploration |
| `STEP_PENALTY` | 0.05 | Efficiency penalty per step |

### 📝 Task Configuration

Modify `MAX_STEPS` and `REWARD_CONFIG` in `web_api.py` to adjust:

- **Maximum steps per task**
- **Reward values for each sub-goal**
- **Expert hint text**

## 📈 Evaluation Metrics

The environment tracks:

| Metric | Description |
|--------|-------------|
| **Score** | Cumulative reward (0.0-1.0) |
| **Discounted Reward** | γ-weighted sum for RL optimization |
| **Steps Taken** | Efficiency metric |
| **Expert Uses** | Reliance on hints |
| **Perfect Completion** | No hints + max score |
| **Q-Table Size** | Number of state-action pairs learned |
| **Policy Confidence** | Max Q-value in current state |
| **Exploration Rate** | Current epsilon value |
| **Learning Trajectory** | Q-value changes over time |

## 🧠 Q-Learning Architecture

### Bellman Equation
Q(s,a) ← Q(s,a) + α [r + γ maxₐ' Q(s',a') - Q(s,a)]

text

Where:
- `α` = Learning rate
- `γ` = Discount factor (0.95)
- `r` = Immediate reward
- `maxₐ' Q(s',a')` = Best future Q-value

### Decision-Making Pipeline

1. **Observe** current state
2. **Epsilon-greedy** exploration vs exploitation:
   - Random action with probability ε
   - Best Q-value action with probability 1-ε
3. **Execute** action, observe reward and next state
4. **Update** Q-table using Bellman equation
5. **Decay** epsilon: ε ← ε × 0.995

### Action Source Tracking

- **q_value** → Action from learned policy (exploitation)
- **exploration** → Random action (trying new strategies)
- **smart_fallback** → LLM-guided action when Q-values are uncertain

## 🚀 Getting Started

1. Set required environment variables
2. Configure hyperparameters in `web_api.py` as needed
3. Run the web server
4. Open `http://localhost:port/` in your browser

## 🤝 Contributing

This environment is part of the OpenEnv ecosystem. To contribute:

1. Fork the repository
2. Create a feature branch
3. Ensure `validate_env.py` passes
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Built for the **OpenEnv Hackathon** (Meta + Hugging Face)
- Uses **FastAPI** for web interface
- **Pydantic** for type safety
- **OpenAI SDK** for LLM integration
- **True Q-Learning** implementation with Bellman updates

## 📞 Support

- **Issues:** GitHub Issues
- **Space:** Hugging Face Space
- **Documentation:** This README

---

**Built with ❤️ for the OpenEnv community | RL + LLM = 🤖💪**

*Complete Q-Learning implementation with real-time metrics, exploration tracking, and Bellman updates*
