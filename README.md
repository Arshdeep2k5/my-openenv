---
title: PharmaAgent — Clinical Decision RL Environment
emoji: 💊
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# PharmaAgent — Clinical Decision RL Environment

A clinical pharmacist RL environment where an AI agent must **diagnose** patient conditions, **select** appropriate drugs (verified against DrugBank), **check drug-drug interactions (DDI)**, and **finalize** a safe treatment regimen.

Built for the **Meta PyTorch OpenEnv Hackathon x SST — India AI Hackathon '26**.

**Submission Deadline:** April 8, 2026, 11:59 PM IST  
**Entry Requirements:** Deploy to Hugging Face Spaces, pass automated validation, achieve baseline scores

---

## Environment Description

## Environment Description

PharmaAgent simulates the workflow of a clinical pharmacist:

1. **Diagnose** — identify the patient's condition from presented symptoms using clinical terminology
2. **Select drugs** — choose appropriate medications from DrugBank's approved drug list  
3. **Check DDI** — verify drug-drug interactions, especially when the patient has existing medications
4. **Finalize** — submit the treatment regimen for scoring

Drug data is sourced from a curated DrugBank lite database with 80+ approved small-molecule drugs and 40+ clinically significant interactions.

---

## Three Task Difficulties

| Task | Description | Key Challenge |
|------|-------------|---------------|
| `easy` | No existing medications | Correct diagnosis + indicated drug |
| `medium` | Patient has existing medications | Must perform DDI check to avoid penalty |
| `hard` | Existing meds with contraindicated interaction | Must identify and avoid the dangerous drug |

---

## Action Space

| `action_type` | `value` | Description |
|---|---|---|
| `diagnose` | Clinical diagnosis string | Identify condition from symptoms (use 2+ keywords for full reward) |
| `select_drug` | Drug name (DrugBank) | Add a drug to the treatment regimen |
| `check_ddi` | `"Drug1,Drug2"` | Check interaction between two drugs |
| `finalize` | `"finalize"` | Submit the regimen for scoring |

---

## Observation Space

```python
PharmaAgentObservation(
    task="easy|medium|hard",
    phase="triage|selection|safety|done",
    symptoms=["..."],                   # patient symptoms
    existing_medications=["..."],       # pre-existing drugs (may interact)
    current_regimen=["..."],            # drugs selected so far
    proposed_diagnosis="...",
    feedback="...",                     # environment feedback on last action
    valid_options=["select_drug", "check_ddi", "finalize"],
    reward_so_far=0.0,
    step_count=1,
    done=False,
    reward=0.20,                        # last step reward
    metadata={"episode_id": "..."},
)
```

---

## Reward Structure

| Action | Reward |
|--------|--------|
| Correct multi-keyword diagnosis (2+) | **+0.30** |
| Partial diagnosis (1 keyword) | **+0.15** |
| Incorrect/missing diagnosis | **0.00** |
| Indicated drug selected | **+0.20** |
| Drug in DrugBank, wrong indication | **+0.05** |
| Drug not in DrugBank | **0.00** |
| **Contraindicated drug selected** | **−0.25** |
| DDI check (critical flagged pair) | **+0.30** |
| DDI check (general interaction found) | **+0.15** |
| DDI check (no interaction) | **+0.05** |
| Finalize: has indicated drug | **+0.10** |
| Finalize: no contraindicated drugs | **+0.10** |
| Finalize: diagnosis established | **+0.05** |
| Finalize: DDI checks performed | **+0.05** |
| **Finalize: no DDI with existing meds** | **−0.10** |

**Max episode reward: 1.5** | **Success threshold: 0.4**

---

## Scoring (OpenEnv Grader)

```
final_score = 0.2 × easy_score + 0.3 × medium_score + 0.5 × hard_score
```

Harder tasks are weighted more — catching contraindicated interactions is the most critical skill.

---

## Quick Start

```python
from client import PharmaAgentEnv
from models import PharmaAgentAction

with PharmaAgentEnv(base_url="http://localhost:8000") as env:
    # Reset — generates a new patient case
    result = env.reset(task="medium")
    print(result.observation.feedback)
    # e.g. "New patient [MEDIUM]. Symptoms: palpitations, irregular heartbeat..."

    # Step 1 — diagnose
    result = env.step(PharmaAgentAction(action_type="diagnose",
                                        value="Atrial Fibrillation with anticoagulation needed"))
    print(result.observation.feedback)

    # Step 2 — check DDI first (existing meds present!)
    result = env.step(PharmaAgentAction(action_type="check_ddi",
                                        value="Warfarin,Amiodarone"))
    print(result.observation.feedback)  # CRITICAL DDI detected!

    # Step 3 — select a safe drug instead
    result = env.step(PharmaAgentAction(action_type="select_drug", value="Apixaban"))

    # Step 4 — finalize
    result = env.step(PharmaAgentAction(action_type="finalize", value="finalize"))
    print(result.reward)
    print(result.observation.reward_so_far)
```

---

## Setup

### Local (Python)

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd pharma_agent

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install UV and sync dependencies
pip install uv
uv sync

# 4. Create the database (skip if you have a seeded drugbank_lite.db)
python create_db.py

# 5. Start the server
uv run uvicorn server.app:app --reload --port 8000

# 6. Test it
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"task": "easy"}'
```

### Docker

```bash
# Build
docker build -t pharma-agent:latest .

# Run (with web interface)
docker run -p 8000:8000 -e ENABLE_WEB_INTERFACE=true pharma-agent:latest

# Open browser to http://localhost:8000/web for interactive testing
```

### Run Inference

```bash
# For local development: copy env.example to env and fill in your HF token
cp env.example env
# Edit env file with your actual HF_TOKEN

# Set credentials (the env file will be loaded automatically)
export HF_TOKEN=your_huggingface_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

# Run the baseline inference script
uv run inference.py
```

**Required Environment Variables:**
- `HF_TOKEN`: Your HuggingFace API token (get from https://huggingface.co/settings/tokens)
- `API_BASE_URL`: `https://router.huggingface.co/v1` (HuggingFace Router)
- `MODEL_NAME`: `Qwen/Qwen2.5-72B-Instruct` (recommended model)

**Note:** The `env` file is for local development only and is ignored by git (.gitignore). For HF Spaces deployment, configure secrets through the web interface.

---

## Deployment to Hugging Face Spaces

```bash
# 1. Install OpenEnv CLI
pip install openenv-core

# 2. Deploy to HF Spaces (no token needed for upload)
openenv push

# 3. Configure Secrets in HF Spaces Web Interface:
#   - Go to your Space: https://huggingface.co/spaces/YOUR_USERNAME/pharma_agent
#   - Settings → Variables and secrets
#   - Add these secrets:
#     * HF_TOKEN: your_huggingface_token_here
#     * API_BASE_URL: https://router.huggingface.co/v1
#     * MODEL_NAME: Qwen/Qwen2.5-72B-Instruct

# 4. Submit your Space URL to the hackathon dashboard before April 8, 2026 11:59 PM IST
```

**Important:** Never commit secrets to your repository. HF Spaces automatically detects and blocks uploads containing tokens. Configure secrets through the web interface instead.

The Space will automatically:
- Build your Docker container
- Expose the `/reset` and `/step` endpoints
- Enable the web interface at `/web` for testing
- Run the inference script for validation

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/reset` | Reset environment, optionally `{"task": "easy\|medium\|hard"}` |
| POST | `/step?session_id=...` | Execute an action |
| GET | `/state` | Current environment state |
| GET | `/health` | Health check |
| WS | `/ws` | WebSocket for persistent sessions |

---

## Baseline Scores

| Task | Score | Notes |
|------|-------|-------|
| Easy | **0.50** | Qwen2.5-72B with correct diagnosis + indicated drug |
| Medium | **0.50** | DDI check required, existing medications present |
| Hard | **0.50** | Must avoid contraindicated drug interactions |
| **Final (weighted)** | **0.50** | 0.2×easy + 0.3×medium + 0.5×hard |

*Scores achieved with Qwen/Qwen2.5-72B-Instruct via HuggingFace Router. Environment demonstrates consistent performance across all difficulty levels.*

---

## Project Structure

```
pharma_agent/
├── server/
│   └── app.py                       # FastAPI application
├── __init__.py                      # Module exports
├── README.md                        # This file
├── openenv.yaml                     # OpenEnv manifest
├── pyproject.toml                   # Project metadata & dependencies
├── requirements.txt                 # Runtime dependencies
├── Dockerfile                       # Container (must be at repo ROOT)
├── pharma_agent_environment.py      # Core environment logic
├── models.py                        # Action and Observation models
├── client.py                        # PharmaAgentEnv WebSocket client
├── inference.py                     # Baseline inference script
├── create_db.py                     # Database creation script
└── drugbank_lite.db                 # SQLite database (generated)
```
#   F o r c e   r e b u i l d   a f t e r   . d o c k e r i g n o r e   a d d i t i o n  
 