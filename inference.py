"""PharmaAgent - inference.py
OpenEnv-compliant inference script.

Mandatory environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    ENV_BASE_URL   Environment server URL.

Stdout format:
    [START] task=<task> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import json
import os
from typing import List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("env", override=False)

# ── Config ─────────────────────────────────────────────────────────────────────
# Matches organiser sample: HF_TOKEN first, then API_KEY
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

BENCHMARK          = "pharma_agent"
TASKS              = ["easy", "medium", "hard"]
MAX_STEPS          = 10
MAX_EPISODE_REWARD = 1.5
SUCCESS_THRESHOLD  = 0.4

# ── Stdout logging ─────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    action_clean = action.replace("\n", " ").replace("\r", "")[:120]
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── HTTP helpers for environment ───────────────────────────────────────────────

def env_reset(task: str) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task": task}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(episode_id: str, action_type: str, value: str) -> dict:
    r = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"action": {"action_type": action_type, "value": value}, "episode_id": episode_id},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are PharmaAgent, an expert AI clinical pharmacist. Your job is to
diagnose a patient's condition and prescribe ONE safe, indicated drug.

EPISODE FLOW — follow this EXACT ORDER:
1. DIAGNOSE once — use 3+ medical keywords for the condition.
   Examples:
   "Type 2 Diabetes Mellitus, hyperglycemia, insulin resistance, glycemic control"
   "Hypertension, high blood pressure, antihypertensive, vascular"
   "Atrial Fibrillation, arrhythmia, anticoagulation, rate control"
   "Hypothyroidism, thyroid hormone deficiency, TSH elevation, levothyroxine"
   "Epilepsy, seizures, anticonvulsant, antiepileptic"
   "Major Depressive Disorder, depression, antidepressant, SSRI, serotonin"
   "Bronchial Asthma, bronchospasm, airway inflammation, beta-2 agonist"
   "Rheumatoid Arthritis, autoimmune, synovitis, DMARD, joint inflammation"
   "Peptic Ulcer Disease, gastric acid, H. pylori, proton pump inhibitor"
   "Chronic Heart Failure, cardiac failure, ejection fraction, diuretic"

2. CHECK_DDI — ONLY if patient has existing medications.
   Check EACH existing med vs your planned drug. Format: "ExistingDrug,PlannedDrug"
   If DDI shows MAJOR or CONTRAINDICATED, pick a DIFFERENT drug and check again.

3. SELECT_DRUG — choose exactly ONE drug from DrugBank matching the diagnosis.
   Use exact DrugBank names. Do NOT select contraindicated drugs.

4. FINALIZE — submit immediately after selecting one drug.

RULES:
- Diagnose EXACTLY ONCE.
- Always check_ddi BEFORE select_drug when existing meds are present.
- Select only ONE drug, then finalize immediately.
- Always respect the valid_options list.

Respond ONLY in JSON (no markdown, no explanation):
{"action_type": "diagnose|select_drug|check_ddi|finalize", "value": "your value"}"""

# ── LLM call — every action goes through the proxy ────────────────────────────

def get_model_action(client: OpenAI, conversation: List[dict], obs: dict) -> tuple:
    """Call LLM for every action decision — required for proxy validation."""
    valid      = obs.get("valid_options", [])
    symptoms   = obs.get("symptoms", [])
    existing   = obs.get("existing_medications", [])
    regimen    = obs.get("current_regimen", [])
    diagnosis  = obs.get("proposed_diagnosis")
    feedback   = obs.get("feedback", "")
    step_count = obs.get("step_count", 0)

    # Clear instruction hint for each phase
    if "diagnose" in valid:
        hint = "DIAGNOSE now. Use 3+ specific clinical keywords for the condition."
    elif "check_ddi" in valid and existing and not regimen:
        hint = (
            f"Patient has existing medications: {', '.join(existing)}. "
            f"You MUST use check_ddi for each one before selecting a drug. "
            f"Format value as: 'ExistingDrug,PlannedDrug'"
        )
    elif "select_drug" in valid and not regimen:
        hint = f"Diagnosis is '{diagnosis}'. SELECT exactly ONE DrugBank drug now."
    elif "finalize" in valid:
        hint = "Drug selected. FINALIZE the regimen now."
    else:
        hint = f"Valid actions: {valid}. Choose the most appropriate next step."

    user_msg = (
        f"PATIENT STATE:\n"
        f"Symptoms: {', '.join(symptoms)}\n"
        f"Existing medications: {', '.join(existing) if existing else 'None'}\n"
        f"Current regimen: {', '.join(regimen) if regimen else 'None'}\n"
        f"Diagnosis so far: {diagnosis or 'Not yet diagnosed'}\n"
        f"Last feedback: {feedback[:400]}\n"
        f"Valid actions RIGHT NOW: {valid}\n"
        f"Steps used: {step_count}/{MAX_STEPS}\n\n"
        f"INSTRUCTION: {hint}\n\n"
        f'Respond ONLY in JSON: {{"action_type": "...", "value": "..."}}'
    )
    conversation.append({"role": "user", "content": user_msg})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=conversation,
            max_tokens=200,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        fallback = valid[0] if valid else "finalize"
        conversation.append({"role": "assistant", "content": f"error: {exc}"})
        return fallback, "finalize", f"error: {exc}"

    conversation.append({"role": "assistant", "content": raw})

    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        action_type = data.get("action_type", valid[0] if valid else "finalize")
        value = data.get("value", "finalize")
        if valid and action_type not in valid:
            action_type = valid[0]
        return action_type, value, raw
    except Exception:
        return valid[0] if valid else "finalize", "finalize", raw

# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(client: OpenAI, task: str) -> float:
    """Run one full episode. Returns normalised score strictly in (0, 1)."""
    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        reset_data = env_reset(task)
        session_id = reset_data.get("session_id", "unknown")
        obs = reset_data.get("observation", reset_data)
        episode_id = obs.get("metadata", {}).get("episode_id") or session_id

        conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step in range(1, MAX_STEPS + 1):
            if obs.get("done", False):
                break

            action_type, value, _ = get_model_action(client, conversation, obs)

            error = None
            try:
                result      = env_step(episode_id, action_type, value)
                step_reward = float(result.get("reward", 0.0))
                done        = result.get("done", False)
                obs         = result.get("observation", {})
            except Exception as e:
                step_reward = 0.0
                done        = True
                error       = str(e)
                obs         = {"done": True}

            rewards.append(step_reward)
            steps_taken = step
            log_step(
                step=step,
                action=f"{action_type}:{value}",
                reward=step_reward,
                done=done,
                error=error,
            )

            if done:
                break

        total_reward = sum(rewards)
        raw_score = total_reward / MAX_EPISODE_REWARD
        # Clamp strictly within (0, 1) — 0.0 and 1.0 are not allowed
        score = round(min(max(raw_score, 0.001), 0.999), 3)
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        rewards = rewards or [0.0]
        score   = 0.001  # never 0.0
        success = False
        log_step(step=steps_taken + 1, action="error", reward=0.0, done=True, error=str(e))

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score

# ── Grader ─────────────────────────────────────────────────────────────────────

def grader(task_scores: dict) -> float:
    weights = {"easy": 0.2, "medium": 0.3, "hard": 0.5}
    final = sum(task_scores.get(t, 0.0) * w for t, w in weights.items())
    # Clamp strictly within (0, 1)
    return round(min(max(final, 0.001), 0.999), 4)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not API_KEY:
        print("API_KEY / HF_TOKEN not set. Must be injected by hackathon or set for local dev.", flush=True)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Verify environment is live
    try:
        r = requests.get(f"{ENV_BASE_URL}/health", timeout=10)
        if r.status_code != 200:
            print(f"Environment returned status {r.status_code}.", flush=True)
            return
    except Exception as e:
        print(f"Cannot reach environment: {e}", flush=True)
        return

    task_scores = {}
    for task in TASKS:
        print(f"\n{'='*50}", flush=True)
        print(f"Running task: {task.upper()}", flush=True)
        score = run_episode(client, task)
        task_scores[task] = score
        print(f"Task '{task}' score: {score:.4f}", flush=True)

    final_score = grader(task_scores)
    print(f"\n{'='*50}", flush=True)
    print(f"FINAL GRADER SCORE: {final_score:.4f} / 1.0000", flush=True)
    print(f"Task scores: {task_scores}", flush=True)

    return final_score


if __name__ == "__main__":
    main()
