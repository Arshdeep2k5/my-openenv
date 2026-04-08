# pharma_agent_environment.py
"""
PharmaAgent Clinical Decision RL Environment.

Three tasks of increasing difficulty:
  easy   - No existing medications. Diagnose + select 1 indicated drug + finalize.
  medium - Patient has existing medications. Must also perform DDI check.
  hard   - Patient has existing medications with a known contraindicated interaction.
           Agent must identify and avoid the dangerous drug AND check DDI.

Reward structure (max ~1.5 per episode):
  diagnose        : +0.30 (multi-keyword match) | +0.15 (partial)
  select_drug     : +0.20 (indicated) | +0.05 (in DB, wrong indication) | -0.25 (contraindicated)
  check_ddi       : +0.30 (critical flagged pair) | +0.15 (flagged or found interaction) | +0.05 (no interaction)
  finalize        : +0.10 (has indicated drug) + 0.10 (no contraindicated) + 0.05 (diagnosis set)
                    + 0.05 (DDI checks done) - 0.10 (no DDI when existing meds present)
                    - 0.20 per contraindicated drug selected
"""

import os
import sys
import random
import re
import sqlite3
import threading
from uuid import uuid4

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in [_HERE, _ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from models import PharmaAgentAction, PharmaAgentObservation

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH = os.environ.get("DB_PATH", os.path.join(_HERE, "drugbank_lite.db"))
MAX_EPISODE_REWARD = 1.5
MAX_REWARDED_DDI_CHECKS = 3
MAX_STEPS = 10

# ── Severity patterns ─────────────────────────────────────────────────────────
_SEVERITY_PATTERNS = [
    ("contraindicated", re.compile(r"\bcontraindicated\b", re.I)),
    ("major", re.compile(r"\b(major|serious|severe|life.?threatening|fatal)\b", re.I)),
    ("moderate", re.compile(r"\b(moderate|caution|monitor)\b", re.I)),
    ("minor", re.compile(r"\b(minor|mild|small)\b", re.I)),
]

# ── Condition seeds ───────────────────────────────────────────────────────────
_CONDITION_SEEDS = [
    {
        "condition": "Hypertension",
        "symptoms": ["persistent headache", "elevated blood pressure", "dizziness", "blurred vision"],
        "indication_keywords": ["hypertension", "high blood pressure", "antihypertensive"],
        "diagnosis_keywords": ["hypertension", "blood pressure", "antihypertensive", "vascular"],
    },
    {
        "condition": "Type 2 Diabetes Mellitus",
        "symptoms": ["excessive thirst", "frequent urination", "fatigue", "blurred vision"],
        "indication_keywords": ["type 2 diabetes", "diabetes mellitus", "hyperglycemia", "glycemic"],
        "diagnosis_keywords": ["diabetes", "hyperglycemia", "insulin resistance", "glycemic", "glucose"],
    },
    {
        "condition": "Chronic Heart Failure",
        "symptoms": ["shortness of breath on exertion", "ankle swelling", "fatigue", "orthopnoea"],
        "indication_keywords": ["heart failure", "cardiac failure", "congestive heart"],
        "diagnosis_keywords": ["heart failure", "cardiac", "ejection fraction", "congestive"],
    },
    {
        "condition": "Rheumatoid Arthritis",
        "symptoms": ["symmetric joint pain", "morning stiffness", "joint swelling", "fatigue"],
        "indication_keywords": ["rheumatoid arthritis", "rheumatoid", "autoimmune arthritis"],
        "diagnosis_keywords": ["rheumatoid", "arthritis", "autoimmune", "synovitis", "joint inflammation"],
    },
    {
        "condition": "Bronchial Asthma",
        "symptoms": ["recurrent wheeze", "chest tightness", "shortness of breath", "nocturnal cough"],
        "indication_keywords": ["asthma", "bronchial asthma", "bronchospasm", "airway obstruction"],
        "diagnosis_keywords": ["asthma", "bronchospasm", "airway", "bronchial", "respiratory"],
    },
    {
        "condition": "Epilepsy",
        "symptoms": ["recurrent seizures", "transient loss of consciousness", "post-ictal confusion"],
        "indication_keywords": ["epilepsy", "seizure", "anticonvulsant", "antiepileptic"],
        "diagnosis_keywords": ["epilepsy", "seizure", "anticonvulsant", "antiepileptic", "ictal"],
    },
    {
        "condition": "Hypothyroidism",
        "symptoms": ["fatigue", "weight gain", "cold intolerance", "constipation", "dry skin"],
        "indication_keywords": ["hypothyroidism", "thyroid deficiency", "levothyroxine"],
        "diagnosis_keywords": ["hypothyroidism", "thyroid", "TSH", "levothyroxine", "thyroid hormone"],
    },
    {
        "condition": "Major Depressive Disorder",
        "symptoms": ["persistent low mood", "anhedonia", "insomnia", "fatigue", "poor concentration"],
        "indication_keywords": ["depression", "major depressive", "antidepressant"],
        "diagnosis_keywords": ["depression", "depressive", "antidepressant", "mood", "serotonin"],
    },
    {
        "condition": "Peptic Ulcer Disease",
        "symptoms": ["epigastric pain", "nausea", "bloating", "pain relieved by food"],
        "indication_keywords": ["peptic ulcer", "gastric ulcer", "duodenal ulcer", "H. pylori"],
        "diagnosis_keywords": ["peptic ulcer", "gastric", "H. pylori", "proton pump", "acid"],
    },
    {
        "condition": "Atrial Fibrillation",
        "symptoms": ["palpitations", "irregular heartbeat", "dyspnoea on exertion", "fatigue"],
        "indication_keywords": ["atrial fibrillation", "AF", "anticoagulation", "rate control"],
        "diagnosis_keywords": ["atrial fibrillation", "arrhythmia", "anticoagul", "rate control", "AF"],
    },
]

# ── Module-level session store ────────────────────────────────────────────────
_SESSION_LOCK = threading.Lock()
_SESSIONS: dict = {}


def _store_session(eid: str, ep: dict) -> None:
    with _SESSION_LOCK:
        _SESSIONS[eid] = ep


def _load_session(eid: str) -> dict | None:
    with _SESSION_LOCK:
        return _SESSIONS.get(eid)


def _clear_session(eid: str) -> None:
    with _SESSION_LOCK:
        _SESSIONS.pop(eid, None)


# ── Database helpers ──────────────────────────────────────────────────────────
def _get_db():
    if not os.path.exists(DB_PATH) or os.path.getsize(DB_PATH) == 0:
        return None
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def db_get_drug(name: str) -> dict | None:
    conn = _get_db()
    if not conn:
        return None
    try:
        row = conn.execute(
            "SELECT name, indication, status, type FROM drugs WHERE LOWER(name)=LOWER(?)",
            (name,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def db_check_interaction(d1: str, d2: str) -> dict | None:
    conn = _get_db()
    if not conn:
        return None
    try:
        row = conn.execute(
            "SELECT description FROM interactions "
            "WHERE (LOWER(drug1_name)=LOWER(?) AND LOWER(drug2_name)=LOWER(?)) "
            "   OR (LOWER(drug1_name)=LOWER(?) AND LOWER(drug2_name)=LOWER(?)) "
            "LIMIT 1",
            (d1, d2, d2, d1),
        ).fetchone()
        if not row:
            return None
        desc = row["description"] or ""
        sev = "unknown"
        for label, pat in _SEVERITY_PATTERNS:
            if pat.search(desc):
                sev = label
                break
        return {"description": desc, "severity_label": sev}
    finally:
        conn.close()


def db_get_interactions_for_drug(name: str, limit: int = 10) -> list:
    conn = _get_db()
    if not conn:
        return []
    try:
        rows = conn.execute(
            "SELECT drug2_name AS partner, description FROM interactions "
            "WHERE LOWER(drug1_name)=LOWER(?) AND description!='' LIMIT ?",
            (name, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def db_drugs_for_indication(keywords: list, limit: int = 40) -> list:
    conn = _get_db()
    if not conn:
        return []
    try:
        results, seen = [], set()
        for kw in keywords:
            rows = conn.execute(
                "SELECT name, indication, status FROM drugs "
                "WHERE LOWER(indication) LIKE LOWER(?) "
                "  AND status LIKE '%approved%' "
                "  AND type='small molecule' "
                "  AND indication!='' LIMIT ?",
                (f"%{kw}%", limit),
            ).fetchall()
            for r in rows:
                if r["name"] not in seen:
                    seen.add(r["name"])
                    results.append(dict(r))
        return results
    finally:
        conn.close()


def _parse_sev(desc: str) -> str:
    for label, pat in _SEVERITY_PATTERNS:
        if pat.search(desc):
            return label
    return "unknown"


# ── Case generation ───────────────────────────────────────────────────────────
def generate_case(task: str) -> dict:
    seed = random.choice(_CONDITION_SEEDS)
    indicated = db_drugs_for_indication(seed["indication_keywords"], limit=40)

    if not indicated:
        return {
            "id": f"fallback_{task}",
            "task": task,
            "condition": seed["condition"],
            "symptoms": seed["symptoms"],
            "existing_medications": [],
            "indication_keywords": seed["indication_keywords"],
            "diagnosis_keywords": seed["diagnosis_keywords"],
            "indicated_drug_names": [],
            "avoid_drugs": [],
            "existing_med_interactions": {},
        }

    ex_name = random.choice(indicated)["name"]
    raw_ix = db_get_interactions_for_drug(ex_name, limit=20)
    avoid, ex_med_ix = [], {}

    for ix in raw_ix:
        p, desc = ix["partner"], ix["description"] or ""
        sev = _parse_sev(desc)
        ex_med_ix[p] = {"description": desc, "severity": sev}
        if sev in ("contraindicated", "major"):
            avoid.append(p)

    if task == "easy":
        existing, avoid, ex_med_ix = [], [], {}
    elif task == "medium":
        existing = [ex_name]
    else:  # hard
        existing = [ex_name]
        if not avoid:
            avoid = [
                p for p, v in ex_med_ix.items()
                if v["severity"] in ("contraindicated", "major", "moderate")
            ][:3]

    return {
        "id": f"{task}_{seed['condition'].replace(' ', '_')}_{random.randint(1000, 9999)}",
        "task": task,
        "condition": seed["condition"],
        "symptoms": seed["symptoms"],
        "existing_medications": existing,
        "indication_keywords": seed["indication_keywords"],
        "diagnosis_keywords": seed["diagnosis_keywords"],
        "indicated_drug_names": [d["name"] for d in indicated],
        "avoid_drugs": avoid,
        "existing_med_interactions": ex_med_ix,
    }


def _fresh_ep(task: str) -> dict:
    return {
        "case": generate_case(task),
        "task": task,
        "proposed_diagnosis": None,
        "selected_drugs": [],
        "checked_interactions": [],
        "cumulative_reward": 0.0,
        "phase": "triage",
        "done": False,
        "step_count": 0,
    }


# ── Scoring functions ─────────────────────────────────────────────────────────
def score_diagnosis(proposed: str, case: dict) -> tuple[float, str]:
    pl = proposed.lower()
    kws = case.get("diagnosis_keywords", [])
    m = [k for k in kws if k.lower() in pl]
    if len(m) >= 2:
        return 0.30, f"Diagnosis supported — matched: {', '.join(m[:3])}."
    if len(m) == 1:
        return 0.15, f"Partial diagnosis — matched '{m[0]}'."
    return 0.00, "Diagnosis does not align with the presenting symptoms."


def score_drug(drug: str, case: dict) -> tuple[float, str]:
    dl = drug.lower()
    avoid = [d.lower() for d in case.get("avoid_drugs", [])]
    if dl in avoid:
        ex = case.get("existing_medications", ["existing medication"])
        return -0.25, f"SAFETY: {drug} has a contraindicated/major interaction with {', '.join(ex)} per DrugBank."
    rec = db_get_drug(drug)
    if rec is None:
        return 0.00, f"'{drug}' not found in DrugBank."
    ind = (rec.get("indication") or "").lower()
    kws = case.get("indication_keywords", [])
    if any(k.lower() in ind for k in kws) or drug in case.get("indicated_drug_names", []):
        return 0.20, f"{drug} is indicated for this condition per DrugBank."
    return 0.05, f"{drug} exists in DrugBank but indication does not clearly match."


def score_ddi(d1: str, d2: str, case: dict, checked: list) -> tuple[float, str]:
    if d1.lower() == d2.lower():
        return 0.0, "Both drugs are the same."
    already = {frozenset([i["drug1"].lower(), i["drug2"].lower()]) for i in checked}
    pair = frozenset([d1.lower(), d2.lower()])
    if pair in already:
        return 0.0, f"{d1} x {d2} already checked."
    if len(already) >= MAX_REWARDED_DDI_CHECKS:
        return 0.0, "DDI reward cap reached."
    avoid = [d.lower() for d in case.get("avoid_drugs", [])]
    flagged = d1.lower() in avoid or d2.lower() in avoid
    res = db_check_interaction(d1, d2)
    if flagged and res:
        sev = res["severity_label"].upper()
        return 0.30, f"CRITICAL DDI [{sev}]: {d1} x {d2}. {res['description'][:200]}"
    if flagged:
        return 0.15, f"{d1} or {d2} flagged as dangerous with existing medications."
    if res:
        sev = res["severity_label"].upper()
        return 0.15, f"Interaction [{sev}]: {d1} x {d2}. {res['description'][:150]}"
    return 0.05, f"No interaction found between {d1} and {d2}."


def score_finalize(ep: dict) -> tuple[float, str]:
    sl = [d.lower() for d in ep["selected_drugs"]]
    ind = [d.lower() for d in ep["case"].get("indicated_drug_names", [])]
    avoid = [d.lower() for d in ep["case"].get("avoid_drugs", [])]
    ex = ep["case"].get("existing_medications", [])
    hits = [d for d in sl if d in ind]
    bad = [d for d in sl if d in avoid]
    r, parts = 0.0, []

    if hits:
        r += 0.10
        parts.append(f"Indicated: {', '.join(d.title() for d in hits)}")
    else:
        parts.append("No indicated drugs in regimen.")

    if not bad:
        r += 0.10
        parts.append("No contraindicated drugs.")
    else:
        pen = 0.20 * len(bad)
        r -= pen
        parts.append(f"Contraindicated present: -{pen:.2f}")

    if ep["proposed_diagnosis"]:
        r += 0.05
        parts.append("Diagnosis established.")

    if ep["checked_interactions"]:
        r += 0.05
        parts.append(f"{len(ep['checked_interactions'])} DDI check(s).")
    elif ex and ep["selected_drugs"]:
        r -= 0.10
        parts.append("Safety penalty: no DDI checks with existing medications.")

    return round(r, 3), "Final Evaluation\n" + "\n".join(f"  {p}" for p in parts)


# ── Environment class ─────────────────────────────────────────────────────────
class PharmaAgentEnvironment(Environment):
    """
    PharmaAgent Clinical Decision RL Environment.

    Simulates clinical pharmacist decision-making:
    - Diagnose the patient's condition from symptoms
    - Select appropriate drugs (checked against DrugBank)
    - Check drug-drug interactions (DDI) when patient has existing meds
    - Finalize the treatment regimen

    Three tasks of increasing difficulty:
      easy   - No existing medications. Common condition.
      medium - Patient has existing medications. DDI check required.
      hard   - Existing meds with a contraindicated interaction to catch.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    TASKS = ["easy", "medium", "hard"]

    def __init__(self, task: str = "easy"):
        self._task = task if task in self.TASKS else "easy"
        self._episode_id: str = str(uuid4())
        self._state = State(episode_id=self._episode_id, step_count=0)

    def reset(self, task: str = None, **kwargs) -> PharmaAgentObservation:
        """Reset the environment and generate a new patient case."""
        if task and task in self.TASKS:
            self._task = task
        self._episode_id = str(uuid4())
        self._state = State(episode_id=self._episode_id, step_count=0)
        ep = _fresh_ep(self._task)
        _store_session(self._episode_id, ep)
        case = ep["case"]
        existing_str = ", ".join(case["existing_medications"]) or "None"
        feedback = (
            f"New patient [{self._task.upper()}].\n"
            f"Symptoms: {', '.join(case['symptoms'])}\n"
            f"Existing meds: {existing_str}\n\n"
            f"Start with action_type='diagnose'."
        )
        return PharmaAgentObservation(
            task=self._task,
            phase="triage",
            symptoms=case["symptoms"],
            existing_medications=case["existing_medications"],
            current_regimen=[],
            proposed_diagnosis=None,
            feedback=feedback,
            valid_options=["diagnose"],
            reward_so_far=0.0,
            step_count=0,
            done=False,
            reward=0.0,
            metadata={"episode_id": self._episode_id, "task": self._task},
        )

    def step(self, action: PharmaAgentAction, episode_id: str = None, **kwargs) -> PharmaAgentObservation:
        """Execute one action and return the next observation."""
        eid = episode_id if episode_id else self._episode_id
        ep = _load_session(eid)

        if ep is None:
            ep = _fresh_ep(self._task)
            _store_session(eid, ep)

        ep["step_count"] = ep.get("step_count", 0) + 1
        case = ep["case"]
        step_r = 0.0
        feedback = ""
        done = False

        if ep["done"]:
            return PharmaAgentObservation(
                task=self._task,
                phase="done",
                symptoms=case["symptoms"],
                existing_medications=case["existing_medications"],
                current_regimen=ep["selected_drugs"],
                proposed_diagnosis=ep["proposed_diagnosis"],
                feedback="Episode complete. Call reset() to start a new episode.",
                valid_options=[],
                reward_so_far=round(ep["cumulative_reward"], 3),
                step_count=ep["step_count"],
                done=True,
                reward=0.0,
                metadata={"episode_id": eid},
            )

        atype = action.action_type.strip().lower()
        value = action.value.strip()

        if atype == "diagnose":
            step_r, feedback = score_diagnosis(value, case)
            ep["proposed_diagnosis"] = value
            ep["cumulative_reward"] += step_r
            ep["phase"] = "selection"
            feedback += "\n\nNext: select_drug, check_ddi, or finalize."

        elif atype == "select_drug":
            step_r, feedback = score_drug(value, case)
            avoid = [d.lower() for d in case.get("avoid_drugs", [])]
            if value.lower() not in avoid and value not in ep["selected_drugs"]:
                ep["selected_drugs"].append(value)
            ep["cumulative_reward"] += step_r
            ep["phase"] = "safety"
            feedback += f"\n\nRegimen: {', '.join(ep['selected_drugs']) or 'None'}"

        elif atype == "check_ddi":
            pts = value.replace(" vs ", ",").replace(" and ", ",").split(",")
            if len(pts) >= 2:
                d1, d2 = pts[0].strip(), pts[1].strip()
                step_r, feedback = score_ddi(d1, d2, case, ep["checked_interactions"])
                ep["checked_interactions"].append({"drug1": d1, "drug2": d2, "reward": step_r})
                ep["cumulative_reward"] += step_r
                ep["phase"] = "safety"
                feedback += f"\n\nRegimen: {', '.join(ep['selected_drugs']) or 'None'}"
            else:
                feedback = "Provide two drugs separated by comma: Drug1,Drug2"

        elif atype == "finalize":
            step_r, feedback = score_finalize(ep)
            ep["cumulative_reward"] += step_r
            done = True
            ep["done"] = True
            ep["phase"] = "done"
            feedback += f"\n\nTotal reward: {round(ep['cumulative_reward'], 3)} / {MAX_EPISODE_REWARD}"
            _clear_session(eid)

        else:
            feedback = (
                f"Unknown action_type '{atype}'. "
                "Valid: diagnose, select_drug, check_ddi, finalize."
            )

        if ep["step_count"] >= MAX_STEPS and not done:
            done = True
            ep["done"] = True
            ep["phase"] = "done"
            feedback += "\n\nStep limit reached. Episode ended."
            _clear_session(eid)

        if not done:
            _store_session(eid, ep)

        valid_opts: list[str]
        if done:
            valid_opts = []
        elif ep["phase"] == "triage":
            valid_opts = ["diagnose"]
        else:
            valid_opts = ["select_drug", "check_ddi", "finalize"]

        return PharmaAgentObservation(
            task=self._task,
            phase=ep["phase"],
            symptoms=case["symptoms"],
            existing_medications=case["existing_medications"],
            current_regimen=ep["selected_drugs"],
            proposed_diagnosis=ep["proposed_diagnosis"],
            feedback=feedback,
            valid_options=valid_opts,
            reward_so_far=round(ep["cumulative_reward"], 3),
            step_count=ep["step_count"],
            done=done,
            reward=round(step_r, 3),
            metadata={"episode_id": eid, "task": self._task},
        )

    @property
    def state(self) -> State:
        return self._state
