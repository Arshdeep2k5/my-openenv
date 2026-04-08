# models.py — PharmaAgent OpenEnv Environment
"""
Data models for the PharmaAgent Clinical Decision Environment.

Three tasks of increasing difficulty:
  - easy:   No existing medications. Diagnose + select 1 indicated drug + finalize.
  - medium: Patient has existing medications. Must also perform DDI check.
  - hard:   Patient has existing medications with a known contraindicated interaction.
            Agent must identify and avoid the dangerous drug AND check DDI.
"""

from typing import Optional, List
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class PharmaAgentAction(Action):
    """
    Action for the PharmaAgent environment.

    action_type options:
      - "diagnose"    : propose a diagnosis from symptoms
      - "select_drug" : add a drug to the treatment regimen
      - "check_ddi"   : check interaction between two drugs (format: "Drug1,Drug2")
      - "finalize"    : submit the final regimen for scoring
    """

    action_type: str = Field(
        ...,
        description="One of: diagnose, select_drug, check_ddi, finalize",
    )
    value: str = Field(
        ...,
        description=(
            "The value for the chosen action "
            "(diagnosis text, drug name, drug pair like 'Drug1,Drug2', or 'finalize')"
        ),
    )


class PharmaAgentObservation(Observation):
    """
    Observation returned after each step in the PharmaAgent environment.
    """

    task: str = Field(
        default="easy",
        description="Current task difficulty: easy | medium | hard",
    )
    phase: str = Field(
        default="triage",
        description="Current phase: triage | selection | safety | done",
    )
    symptoms: List[str] = Field(
        default_factory=list,
        description="Patient symptoms",
    )
    existing_medications: List[str] = Field(
        default_factory=list,
        description="Patient's existing medications (may interact with new drugs)",
    )
    current_regimen: List[str] = Field(
        default_factory=list,
        description="Drugs selected so far in this episode",
    )
    proposed_diagnosis: Optional[str] = Field(
        default=None,
        description="Diagnosis proposed by the agent so far",
    )
    feedback: str = Field(
        default="",
        description="Environment feedback on the last action",
    )
    valid_options: List[str] = Field(
        default_factory=list,
        description="Suggested valid action types for the next step",
    )
    reward_so_far: float = Field(
        default=0.0,
        description="Cumulative reward accumulated in this episode",
    )
    step_count: int = Field(
        default=0,
        description="Number of steps taken so far",
    )
    done: bool = Field(
        default=False,
        description="Whether the episode has ended",
    )
    reward: float = Field(
        default=0.0,
        description="Reward earned by the last action",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Episode metadata (episode_id, task, etc.)",
    )
