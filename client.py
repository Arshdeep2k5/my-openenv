# client.py
"""Pharma Agent Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import PharmaAgentAction, PharmaAgentObservation


class PharmaAgentEnv(
    EnvClient[PharmaAgentAction, PharmaAgentObservation, State]
):
    """
    Client for the Pharma Agent Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.

    Example:
        >>> with PharmaAgentEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     print(result.observation.feedback)
        ...     result = env.step(PharmaAgentAction(action_type="diagnose", value="Hypertension"))
        ...     result = env.step(PharmaAgentAction(action_type="select_drug", value="Lisinopril"))
        ...     result = env.step(PharmaAgentAction(action_type="finalize", value="finalize"))
    """

    def _step_payload(self, action: PharmaAgentAction) -> Dict:
        return {
            "action_type": action.action_type,
            "value": action.value,
        }

    def _parse_result(self, payload: Dict) -> StepResult[PharmaAgentObservation]:
        obs_data = payload.get("observation", {})
        observation = PharmaAgentObservation(
            task=obs_data.get("task", "easy"),
            phase=obs_data.get("phase", "triage"),
            symptoms=obs_data.get("symptoms", []),
            existing_medications=obs_data.get("existing_medications", []),
            current_regimen=obs_data.get("current_regimen", []),
            proposed_diagnosis=obs_data.get("proposed_diagnosis"),
            feedback=obs_data.get("feedback", ""),
            valid_options=obs_data.get("valid_options", []),
            reward_so_far=obs_data.get("reward_so_far", 0.0),
            step_count=obs_data.get("step_count", 0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
