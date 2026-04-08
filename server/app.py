# app.py
"""
FastAPI application for the Pharma Agent Environment.

Endpoints:
    POST /reset  — Reset the environment, optionally pass {"task": "easy|medium|hard"}
    POST /step   — Execute an action
    GET  /state  — Get current environment state
    GET  /health — Health check
    WS   /ws     — WebSocket for persistent sessions
"""
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in [_HERE, _ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: pip install openenv-core"
    ) from e

from models import PharmaAgentAction, PharmaAgentObservation
from pharma_agent_environment import PharmaAgentEnvironment


app = create_app(
    PharmaAgentEnvironment,
    PharmaAgentAction,
    PharmaAgentObservation,
    env_name="pharma_agent",
    max_concurrent_envs=8,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    main(host=args.host, port=args.port)
