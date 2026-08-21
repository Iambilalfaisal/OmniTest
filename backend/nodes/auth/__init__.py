"""Auth subgraph: establishes ONE shared logged-in session before the worker fan-out,
replacing the original nodes/auth_setup.py hand-rolled loop with the same three-node
shape as nodes/worker/nodes.py (agent/tool/save instead of agent/tool/verdict) so it
gains ask_human and risky-action review for free via nodes/agent_loop.py's shared
primitives. Only build_auth_subgraph is used outside this package (graph/builder.py).
"""
from .nodes import build_auth_subgraph

__all__ = ["build_auth_subgraph"]
