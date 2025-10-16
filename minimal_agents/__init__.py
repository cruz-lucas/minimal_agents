"""Minimal tabular reinforcement learning agents powered by JAX."""

from minimal_agents.agents import (
    MBIEAgent,
    QLearningAgent,
    IntrinsicQLearningAgent,
    RMaxAgent,
    SARSAAgent,
    TabularAgent,
    UpdateResult,
)
from minimal_agents import policies

__all__ = [
    "TabularAgent",
    "UpdateResult",
    "QLearningAgent",
    "IntrinsicQLearningAgent",
    "SARSAAgent",
    "RMaxAgent",
    "MBIEAgent",
    "policies",
]
