"""User-facing entry-points for the agents package."""

from minimal_agents.agents.base import TabularAgent, UpdateResult
from minimal_agents.agents.mbie import MBIEAgent
from minimal_agents.agents.q_learning import QLearningAgent
from minimal_agents.agents.q_learning_intrinsic import IntrinsicQLearningAgent
from minimal_agents.agents.rmax import RMaxAgent
from minimal_agents.agents.sarsa import SARSAAgent

__all__ = [
    "TabularAgent",
    "UpdateResult",
    "QLearningAgent",
    "IntrinsicQLearningAgent",
    "SARSAAgent",
    "RMaxAgent",
    "MBIEAgent",
]
