"""Shared abstractions for tabular agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import jax.random as jrandom

from minimal_agents.policies import ActionSelectionPolicy

PRNGKey = jax.Array


def _canonical_seed(seed: Optional[int]) -> int:
    """Converts an optional seed into a 32-bit unsigned integer."""
    if seed is None:
        # JAX expects 32-bit unsigned integers as seeds. We rely on Python's secrets
        # module for entropy to keep the default behaviour reproducible-enough without
        # forcing consumers to pass a seed.
        import secrets

        return secrets.randbits(32)

    if not isinstance(seed, int):
        raise TypeError("Seed must be an integer or None.")
    return seed & 0xFFFFFFFF


@dataclass(slots=True)
class UpdateResult:
    """Standard container for returning diagnostic information from updates."""

    td_error: Optional[float] = None
    info: Dict[str, Any] | None = None

    def as_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation for convenience."""
        payload: Dict[str, Any] = {}
        if self.td_error is not None:
            payload["td_error"] = self.td_error
        if self.info:
            payload.update(self.info)
        return payload


class TabularAgent(ABC):
    """Base class for all tabular control agents in the library."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        discount: float = 0.95,
        *,
        seed: int | None = None,
        policy: ActionSelectionPolicy | None = None,
    ) -> None:
        if num_states <= 0 or num_actions <= 0:
            raise ValueError("`num_states` and `num_actions` must be positive.")

        self.num_states = int(num_states)
        self.num_actions = int(num_actions)
        self.discount = float(discount)

        self._seed = _canonical_seed(seed)
        self._key: PRNGKey = jrandom.PRNGKey(self._seed)

        self._policy: ActionSelectionPolicy | None = policy
        self._last_action_info: Dict[str, Any] | None = None

        self.reset()

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None) -> None:
        """Resets the agent state, optionally re-seeding the RNG."""
        if seed is not None:
            self._seed = _canonical_seed(seed)
            self._key = jrandom.PRNGKey(self._seed)

        self._initialise_parameters()
        if self._policy is None:
            self._policy = self._default_policy()

    @abstractmethod
    def _initialise_parameters(self) -> None:
        """Subclasses must set up their internal state."""

    @abstractmethod
    def _default_policy(self) -> ActionSelectionPolicy:
        """Provides a sensible default policy for the agent."""

    # ------------------------------------------------------------------ #
    # Interaction                                                         #
    # ------------------------------------------------------------------ #
    def select_action(self, obs: int) -> int:
        """Selects an action using the configured policy."""
        if self._policy is None:
            raise RuntimeError("No policy has been set for this agent.")

        obs_idx = int(obs)
        if obs_idx < 0 or obs_idx >= self.num_states:
            raise IndexError(f"Observation {obs_idx} outside [0, {self.num_states}).")

        q_values = self.q_values[obs_idx]
        extras = self._policy_extras(obs_idx)
        action, new_key, info = self._policy.select(self._key, q_values, extras)

        self._key = new_key
        self._last_action_info = info
        return int(action)

    @abstractmethod
    def update(
        self,
        obs: int,
        action: int,
        reward: float,
        next_obs: int,
        *,
        terminated: bool = False,
    ) -> UpdateResult:
        """Updates the agent with a transition and returns diagnostic info."""

    # ------------------------------------------------------------------ #
    # Utilities                                                           #
    # ------------------------------------------------------------------ #
    def set_policy(self, policy: ActionSelectionPolicy) -> None:
        """Configures the action-selection policy."""
        self._policy = policy

    def last_action_info(self) -> Dict[str, Any] | None:
        """Returns diagnostic info from the last call to `select_action`."""
        return self._last_action_info

    def _policy_extras(self, obs: int) -> Dict[str, Any]:
        """Optional hook for subclasses to provide additional policy context."""
        return {}

    @property
    def q_values(self) -> jnp.ndarray:
        """Returns the current Q-value table."""
        return self._q_values

    @property
    def rng_key(self) -> PRNGKey:
        """Returns the current JAX PRNGKey."""
        return self._key
