"""Classical tabular Q-learning implemented with JAX arrays."""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp

from minimal_agents.agents.base import TabularAgent, UpdateResult
from minimal_agents.policies import ActionSelectionPolicy, EpsilonGreedyPolicy


class QLearningAgent(TabularAgent):
    """Standard off-policy Q-learning agent."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        *,
        learning_rate: float = 0.5,
        epsilon: float = 0.1,
        discount: float = 0.95,
        initial_value: float = 0.0,
        seed: int | None = None,
        policy: ActionSelectionPolicy | None = None,
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.epsilon = float(epsilon)
        self.initial_value = float(initial_value)

        self._td_errors: list[float] = []
        self._visit_counts = jnp.zeros((num_states, num_actions), dtype=jnp.float32)

        super().__init__(
            num_states=num_states,
            num_actions=num_actions,
            discount=discount,
            seed=seed,
            policy=policy,
        )

    # ------------------------------------------------------------------ #
    def _initialise_parameters(self) -> None:
        self._q_values = jnp.full(
            (self.num_states, self.num_actions),
            self.initial_value,
            dtype=jnp.float32,
        )
        self._td_errors.clear()
        self._visit_counts = jnp.zeros_like(self._visit_counts)

    def _default_policy(self) -> ActionSelectionPolicy:
        return EpsilonGreedyPolicy(self.epsilon)

    # ------------------------------------------------------------------ #
    def _policy_extras(self, obs: int) -> Dict[str, jnp.ndarray]:
        counts_row = self._visit_counts[obs]
        total = jnp.sum(counts_row) + 1.0
        return {"counts": counts_row, "total": total}

    def update(
        self,
        obs: int,
        action: int,
        reward: float,
        next_obs: int,
        *,
        done: bool = False,
    ) -> UpdateResult:
        obs_idx = int(obs)
        action_idx = int(action)
        next_obs_idx = int(next_obs)

        reward_val = jnp.asarray(reward, dtype=jnp.float32)
        done_mask = jnp.asarray(done, dtype=jnp.float32)

        next_value = jnp.max(self._q_values[next_obs_idx])
        target = reward_val + self.discount * (1.0 - done_mask) * next_value
        td_error = target - self._q_values[obs_idx, action_idx]

        self._q_values = self._q_values.at[obs_idx, action_idx].add(
            self.learning_rate * td_error
        )
        self._visit_counts = self._visit_counts.at[obs_idx, action_idx].add(1.0)

        td_float = float(td_error)
        self._td_errors.append(td_float)
        return UpdateResult(td_error=td_float)

    # ------------------------------------------------------------------ #
    @property
    def td_errors(self) -> list[float]:
        return self._td_errors

    def set_epsilon(self, epsilon: float) -> None:
        """Updates the exploration rate and refreshes the default policy."""
        self.epsilon = float(epsilon)
        if isinstance(self._policy, EpsilonGreedyPolicy):
            self._policy.epsilon = self.epsilon
