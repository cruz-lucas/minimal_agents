"""Implementation of the R-MAX algorithm with JAX arrays."""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp

from minimal_agents.agents.base import TabularAgent, UpdateResult
from minimal_agents.policies import ActionSelectionPolicy, EpsilonGreedyPolicy


class RMaxAgent(TabularAgent):
    """Model-based R-MAX agent."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        *,
        threshold: float,
        r_max: float = 1.0,
        m: int = 5,
        discount: float = 0.95,
        seed: int | None = None,
        policy: ActionSelectionPolicy | None = None,
    ) -> None:
        self.threshold = float(threshold)
        self.r_max = float(r_max)
        self.m = int(m)

        self._sa_counts = jnp.zeros((num_states, num_actions), dtype=jnp.int32)
        self._reward_sums = jnp.zeros((num_states, num_actions), dtype=jnp.float32)
        self._trans_counts = jnp.zeros(
            (num_states, num_actions, num_states), dtype=jnp.int32
        )

        super().__init__(
            num_states=num_states,
            num_actions=num_actions,
            discount=discount,
            seed=seed,
            policy=policy,
        )

    # ------------------------------------------------------------------ #
    def _initialise_parameters(self) -> None:
        optimistic_value = self.r_max / (1.0 - self.discount)
        self._q_values = jnp.full(
            (self.num_states, self.num_actions),
            optimistic_value,
            dtype=jnp.float32,
        )
        self._sa_counts = jnp.zeros_like(self._sa_counts)
        self._reward_sums = jnp.zeros_like(self._reward_sums)
        self._trans_counts = jnp.zeros_like(self._trans_counts)

    def _default_policy(self) -> ActionSelectionPolicy:
        return EpsilonGreedyPolicy(epsilon=0.0)

    # ------------------------------------------------------------------ #
    def _policy_extras(self, obs: int) -> Dict[str, jnp.ndarray]:
        counts_row = self._sa_counts[obs].astype(jnp.float32)
        total = jnp.sum(counts_row) + 1.0
        return {"counts": counts_row, "total": total}

    def is_known(self, obs: int, action: int) -> bool:
        return int(self._sa_counts[obs, action]) >= self.m

    def _reward_estimate(self, obs: int, action: int) -> float:
        count = int(self._sa_counts[obs, action])
        if count == 0:
            return self.r_max
        return float(self._reward_sums[obs, action] / count)

    def _transition_estimate(self, obs: int, action: int) -> jnp.ndarray:
        count = int(self._sa_counts[obs, action])
        if count == 0:
            return jnp.ones(self.num_states, dtype=jnp.float32) / float(self.num_states)
        return self._trans_counts[obs, action] / count

    # ------------------------------------------------------------------ #
    def update(
        self,
        obs: int,
        action: int,
        reward: float,
        next_obs: int,
        *,
        terminated: bool = False,
    ) -> UpdateResult:
        del terminated  # R-MAX is episodic but update logic does not depend on terminal flag.

        obs_idx = int(obs)
        action_idx = int(action)
        next_obs_idx = int(next_obs)

        if not self.is_known(obs_idx, action_idx):
            self._sa_counts = self._sa_counts.at[obs_idx, action_idx].add(1)
            self._reward_sums = self._reward_sums.at[obs_idx, action_idx].add(
                float(reward)
            )
            self._trans_counts = self._trans_counts.at[
                obs_idx, action_idx, next_obs_idx
            ].add(1)

            if self.is_known(obs_idx, action_idx):
                max_steps = int(
                    jnp.ceil(
                        jnp.log(1.0 / (self.threshold * (1.0 - self.discount)))
                        / (1.0 - self.discount)
                    )
                )
                for _ in range(max_steps):
                    old_q_values = self._q_values

                    for state in range(self.num_states):
                        for act in range(self.num_actions):
                            if self.is_known(state, act):
                                r_hat = self._reward_estimate(state, act)
                                t_hat = self._transition_estimate(state, act)
                                next_values = jnp.max(self._q_values, axis=1)
                                updated = r_hat + self.discount * jnp.dot(
                                    t_hat, next_values
                                )
                                self._q_values = self._q_values.at[state, act].set(
                                    float(updated)
                                )
                    
                    if jnp.max(jnp.abs(old_q_values - self._q_values)) < self.threshold:
                        break
        return UpdateResult()
