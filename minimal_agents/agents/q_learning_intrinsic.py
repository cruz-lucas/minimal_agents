"""Q-learning agent augmented with count-based intrinsic motivation."""

from __future__ import annotations

import jax.numpy as jnp

from minimal_agents.agents.base import UpdateResult
from minimal_agents.agents.q_learning import QLearningAgent


class IntrinsicQLearningAgent(QLearningAgent):
    """Extends Q-learning with a simple count-based intrinsic reward signal."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        *,
        intrinsic_scale: float = 0.1,
        discount: float = 0.95,
        learning_rate: float = 0.5,
        epsilon: float = 0.1,
        initial_value: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.intrinsic_scale = float(intrinsic_scale)
        super().__init__(
            num_states=num_states,
            num_actions=num_actions,
            discount=discount,
            learning_rate=learning_rate,
            epsilon=epsilon,
            initial_value=initial_value,
            seed=seed,
        )

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

        count = self._visit_counts[obs_idx, action_idx] + 1.0
        intrinsic_reward = self.intrinsic_scale / jnp.sqrt(count)
        augmented_reward = float(reward) + float(intrinsic_reward)

        result = super().update(
            obs=obs_idx,
            action=action_idx,
            reward=augmented_reward,
            next_obs=next_obs,
            done=done,
        )

        info = dict(result.info or {})
        info["intrinsic_reward"] = float(intrinsic_reward)
        return UpdateResult(td_error=result.td_error, info=info)
