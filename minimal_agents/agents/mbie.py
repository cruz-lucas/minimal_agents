"""Model-Based Interval Estimation with Exploration Bonus (MBIE-EB)."""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp

from minimal_agents.agents.base import TabularAgent, UpdateResult
from minimal_agents.policies import ActionSelectionPolicy, EpsilonGreedyPolicy


class MBIEAgent(TabularAgent):
    """Implementation of MBIE and MBIE-EB."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        *,
        r_max: float,
        discount: float = 0.95,
        epsilon_r_coeff: float = 0.3,
        epsilon_t_coeff: float = 0.0,
        exploration_coeff: float = 0.4,
        epsilon1: float = 0.01,
        m: int = 16,
        use_exploration_bonus: bool = True,
        seed: int | None = None,
        policy: ActionSelectionPolicy | None = None,
    ) -> None:
        self.r_max = float(r_max)
        self.epsilon_r_coeff = float(epsilon_r_coeff)
        self.epsilon_t_coeff = float(epsilon_t_coeff)
        self.exploration_coeff = float(exploration_coeff)
        self.epsilon1 = float(epsilon1)
        self.m = int(m)
        self.use_exploration_bonus = bool(use_exploration_bonus)

        self.beta = self.exploration_coeff * self.r_max
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

    def _policy_extras(self, obs: int) -> Dict[str, jnp.ndarray]:
        counts_row = self._sa_counts[obs].astype(jnp.float32)
        total = jnp.sum(counts_row) + 1.0
        return {"counts": counts_row, "total": total}

    # ------------------------------------------------------------------ #
    def is_known(self, obs: int, action: int) -> bool:
        return int(self._sa_counts[obs, action]) >= self.m

    def _reward_estimate(self, obs: int, action: int) -> tuple[float, float]:
        count = int(self._sa_counts[obs, action])
        if count > 0:
            mean = float(self._reward_sums[obs, action] / count)
            conf = self.epsilon_r_coeff * self.r_max / jnp.sqrt(count)
            return mean, float(conf)
        return self.r_max, float("inf")

    def _transition_estimate(self, obs: int, action: int) -> tuple[jnp.ndarray, float]:
        count = int(self._sa_counts[obs, action])
        if count > 0:
            dist = self._trans_counts[obs, action] / count
            conf = self.epsilon_t_coeff / jnp.sqrt(count)
            return dist, float(conf)
        uniform = jnp.ones(self.num_states, dtype=jnp.float32) / float(self.num_states)
        return uniform, float("inf")

    def _optimistic_transition_value(
        self,
        T_hat: jnp.ndarray,
        V: jnp.ndarray,
        epsilon_T: float,
    ) -> float:
        T_tilde = T_hat.astype(jnp.float32).copy()
        extra = epsilon_T / 2.0
        s_star = int(jnp.argmax(V))
        T_tilde = T_tilde.at[s_star].add(extra)

        remaining = extra
        order = jnp.argsort(V)
        for idx in order:
            idx_int = int(idx)
            if idx_int == s_star:
                continue
            if remaining <= 0:
                break
            removable = min(float(T_tilde[idx_int]), remaining)
            T_tilde = T_tilde.at[idx_int].add(-removable)
            remaining -= removable

        T_tilde = jnp.clip(T_tilde, a_min=0.0)
        T_tilde = T_tilde / jnp.sum(T_tilde)
        return float(jnp.dot(T_tilde, V))

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
        del terminated  # MBIE is episodic but update logic does not depend on terminal flag.

        obs_idx = int(obs)
        action_idx = int(action)
        next_obs_idx = int(next_obs)

        current_count = int(self._sa_counts[obs_idx, action_idx])
        if current_count < self.m:
            self._sa_counts = self._sa_counts.at[obs_idx, action_idx].add(1)
            self._reward_sums = self._reward_sums.at[obs_idx, action_idx].add(
                float(reward)
            )
            self._trans_counts = self._trans_counts.at[
                obs_idx, action_idx, next_obs_idx
            ].add(1)

            max_iters = int(
                jnp.ceil(
                    jnp.log(1.0 / (self.epsilon1 * (1.0 - self.discount)))
                    / (1.0 - self.discount)
                )
            )

            Q_new = self._q_values
            for _ in range(max_iters):
                V = jnp.max(Q_new, axis=1)
                updated = Q_new
                for state in range(self.num_states):
                    for act in range(self.num_actions):
                        count = int(self._sa_counts[state, act])
                        if count == 0:
                            updated = updated.at[state, act].set(
                                self.r_max / (1.0 - self.discount)
                            )
                            continue

                        r_hat, r_conf = self._reward_estimate(state, act)
                        T_hat, t_conf = self._transition_estimate(state, act)

                        if self.use_exploration_bonus:
                            bonus = self.beta / jnp.sqrt(float(count))
                            target = (
                                r_hat
                                + self.discount * float(jnp.dot(T_hat, V))
                                + float(bonus)
                            )
                        else:
                            opt_val = self._optimistic_transition_value(
                                T_hat, V, t_conf
                            )
                            target = r_hat + self.discount * opt_val + r_conf

                        updated = updated.at[state, act].set(float(target))
                if jnp.max(jnp.abs(updated - Q_new)) < 1e-2:
                    Q_new = updated
                    break
                Q_new = updated

            self._q_values = Q_new

        return UpdateResult()
