"""Action-selection policies used by the tabular agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol

import jax
import jax.numpy as jnp
import jax.random as jrandom

PRNGKey = jax.Array


def _sample_best_action(key: PRNGKey, values: jnp.ndarray) -> int:
    """Samples uniformly among the maximisers of `values`."""
    num_actions = values.shape[-1]
    action_space = jnp.arange(num_actions, dtype=jnp.int32)
    best_value = jnp.max(values)
    mask = jnp.where(values == best_value, 1.0, 0.0)
    probs = mask / jnp.sum(mask)
    return int(jrandom.choice(key, action_space, p=probs))


class ActionSelectionPolicy(Protocol):
    """Protocol implemented by all action-selection policies."""

    def select(
        self,
        key: PRNGKey,
        values: jnp.ndarray,
        extras: Dict[str, jnp.ndarray] | None = None,
    ) -> tuple[int, PRNGKey, Dict[str, float]]:
        ...


@dataclass(slots=True)
class RandomWalkPolicy:
    """Chooses actions uniformly at random, ignoring value estimates."""

    def select(
        self,
        key: PRNGKey,
        values: jnp.ndarray,
        extras: Dict[str, jnp.ndarray] | None = None,
    ) -> tuple[int, PRNGKey, Dict[str, float]]:
        num_actions = values.shape[-1]
        action_space = jnp.arange(num_actions, dtype=jnp.int32)
        key, subkey = jrandom.split(key)
        action = jrandom.choice(subkey, action_space)
        return int(action), key, {"exploratory": 1.0}


@dataclass(slots=True)
class EpsilonGreedyPolicy:
    """Chooses the greedy action with probability 1 - epsilon; otherwise explores."""

    epsilon: float = 0.1

    def select(
        self,
        key: PRNGKey,
        values: jnp.ndarray,
        extras: Dict[str, jnp.ndarray] | None = None,
    ) -> tuple[int, PRNGKey, Dict[str, float]]:
        num_actions = values.shape[-1]
        action_space = jnp.arange(num_actions, dtype=jnp.int32)
        key, explore_key, random_key, greedy_key = jrandom.split(key, 4)

        explore = jrandom.uniform(explore_key) < self.epsilon
        random_action = jrandom.choice(random_key, action_space)
        greedy_action = jnp.asarray(
            _sample_best_action(greedy_key, values), dtype=jnp.int32
        )

        action = jnp.where(explore, random_action, greedy_action)
        return int(action), key, {"exploratory": float(explore)}


@dataclass(slots=True)
class UCBPolicy:
    """Upper-confidence bound action selection."""

    confidence: float = 1.0
    epsilon: float = 1e-6

    def select(
        self,
        key: PRNGKey,
        values: jnp.ndarray,
        extras: Dict[str, jnp.ndarray] | None = None,
    ) -> tuple[int, PRNGKey, Dict[str, float]]:
        if not extras or "counts" not in extras or "total" not in extras:
            raise ValueError("UCBPolicy requires 'counts' and 'total' in extras.")

        counts = jnp.asarray(extras["counts"], dtype=jnp.float32) + self.epsilon
        total = jnp.maximum(jnp.asarray(extras["total"], dtype=jnp.float32), 1.0)

        bonuses = self.confidence * jnp.sqrt(jnp.log(total + 1.0) / counts)
        scores = values + bonuses
        key, choice_key = jrandom.split(key)
        action = _sample_best_action(choice_key, scores)
        return action, key, {"exploratory": 0.0}


__all__ = [
    "ActionSelectionPolicy",
    "RandomWalkPolicy",
    "EpsilonGreedyPolicy",
    "UCBPolicy",
]
