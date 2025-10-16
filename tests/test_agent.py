
"""Tests for the minimal_agents package."""

import jax
import jax.numpy as jnp
import pytest

from minimal_agents.agents import (
    IntrinsicQLearningAgent,
    QLearningAgent,
    SARSAAgent,
)
from minimal_agents.policies import (
    EpsilonGreedyPolicy,
    RandomWalkPolicy,
    UCBPolicy,
)


@pytest.fixture
def q_agent() -> QLearningAgent:
    return QLearningAgent(
        num_states=4,
        num_actions=2,
        learning_rate=0.5,
        epsilon=0.1,
        discount=0.9,
        initial_value=0.0,
        seed=42,
    )


@pytest.fixture
def sarsa_agent() -> SARSAAgent:
    return SARSAAgent(
        num_states=4,
        num_actions=2,
        learning_rate=0.5,
        epsilon=0.1,
        discount=0.9,
        initial_value=0.0,
        seed=42,
    )


def test_q_learning_initialisation(q_agent: QLearningAgent) -> None:
    assert q_agent.num_states == 4
    assert q_agent.num_actions == 2
    assert q_agent.q_values.shape == (4, 2)
    assert jnp.allclose(q_agent.q_values, 0.0)


def test_sarsa_initialisation(sarsa_agent: SARSAAgent) -> None:
    assert sarsa_agent.num_states == 4
    assert sarsa_agent.num_actions == 2
    assert sarsa_agent.q_values.shape == (4, 2)
    assert jnp.allclose(sarsa_agent.q_values, 0.0)


def test_greedy_action_selection(q_agent: QLearningAgent) -> None:
    q_agent.set_epsilon(0.0)
    q_agent._q_values = q_agent._q_values.at[0].set(
        jnp.array([10.0, 5.0], dtype=jnp.float32)
    )
    action = q_agent.select_action(0)
    assert action == 0


def test_random_action_selection(q_agent: QLearningAgent) -> None:
    q_agent.set_epsilon(1.0)
    actions = {q_agent.select_action(0) for _ in range(30)}
    assert actions == {0, 1}


def test_q_learning_update(q_agent: QLearningAgent) -> None:
    q_agent._q_values = q_agent._q_values.at[0, 1].set(5.0)
    q_agent._q_values = q_agent._q_values.at[2].set(
        jnp.array([1.0, 3.0], dtype=jnp.float32)
    )

    result = q_agent.update(obs=0, action=1, reward=2.0, next_obs=2)

    assert pytest.approx(q_agent.q_values[0, 1], rel=1e-6) == 4.85
    assert pytest.approx(result.td_error, rel=1e-6) == -0.3
    assert pytest.approx(q_agent.td_errors[-1], rel=1e-6) == -0.3


def test_sarsa_update_returns_next_action(sarsa_agent: SARSAAgent) -> None:
    sarsa_agent.set_epsilon(0.0)
    sarsa_agent._q_values = sarsa_agent._q_values.at[0, 1].set(5.0)
    sarsa_agent._q_values = sarsa_agent._q_values.at[2, 0].set(3.0)
    sarsa_agent._q_values = sarsa_agent._q_values.at[2, 1].set(1.0)

    result = sarsa_agent.update(obs=0, action=1, reward=2.0, next_obs=2)

    assert pytest.approx(sarsa_agent.q_values[0, 1], rel=1e-6) == 4.85
    assert pytest.approx(result.td_error, rel=1e-6) == -0.3
    assert result.info["next_action"] == 0


def test_intrinsic_bonus_applied() -> None:
    agent = IntrinsicQLearningAgent(
        num_states=3,
        num_actions=2,
        intrinsic_scale=1.0,
        learning_rate=0.5,
        epsilon=0.0,
        discount=0.9,
        seed=0,
    )

    agent._q_values = agent._q_values.at[1].set(jnp.array([0.0, 0.0], dtype=jnp.float32))
    result = agent.update(obs=0, action=0, reward=0.0, next_obs=1)

    assert pytest.approx(result.info["intrinsic_reward"], rel=1e-6) == 1.0
    assert pytest.approx(result.td_error, rel=1e-6) == 1.0
    assert pytest.approx(agent.td_errors[-1], rel=1e-6) == 1.0


def test_policy_modules_behaviour() -> None:
    values = jnp.array([1.0, 2.0, 0.5], dtype=jnp.float32)

    greedy = EpsilonGreedyPolicy(epsilon=0.0)
    key = jax.random.PRNGKey(0)
    action, key, _ = greedy.select(key, values)
    assert action == 1

    random_policy = RandomWalkPolicy()
    actions = set()
    for _ in range(20):
        action, key, _ = random_policy.select(key, values)
        actions.add(action)
    assert actions <= {0, 1, 2}
    assert actions  # ensure at least one action was sampled

    ucb = UCBPolicy(confidence=1.0)
    with pytest.raises(ValueError):
        ucb.select(key, values)

    extras = {"counts": jnp.array([10.0, 1.0, 1.0]), "total": jnp.array(12.0)}
    action, _, _ = ucb.select(key, values, extras=extras)
    assert action == 1


def test_seeding_reproducibility() -> None:
    agent_a = QLearningAgent(
        num_states=2,
        num_actions=3,
        learning_rate=0.5,
        epsilon=0.2,
        discount=0.95,
        seed=123,
    )
    agent_b = QLearningAgent(
        num_states=2,
        num_actions=3,
        learning_rate=0.5,
        epsilon=0.2,
        discount=0.95,
        seed=123,
    )

    sequence_a = [agent_a.select_action(0) for _ in range(10)]
    sequence_b = [agent_b.select_action(0) for _ in range(10)]
    assert sequence_a == sequence_b
