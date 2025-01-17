"""Tests module."""

import numpy as np
import pytest
from agents import QLearningAgent, SARSAAgent


@pytest.fixture
def q_agent() -> QLearningAgent:
    """Pytest fixture to create a Q-Learning Agent with default parameters."""
    return QLearningAgent(
        num_states=4,
        num_actions=2,
        epsilon=0.1,
        alpha=0.5,
        discount=0.9,
        initial_value=0.0,
        seed=42,
    )


@pytest.fixture
def sarsa_agent() -> SARSAAgent:
    """Pytest fixture to create a SARSA Agent with default parameters."""
    return SARSAAgent(
        num_states=4,
        num_actions=2,
        epsilon=0.1,
        alpha=0.5,
        discount=0.9,
        initial_value=0.0,
        seed=42,
    )


def test_initialization_qlearning(q_agent: QLearningAgent) -> None:
    """Test that the QLearningAgent initializes correctly.

    Args:
        q_agent (QLearningAgent): Agent to be tested.
    """
    assert q_agent.num_states == 4
    assert q_agent.num_actions == 2
    assert q_agent.epsilon == 0.1
    assert q_agent.alpha == 0.5
    assert q_agent.discount == 0.9
    assert q_agent.Q.shape == (4, 2)
    assert np.all(q_agent.Q == 0.0)


def test_initialization_sarsa(sarsa_agent: SARSAAgent) -> None:
    """Test that the SARSAAgent initializes correctly.

    Args:
        sarsa_agent (SARSAAgent): Agent to be tested.
    """
    assert sarsa_agent.num_states == 4
    assert sarsa_agent.num_actions == 2
    assert sarsa_agent.epsilon == 0.1
    assert sarsa_agent.alpha == 0.5
    assert sarsa_agent.discount == 0.9
    assert sarsa_agent.Q.shape == (4, 2)
    assert np.all(sarsa_agent.Q == 0.0)


def test_action_selection_greedy_qlearning(q_agent: QLearningAgent) -> None:
    """Test QLearningAgent's greedy action selection.

    With epsilon=0.0, the agent should always pick the action
    with the highest Q-value.

    Args:
        q_agent (QLearningAgent): Agent to be tested.
    """
    q_agent.epsilon = 0.0
    q_agent.Q[0] = [10.0, 5.0]  # Best action = 0
    action = q_agent.act(0)
    assert action == 0, "QLearningAgent did not pick the greedy action."


def test_action_selection_greedy_sarsa(sarsa_agent: SARSAAgent) -> None:
    """Test SARSAAgent's greedy action selection.

    With epsilon=0.0, the agent should always pick the action
    with the highest Q-value.

    Args:
        sarsa_agent (SARSAAgent): Agent to be tested.
    """
    sarsa_agent.epsilon = 0.0
    sarsa_agent.Q[0] = [2.0, 5.0]  # Best action = 1
    action = sarsa_agent.act(0)
    assert action == 1, "SARSAAgent did not pick the greedy action."


def test_action_selection_random(q_agent: QLearningAgent) -> None:
    """Test QLearningAgent's random action selection with epsilon=1.0.

    With epsilon=1.0, the agent should pick actions at random.

    Args:
        q_agent (QLearningAgent): Agent to be tested.
    """
    q_agent.epsilon = 1.0
    actions = [q_agent.act(0) for _ in range(50)]
    assert len(set(actions)) == 2, "QLearningAgent is not picking random actions."


def test_td_update_qlearning(q_agent: QLearningAgent) -> None:
    """Test QLearningAgent's Q-value update (temporal difference update).

    We set a known Q-value, perform update, and check new Q-value.

    Args:
        q_agent (QLearningAgent): Agent to be tested.
    """
    initial_q = 5.0
    q_agent.Q[0, 1] = initial_q  # Q(0,1) = 5.0
    q_agent.Q[2] = [1.0, 3.0]  # The max of Q(2,.) = 3.0

    q_agent.update(obs=0, action=1, next_obs=2, reward=2)

    # TD error = reward + discount*maxQ(next_obs) - Q(obs,action) = 2 + 0.9*3 - 5 = 2 + 2.7 - 5 = -0.3
    # Q(obs, action) += alpha * TD error = 5 + 0.5*(-0.3) = 5 - 0.15 = 4.85
    expected_new_q = 4.85
    np.testing.assert_almost_equal(
        q_agent.Q[0, 1],
        expected_new_q,
        decimal=5,
        err_msg="QLearningAgent Q-value update is incorrect.",
    )

    assert len(q_agent.td_errors) == 1, "TD error not recorded properly."
    np.testing.assert_almost_equal(
        q_agent.td_errors[-1], -0.3, decimal=5, err_msg="TD error not stored correctly."
    )


def test_td_update_sarsa(sarsa_agent: SARSAAgent) -> None:
    """Test SARSAAgent's Q-value update (temporal difference update).

    We'll check that, after one update call, the Q-values match the expected update formula.

    Args:
        sarsa_agent (_type_): Agent to be tested.
    """
    # We'll force the next action to be something we control by setting epsilon=0.0
    sarsa_agent.epsilon = 0.0
    sarsa_agent.Q[0, 1] = 5.0  # Current Q(0,1) = 5
    sarsa_agent.Q[2, 0] = (
        3.0  # Suppose we want next_action=0 to be the best for state=2
    )
    sarsa_agent.Q[2, 1] = 1.0

    # Update with: obs=0, action=1, next_obs=2, reward=2
    # The agent will pick next_action=0 in next_obs=2 because Q[2,0] > Q[2,1].
    next_action = sarsa_agent.update(obs=0, action=1, next_obs=2, reward=2)

    # For SARSA, TD error = R + discount * Q(next_obs, next_action) - Q(obs, action)
    # = 2 + 0.9*3.0 - 5.0 = 2 + 2.7 - 5 = -0.3
    # Q(0,1) += alpha * TD error = 5 + 0.5*(-0.3) = 5 - 0.15 = 4.85
    expected_q_value = 4.85
    expected_td_error = -0.3

    np.testing.assert_almost_equal(
        sarsa_agent.Q[0, 1],
        expected_q_value,
        decimal=5,
        err_msg="SARSAAgent Q-value update is incorrect.",
    )

    assert next_action == 0, "SARSAAgent did not return the expected next action."

    assert len(sarsa_agent.td_errors) == 1, "TD error not recorded properly for SARSA."
    np.testing.assert_almost_equal(
        sarsa_agent.td_errors[-1],
        expected_td_error,
        decimal=5,
        err_msg="SARSAAgent TD error not stored correctly.",
    )


def test_agent_seeding() -> None:
    """Test that seeding leads to deterministic behavior for both agents.

    We'll initialize two agents with the same seed and verify that
    they produce the same sequence of actions for the same states.
    """
    seed = 1234
    q_agent1 = QLearningAgent(4, 2, 0.5, 0.5, seed=seed)
    q_agent2 = QLearningAgent(4, 2, 0.5, 0.5, seed=seed)

    sarsa_agent1 = SARSAAgent(4, 2, 0.5, 0.5, seed=seed)
    sarsa_agent2 = SARSAAgent(4, 2, 0.5, 0.5, seed=seed)

    # Test action sequences for QLearningAgent
    actions1 = [q_agent1.act(0) for _ in range(10)]
    actions2 = [q_agent2.act(0) for _ in range(10)]
    assert (
        actions1 == actions2
    ), "QLearningAgents with the same seed produced different actions."

    # Test action sequences for SARSAAgent
    actions1 = [sarsa_agent1.act(0) for _ in range(10)]
    actions2 = [sarsa_agent2.act(0) for _ in range(10)]
    assert (
        actions1 == actions2
    ), "SARSAAgents with the same seed produced different actions."
