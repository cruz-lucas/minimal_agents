"""Implementation of SARSA Agent."""

import numpy as np
from gymnasium.utils import seeding


class SARSAAgent:
    """A simple tabular SARSA agent."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        epsilon: np.float32,
        alpha: np.float32,
        discount: float = 0.95,
        initial_value: int | float = 0,
        seed: int | None = None,
    ):
        """Initializes the SARSA agent with the given parameters.

        Args:
            num_states (int): Number of states in the environment.
            num_actions (int): Number of possible actions.
            epsilon (np.float32): Chance of choosing a random action.
            alpha (np.float32): Step size.
            discount (float, optional): discount (float, optional): Discount factor. Defaults to 0.95.
            initial_value (int | float): Initial value for the Q values. Defaults to 0.
            seed (int | None, optional): Seed to ensure reproducibility. Defaults to None.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.discount = discount
        self.epsilon = epsilon
        self.alpha = alpha

        self.td_errors: np.ndarray = np.array([])

        self.Q = np.full((num_states, num_actions), dtype=np.float64, fill_value=0)
        self.np_random, _ = seeding.np_random(seed)

    def update(self, obs: int, action: int, next_obs: int, reward: int | float) -> int:
        """Updates the Q-values.

        Args:
            obs (int): Current observation.
            action (int): Action taken in the current state.
            next_obs (int): Observation received after taking the action.
            reward (int | float): Reward received upon transitioning to next_state.

        Returns:
            int: The next action.
        """
        next_action = self.act(next_obs)

        td_error = (
            reward + self.discount * self.Q[next_obs, next_action] - self.Q[obs, action]
        )
        self.Q[obs, action] += self.alpha * td_error

        self.td_errors = np.append(self.td_errors, td_error)

        return next_action

    def act(self, obs) -> int:
        """Selects an action based on the current value function.

        Args:
            obs (int): Current observation from which to choose an action.

        Returns:
            int: The action that maximizes the estimated Q-value or a random action.
        """
        q_values = self.Q[obs]
        if self.np_random.random() < self.epsilon:
            return int(self.np_random.choice(self.num_actions))
        return int(
            self.np_random.choice(
                np.argwhere(q_values == np.max(q_values)).reshape(-1, 1)
            )[0]
        )
