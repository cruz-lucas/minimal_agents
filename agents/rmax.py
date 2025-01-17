"""Implementation of R-max Agent."""

import numpy as np
from gymnasium.utils import seeding


class RMaxAgent:
    """A simple tabular R-max agent."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        epsilon1: float,
        r_max: float = 1.0,
        m: int = 5,
        discount: float = 0.95,
        seed: int | None = None,
    ):
        """Initializes the R-max agent with the given parameters.

        Creates the necessary data structures for counting visits and transitions,
        as well as storing estimates for rewards and the value function.

        Args:
            num_states (int): Number of states in the environment.
            num_actions (int): Number of possible actions.
            epsilon1 (float): Number of iterations the optimal policy will return near-optimal value (U(pi) - epsilon) on average.
                Defaults to 10.
            r_max (float, optional): Maximum possible reward for unknown
                state-action pairs. Defaults to 1.0.
            m (int, optional): Minimum number of visits required to consider
                a state-action pair as known. Defaults to 5.
            discount (float, optional): Discount factor. Defaults to 0.95.
            seed (int | None, optional): Seed to ensure reproducibility. Defaults to None.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.r_max = r_max
        self.m = m
        self.discount = discount

        self.epsilon1 = epsilon1

        self.sa_counts = np.zeros((num_states, num_actions), dtype=np.int32)
        self.reward_sums = np.zeros((num_states, num_actions), dtype=np.float32)
        self.trans_counts = np.zeros(
            (num_states, num_actions, num_states), dtype=np.int32
        )

        self.Q = np.full(
            (num_states, num_actions),
            dtype=np.float32,
            fill_value=r_max / (1 - discount),
        )
        self.np_random, _ = seeding.np_random(seed)

    def update(self, obs, action, next_obs, reward):
        """Updates the internal model with a new experience.

        Increments visit counts for the given state-action pair, adds
        the observed reward, and updates transition counts.

        Args:
            obs (int): Current state.
            action (int): Action taken in the current state.
            next_obs (int): State reached after taking the action.
            reward (float): Reward received upon transitioning to next_state.
        """
        if not self.is_known(obs, action):
            self.sa_counts[obs, action] += 1
            self.reward_sums[obs, action] += reward
            self.trans_counts[obs, action, next_obs] += 1

            if self.is_known(obs, action):
                max_steps = int(
                    np.log(1 / (self.epsilon1 * (1 - self.discount)))
                    / (1 - self.discount)
                )
                for _ in range(max_steps):
                    for state in range(self.num_states):
                        for action in range(self.num_actions):
                            if self.is_known(state, action):
                                r_hat = self.get_reward_estimate(
                                    obs=state, action=action
                                )
                                t_hat = self.get_transition_estimate(
                                    obs=state, action=action
                                )
                                self.Q[state, action] = r_hat + self.discount * np.dot(
                                    t_hat, self.Q.max(axis=1)
                                )

    def is_known(self, obs, action):
        """Determines if a given state-action pair is known.

        A pair is considered known if it has been visited at least m times.

        Args:
            state (int): State of interest.
            action (int): Action of interest.

        Returns:
            bool: True if the pair is known, otherwise False.
        """
        return self.sa_counts[obs, action] >= self.m

    def get_reward_estimate(self, obs, action):
        """Computes the estimated reward for a state-action pair.

        If the pair is known, returns the average observed reward.
        Otherwise, returns r_max.

        Args:
            state (int): State of interest.
            action (int): Action of interest.

        Returns:
            float: Estimated reward for the given state-action pair.
        """
        if self.is_known(obs, action):
            return self.reward_sums[obs, action] / self.sa_counts[obs, action]
        else:
            return self.r_max

    def get_transition_estimate(self, obs, action):
        """Computes the transition probability distribution.

        If the state-action pair is known, returns the observed distribution.
        Otherwise, returns an optimistic (e.g., uniform) distribution.

        Args:
            state (int): State of interest.
            action (int): Action of interest.

        Returns:
            np.ndarray: Transition probabilities for all next states.
        """
        if self.is_known(obs, action):
            total = self.sa_counts[obs, action]
            return (
                self.trans_counts[obs, action] / total
            )  # this will return a vector of shape (n_states, ) that sum to 1.
        else:
            return np.ones(self.num_states) / float(self.num_states)  # uniform dist

    def act(self, obs) -> int:
        """Selects an action based on the current value function.

        Computes the estimated Q-value for each possible action and
        returns the action that yields the highest value.

        Args:
            obs (int): Current state from which to choose an action.

        Returns:
            int: The action that maximizes the estimated Q-value.
        """
        q_values = self.Q[obs]
        return self.np_random.choice(
            np.argwhere(q_values == np.max(q_values)).reshape((-1,))
        )
