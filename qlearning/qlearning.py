"""Implementation of Q-learning Agent."""

import numpy as np
from gymnasium.utils import seeding


class QLearningAgent:
    """A simple tabular Q-learning agent."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        epsilon: np.float32,
        alpha: np.float32,
        discount: float = 0.95,
        seed: int | None = None,
    ):
        """Initializes the R-max agent with the given parameters.

        Creates the necessary data structures for counting visits and transitions,
        as well as storing estimates for rewards and the value function.

        Args:
            num_states (int): Number of states in the environment.
            num_actions (int): Number of possible actions.
            epsilon (np.float32): Chance of choosing a random action.
            alpha (np.float32): Step size.
            discount (float, optional): discount (float, optional): Discount factor. Defaults to 0.95.
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

    def update(self, obs: int, action: int, next_obs: int, reward: int | float):
        """Updates the internal model with a new experience.

        Increments visit counts for the given state-action pair, adds
        the observed reward, and updates transition counts.

        Args:
            obs (int): Current state.
            action (int): Action taken in the current state.
            next_obs (int): State reached after taking the action.
            reward (int | float): Reward received upon transitioning to next_state.
        """
        td_error = (
            reward
            + self.discount * np.max(self.Q[next_obs, action])
            - self.Q[obs, action]
        )
        self.Q[obs, action] += self.alpha * td_error

        self.td_errors = np.append(self.td_errors, td_error)

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
        if self.np_random.random() < self.epsilon:
            return int(self.np_random.choice(self.num_actions))
        return int(
            self.np_random.choice(
                np.argwhere(q_values == np.max(q_values)).reshape(-1, 1)
            )[0]
        )


if __name__ == "__main__":
    """
    Temporary environment tester function.
    """
    import gymnasium as gym
    import matplotlib.pyplot as plt
    import riverswim  # noqa: F401

    discount = 0.99
    epsilon = 1
    alpha = 0.2

    # env = gym.make("FrozenLake-v1", is_slippery=True)
    env = gym.make(
        id="RiverSwim-v0",
    )

    num_states = env.observation_space.n
    num_actions = env.action_space.n
    agent = QLearningAgent(
        num_states=num_states,
        num_actions=num_actions,
        epsilon=epsilon,
        alpha=alpha,
        discount=discount,
        seed=42,
    )

    n_steps = 100_000
    obs, _ = env.reset()
    for step in range(n_steps):
        action = agent.act(obs)
        next_obs, reward, done, truncated, info = env.step(action)
        agent.update(obs, action, next_obs, reward)
        obs = next_obs

    agent.epsilon = 0.0
    n_steps = 100
    total_return = 0
    returns = []

    obs, _ = env.reset()
    done = False

    for step in range(n_steps):
        action = agent.act(obs)
        next_obs, reward, done, truncated, info = env.step(action)
        total_return += reward
        obs = next_obs

        returns.append(total_return)

    plt.figure(figsize=(10, 6))
    plt.plot(returns, label="Cumulative Reward", color="blue")
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.title("Return")
    plt.legend()
    plt.grid()
    plt.show()

    env.close()
