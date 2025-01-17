"""Base agent class."""

from gymnasium.utils import seeding


class BaseAgent:
    """A base class for tabular RL agents.

    This class defines the required methods
    and basic attributes that all agents should have, but does not implement
    any specific RL algorithm. Subclasses should override `update` and `act`
    with algorithm-specific logic.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        discount: float = 0.95,
        seed: int | None = None,
    ):
        """Initializes common agent attributes.

        Args:
            num_states (int): Number of states in the environment.
            num_actions (int): Number of possible actions.
            discount (float, optional): Discount factor for future rewards.
                Defaults to 0.95.
            seed (int | None, optional): Random seed for reproducibility.
                Defaults to None.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.discount = discount

        self.np_random, _ = seeding.np_random(seed)

    def update(self, obs: int, action: int, next_obs: int, reward: int | float):
        """Updates the agent's internal parameters (e.g., Q-table).

        Subclasses must override this method to implement the
        algorithm-specific logic for updating Q-values or other parameters.

        Args:
            obs (int): The current state (observation).
            action (int): The action taken in the current state.
            next_obs (int): The new state after taking the action.
            reward (int | float): The reward received upon transitioning.
        """
        raise NotImplementedError(
            "The `update` method must be overridden by subclasses."
        )

    def act(self, obs: int) -> int:
        """Selects an action based on the agent's current policy.

        Subclasses must override this method to implement the
        algorithm-specific action-selection logic.

        Args:
            obs (int): The current state (observation).

        Returns:
            int: The action selected by the agent.
        """
        raise NotImplementedError("The `act` method must be overridden by subclasses.")
