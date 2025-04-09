"""Implementation of MBIE and MBIE-EB Agent."""

import numpy as np
from agents import BaseAgent
from gymnasium.utils import seeding


class MBIEAgent(BaseAgent):
    """A Model-Based Interval Estimation (MBIE) with Exploration Bonus (MBIE-EB) agent.

    This implementation is based on Strehl and Littman’s "An analysis of model-based Interval Estimation
    for Markov Decision Processes" (J. Computer and System Sciences, 2008).
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        r_max: float,
        discount: float = 0.95,
        epsilon_r_coeff: float = 0.3,
        epsilon_t_coeff: float = 0.0,
        exploration_coeff: float = 0.4,
        epsilon: float = 0.01,
        m: int = 16,
        use_exploration_bonus: bool = True,
        seed: int | None = None,
    ):
        """Initializes the MBIE agent with the given parameters.

        Args:
            num_states (int): Total number of states.
            num_actions (int): Total number of actions.
            r_max (float, optional): Maximum possible reward for unknown state-action pairs.
            discount (float): Discount factor (γ).
            epsilon_r_coeff (float, optional): Reward epsilon. Defaults to 0.3.
            epsilon_t_coeff (float, optional): Transition epsilon. Defaults to 0.0.
            exploration_coeff (float, optional): Exploration coefficient defined as beta / r_max. Defaults to 0.4.
            epsilon (float, optional): Number of iterations the optimal policy will return near-optimal value (U(pi) - epsilon) on average. Defaults to 0.01.
            m (int, optional): Minimum number of visits required to consider a state-action pair as known. Defaults to 16.
            use_exploration_bonus (bool, optional): _description_. Defaults to True.
            seed (int, optional): Random seed.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.discount = discount
        self.r_max = r_max
        self.epsilon_r_coeff = epsilon_r_coeff
        self.epsilon_t_coeff = epsilon_t_coeff
        self.epsilon = epsilon
        self.use_exploration_bonus = use_exploration_bonus

        self.Q = np.full(
            (num_states, num_actions), r_max / (1 - discount), dtype=np.float32
        )

        self.sa_counts = np.zeros((num_states, num_actions), dtype=np.int32)
        self.reward_sums = np.zeros((num_states, num_actions), dtype=np.float32)
        self.trans_counts = np.zeros(
            (num_states, num_actions, num_states), dtype=np.int32
        )

        # See Section 6
        self.m = m  # np.inf
        self.beta = exploration_coeff * r_max

        self.np_random, _ = seeding.np_random(seed)

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
        """Returns the empirical mean reward for (s,a); if unknown, returns r_max."""
        count = self.sa_counts[obs, action]

        if count > 0:
            # See Section 6
            return self.reward_sums[
                obs, action
            ] / count, self.epsilon_r_coeff * self.r_max / np.sqrt(count)
        else:
            return self.r_max, np.inf

    def get_transition_estimate(self, obs, action):
        """Returns the maximum-likelihood estimate of the transition distribution for (s,a).

        If (s,a) is unknown, returns a uniform distribution.
        """
        count = self.sa_counts[obs, action]

        if count > 0:
            # See Section 6
            return self.trans_counts[
                obs, action
            ] / count, self.epsilon_t_coeff / np.sqrt(count)
        else:
            return np.ones(self.num_states) / float(self.num_states), np.inf

    def optimistic_transition_value(
        self, T_hat: np.ndarray, V: np.ndarray, epsilon_T: float
    ) -> float:
        """Given the empirical next-state distribution T_hat for (s,a), returns the maximum achievable value ∑ₛ′ T̃(s′)*V(s′) under any distribution T̃ satisfying ||T̃ - T_hat||₁ ≤ ε_T.

        Procedure (following the paper): start with T_tilde = T_hat. Increase the mass at the state
        with maximum V by ε_T/2 and then subtract exactly ε_T/2 mass from states with the lowest V values.
        Finally, normalize T_tilde to be a valid probability distribution.
        """
        T_tilde = T_hat.copy()
        extra = epsilon_T / 2.0
        s_star = np.argmax(V)
        T_tilde[s_star] += extra  # add extra mass to best state

        # Now T_tilde sums to (1 + extra). Remove extra mass from states with low value.
        remaining = extra
        indices = np.argsort(V)  # indices from lowest to highest V
        for s in indices:
            if s == s_star:
                continue
            if remaining <= 0:
                break
            remove_mass = min(T_tilde[s], remaining)
            T_tilde[s] -= remove_mass
            remaining -= remove_mass

        # Normalize to ensure a proper probability distribution (numerical issues may appear)
        T_tilde = np.clip(T_tilde, 0, None)
        T_tilde = T_tilde / np.sum(T_tilde)
        return np.dot(T_tilde, V)

    def update(self, obs: int, action: int, next_obs: int, reward: float):
        """Updates the internal model with a new experience.

        Increments visit counts for the given state-action pair, adds
        the observed reward, and updates transition counts.
        """
        if self.sa_counts[obs, action] < self.m:
            self.sa_counts[obs, action] += 1
            self.reward_sums[obs, action] += reward
            self.trans_counts[obs, action, next_obs] += 1

            max_iters = int(
                np.ceil(
                    np.log(1 / (self.epsilon * (1 - self.discount)))
                    / (1 - self.discount)
                )
            )
            for _ in range(max_iters):
                Q_new = np.empty_like(self.Q)
                V = self.Q.max(axis=1)  # state values
                for s in range(self.num_states):
                    for a in range(self.num_actions):
                        count = self.sa_counts[s, a]
                        if count > 0:
                            r_hat, r_conf = self.get_reward_estimate(s, a)
                            R_upper = r_hat + r_conf

                            T_hat, t_conf = self.get_transition_estimate(s, a)

                            if self.use_exploration_bonus:
                                Q_new[s, a] = (
                                    r_hat
                                    + self.discount * np.dot(T_hat, V)
                                    + self.beta / np.sqrt(count)
                                )
                            else:
                                opt_val = self.optimistic_transition_value(
                                    T_hat, V, t_conf
                                )
                                Q_new[s, a] = R_upper + self.discount * opt_val
                        else:
                            Q_new[s, a] = self.r_max / (1 - self.discount)
                if np.max(np.abs(Q_new - self.Q)) < 1e-2:
                    self.Q = Q_new
                    break
            self.Q = Q_new

    def act(self, obs: int) -> int:
        """Returns an action chosen greedily with respect to the current Q–values for state obs."""
        q_vals = self.Q[obs]
        best = np.argwhere(q_vals == np.max(q_vals)).flatten()
        return int(self.np_random.choice(best))
