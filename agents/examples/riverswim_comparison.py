"""Comparison between model-based planning algorithms in riverswim."""

import concurrent.futures

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import riverswim  # noqa: F401
from agents import MBIEAgent, RMaxAgent
from scipy.stats import sem, t


# Global experimental parameters
DISCOUNT = 0.95
R_MAX = 10_000
M = 16
MAX_STEPS = 5_000
N_SEEDS = 50
CONFIDENCE_LEVEL = 0.95

# MBIE-specific parameters
EPSILON = 0.1
EPSILON_R_COEFF = 0.3
EPSILON_T_COEFF = 0.0
EXPLORATION_COEFF = 0.4

# R-Max-specific parameter
EPSILON1 = 0.1


def run_rmax(seed):
    """Runs a single simulation episode for the R-Max agent.

    Returns the final cumulative reward.
    """
    env = gym.make(
        id="RiverSwim-v0",
        n_states=6,
        max_reward=R_MAX,
        intermediate_reward=5,
        commom_reward=0,
        p_right=0.3,
        p_left=0.1,
    )
    agent = RMaxAgent(
        num_states=env.observation_space.n,
        num_actions=env.action_space.n,
        r_max=R_MAX,
        epsilon1=EPSILON1,
        m=M,
        discount=DISCOUNT,
        seed=seed,
    )
    state, _ = env.reset(seed=seed)
    total_reward = 0.0

    for step in range(MAX_STEPS):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.update(state, action, next_state, reward)
        state = next_state
        total_reward += reward
        if terminated or truncated:
            break
    env.close()
    return total_reward


def run_mbie(seed, use_exploration_bonus):
    """Runs a single simulation episode for the MBIE agent.

    The flag 'use_exploration_bonus' toggles the exploration bonus.
    Returns the final cumulative reward.
    """
    env = gym.make(
        id="RiverSwim-v0",
        n_states=6,
        max_reward=R_MAX,
        intermediate_reward=5,
        commom_reward=0,
        p_right=0.3,
        p_left=0.1,
    )
    agent = MBIEAgent(
        num_states=env.observation_space.n,
        num_actions=env.action_space.n,
        r_max=R_MAX,
        m=M,
        epsilon=EPSILON,
        discount=DISCOUNT,
        epsilon_r_coeff=EPSILON_R_COEFF,
        epsilon_t_coeff=EPSILON_T_COEFF,
        exploration_coeff=EXPLORATION_COEFF,
        use_exploration_bonus=use_exploration_bonus,
        seed=seed,
    )
    state, _ = env.reset(seed=seed)
    total_reward = 0.0

    for step in range(MAX_STEPS):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.update(state, action, next_state, reward)
        state = next_state
        total_reward += reward
        if terminated or truncated:
            break
    env.close()
    return total_reward


def run_mbie_bonus(seed):
    """Runs MBIE-EB."""
    return run_mbie(seed, use_exploration_bonus=True)


def run_mbie_no_bonus(seed):
    """Runs MBIE."""
    return run_mbie(seed, use_exploration_bonus=False)


def compute_confidence_interval(data, confidence_level):
    """Computes the mean and the margin of error (half the confidence interval width) for a given dataset."""
    mean_val = np.mean(data)
    std_err = sem(data)
    ci = t.interval(confidence_level, len(data) - 1, loc=mean_val, scale=std_err)
    error_margin = (ci[1] - ci[0]) / 2
    return mean_val, error_margin, ci


def main():
    """Compares algorithms in Riverswim."""
    configurations = [
        ("MBIE", run_mbie_no_bonus),
        ("MBIE-EB", run_mbie_bonus),
        ("R-Max", run_rmax),
    ]

    results = {}

    for name, sim_function in configurations:
        print(f"Running configuration: {name}")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            final_rewards = list(executor.map(sim_function, range(N_SEEDS)))
        results[name] = final_rewards

        mean_val, margin, ci = compute_confidence_interval(
            final_rewards, CONFIDENCE_LEVEL
        )
        print(f"{name}: Mean Final Reward = {mean_val:.2f}, 95% CI = {ci}")

    config_names = []
    means = []
    error_margins = []
    for name, rewards in results.items():
        mean_val, margin, _ = compute_confidence_interval(rewards, CONFIDENCE_LEVEL)
        config_names.append(name)
        means.append(mean_val)
        error_margins.append(margin)

    plt.figure(figsize=(8, 6))
    x_positions = np.arange(len(config_names))
    plt.bar(
        x_positions, means, yerr=error_margins, align="center", alpha=0.7, capsize=10
    )
    plt.xticks(x_positions, config_names)
    plt.ylabel("Cumulative Reward")
    plt.xlabel("Algorithm")
    plt.title("RiverSwim MDP")
    plt.show()


if __name__ == "__main__":
    main()
