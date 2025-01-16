"""Example for R-Max with RiverSwim with mean and confidence interval."""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import riverswim  # noqa: F401
from agents import RMaxAgent
from scipy.stats import sem, t


discount = 0.95
epsilon1 = 0.1
m = 5
r_max = 100_000

max_steps = 10_000
n_seeds = 50
confidence_level = 0.95

env = gym.make(
    id="RiverSwim-v0",
    n_states=6,
    max_reward=r_max,
    intermediate_reward=5,
    commom_reward=0,
    p_right=0.3,
    p_left=0.1,
    random_initial_state=False,
    initial_state=0,
)

all_returns = []

for seed in range(n_seeds):
    agent = RMaxAgent(
        num_states=env.observation_space.n,
        num_actions=env.action_space.n,
        r_max=r_max,
        epsilon1=epsilon1,
        m=m,
        discount=discount,
        seed=seed,
    )

    state, _ = env.reset(seed=seed)
    returns = []
    total_discounted_reward = 0

    for step in range(max_steps):
        action = agent.act(state)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(state, action, next_obs, reward)
        state = next_obs

        total_discounted_reward += reward
        returns.append(total_discounted_reward)

        if terminated or truncated:
            break

    all_returns.append(returns)

all_returns = np.array(all_returns)

mean_returns = np.mean(all_returns, axis=0)
std_error = sem(all_returns, axis=0)
confidence_interval = t.interval(
    confidence_level, df=n_seeds - 1, loc=mean_returns, scale=std_error
)

plt.figure(figsize=(10, 6))
plt.plot(mean_returns, label="Mean Return", color="blue")
plt.fill_between(
    range(len(mean_returns)),
    confidence_interval[0],
    confidence_interval[1],
    color="blue",
    alpha=0.2,
    label=f"{int(confidence_level*100)}% Confidence Interval",
)
plt.xlabel("Steps")
plt.ylabel("Cumulative Reward")
plt.title("Mean Return with Confidence Interval (R-Max)")
plt.legend()
plt.grid()
plt.show()

print("Mean Returns:", mean_returns)
print(f"Confidence Interval ({confidence_level*100}%):", confidence_interval)
