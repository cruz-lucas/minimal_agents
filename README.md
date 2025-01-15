# Tabular Q-Learning with Epsilon-Greedy Exploration

This repository contains a **minimal implementation** of a tabular Q-learning agent using an epsilon-greedy exploration strategy. The code is provided in `qlearning.py`, and can be used as a simple baseline or a starting point for more advanced algorithms.

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
    - [Installing via uv](#1-installing-via-uv)
    - [Installing via pip](#2-installing-via-pip)
3. [Usage](#usage)
4. [Contributing](#contributing)
5. [License](#license)

## Overview

Q-learning is an off-policy Temporal Difference (TD) control algorithm for Reinforcement Learning. It learns an action-value function $Q(s,a)$ that gives the expected return (discounted sum of rewards) when taking action $a$ in state $s$, and thereafter following the current best action-value estimate.

### Features of This Implementation
- **Tabular**: The state-action space is represented by a 2D array $Q$ of shape `(num_states, num_actions)`.
- **Epsilon-Greedy Exploration**: With probability $\epsilon$, the agent chooses a random action. Otherwise, it chooses the action with the highest Q-value.
- **Parameterizable**: You can configure:
  - Learning rate $\alpha$
  - Epsilon $\epsilon$
  - Discount factor $\gamma$
- **Minimal Dependencies**: Requires only `numpy` and `gymnasium` utilities for random seeding.


## Installation

You can install this package using either **pip** or **uv** (a minimal package manager/distribution manager example in Python).
**Note**: If you're unfamiliar with `uv`, you can skip directly to the `pip` instructions.

### 1. Installing via `uv`

1. Clone this repository:
   ```bash
   git clone https://github.com/cruz-lucas/riverswim.git
   cd riverswim
   ```
2. Install using `uv`:
   ```bash
   uv pip install .
   ```
   This command should handle the necessary dependencies and set up a virtual environment.

### 2. Installing via `pip`

1. Clone this repository:
   ```bash
   git clone https://github.com/cruz-lucas/qlearning.git
   cd qlearning
   ```
2. Install the package locally:
   ```bash
   pip install .
   ```

## Usage

Once installed, you can use the Q-learning agent in your Python code. Below is a minimal example of using this agent in a Gymnasium (Gym) environment:

```python
import gymnasium as gym
import matplotlib.pyplot as plt
import riverswim

discount = 0.99
epsilon = 1
alpha = 0.2

# env = gym.make("FrozenLake-v1", is_slippery=True)
env = gym.make(id='RiverSwim-v0', )

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
```

Feel free to modify the hyperparameters ($\epsilon$, $\alpha$, $\gamma$) to experiment and see how they affect learning performance.


## Contributing

Contributions and suggestions to improve this minimal implementation are always welcome. Feel free to open an issue or a pull request.


## License

This project is licensed under the [MIT License](LICENSE). Please see the [LICENSE](LICENSE) file for more information.
