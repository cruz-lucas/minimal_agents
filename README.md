# Minimal Agents

This repository contains a **minimal implementation** of tabular reinforcement-learning agents powered by [JAX](https://github.com/google/jax). It ships with Q-learning (including an intrinsic-reward variant), SARSA, R-MAX and MBIE-EB agents, together with a policy module featuring random-walk, epsilon-greedy and UCB exploration strategies. The code is intentionally concise and aims to serve both as a baseline and a reference for more advanced work.

## Table of Contents
1. [Installation](#installation)
    - [Installing via uv](#1-installing-via-uv)
    - [Installing via pip](#2-installing-via-pip)
2. [Usage](#usage)
3. [Contributing](#contributing)
4. [License](#license)


## Installation

You can install this package using either **pip** or **uv**.
**Note**: If you're unfamiliar with `uv`, you can skip directly to the `pip` instructions.

### 1. Installing via `uv`

   ```bash
   uv add https://github.com/cruz-lucas/minimal_agents.git
   ```
   This should handle the necessary dependencies and set up the virtual environment if you have a `pyproject.toml` file, if not, see use `uv init`.

### 2. Installing via `pip`

   ```bash
   pip install git+https://github.com/cruz-lucas/minimal_agents.git
   ```

This command installs the RiverSwim environment into your Python environment (**consider using a virtual environment**).

## Usage

Once installed, import the agents and policies directly from the package:

```python
from minimal_agents.agents import QLearningAgent
from minimal_agents.policies import EpsilonGreedyPolicy

agent = QLearningAgent(
    num_states=10,
    num_actions=4,
    learning_rate=0.2,
    epsilon=0.1,
    discount=0.99,
    seed=0,
)

# Configure an alternative policy if required.
agent.set_policy(EpsilonGreedyPolicy(epsilon=0.05))

state = 0
action = agent.select_action(state)
transition = env.step(action)  # user-defined environment interaction
agent.update(
    obs=state,
    action=action,
    reward=transition.reward,
    next_obs=transition.next_state,
    done=transition.terminated,
)
```

Consult the docstrings for each agent to learn about the available diagnostics (e.g. TD errors) and policy hooks.

### Examples

- [`examples/cartpole_q_learning.py`](./examples/cartpole_q_learning.py) trains a Q-learning agent with simple discretisation on Gymnasium's publicly available `CartPole-v1`.

### Running Tests

After installing the project dependencies, run the unit suite from the repository root:

```bash
pytest
```

If you use [`uv`](https://github.com/astral-sh/uv) to manage environments, the equivalent command is:

```bash
uv run pytest
```

## Contributing

Contributions and suggestions to improve this minimal implementation are always welcome. Feel free to open an issue or a pull request.


## License

This project is licensed under the [MIT License](LICENSE). Please see the [LICENSE](LICENSE) file for more information.
