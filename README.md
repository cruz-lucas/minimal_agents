# Minimal Agents

This repository contains a **minimal implementation** of a tabular Q-learning, SARSA and R-max agents (other types of agents to be included in the future). The agents are pretty simple and can be used as baselines or a starting point for more advanced algorithms.

## Table of Contents
1. [Installation](#installation)
    - [Installing via uv](#1-installing-via-uv)
    - [Installing via pip](#2-installing-via-pip)
2. [Usage](#usage)
3. [Contributing](#contributing)
4. [License](#license)


## Installation

You can install this package using either **pip** or **uv** (a minimal package manager/distribution manager example in Python).
**Note**: If you're unfamiliar with `uv`, you can skip directly to the `pip` instructions.

### 1. Installing via `uv`

1. Clone this repository:
   ```bash
   git clone https://github.com/cruz-lucas/minimal_agents.git
   cd minimal_agents
   ```
2. Install using `uv`:
   ```bash
   uv sync
   ```
   This command should handle the necessary dependencies and set up a virtual environment.

### 2. Installing via `pip`

1. Clone this repository:
   ```bash
   git clone https://github.com/cruz-lucas/minimal_agents.git
   cd minimal_agents
   ```
2. Install the package locally:
   ```bash
   pip install .
   ```

## Usage

Once installed, you can use the agents in your Python code. You can find examples of usage in the [examples](./agents/examples/).

## Contributing

Contributions and suggestions to improve this minimal implementation are always welcome. Feel free to open an issue or a pull request.


## License

This project is licensed under the [MIT License](LICENSE). Please see the [LICENSE](LICENSE) file for more information.
