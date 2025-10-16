"""Train a tabular Q-learning agent on Gymnasium's CartPole-v1."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from minimal_agents.agents import QLearningAgent
from minimal_agents.policies import EpsilonGreedyPolicy


@dataclass(frozen=True)
class Discretiser:
    """Maps continuous CartPole observations into a discrete state index."""

    bins: tuple[np.ndarray, ...]

    @classmethod
    def build(cls) -> "Discretiser":
        # Bin cart position, cart velocity, pole angle, and pole angular velocity.
        cart_position = np.linspace(-2.4, 2.4, 9)
        cart_velocity = np.linspace(-3.0, 3.0, 9)
        pole_angle = np.linspace(-0.2, 0.2, 9)
        pole_velocity = np.linspace(-2.5, 2.5, 9)
        return cls((cart_position, cart_velocity, pole_angle, pole_velocity))

    def num_states(self) -> int:
        sizes = [len(edges) + 1 for edges in self.bins]
        return int(np.prod(sizes))

    def encode(self, observation: np.ndarray) -> int:
        bucket_indices = [
            int(np.digitize(observation[i], edges)) for i, edges in enumerate(self.bins)
        ]
        sizes = [len(edges) + 1 for edges in self.bins]
        return int(np.ravel_multi_index(bucket_indices, sizes))


def run_episode(
    env: gym.Env,
    agent: QLearningAgent,
    discretiser: Discretiser,
    max_steps: int = 500,
) -> int:
    obs, _ = env.reset()
    state = discretiser.encode(obs)
    total_reward = 0

    for _ in range(max_steps):
        action = agent.select_action(state)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = discretiser.encode(next_obs)

        agent.update(
            obs=state,
            action=action,
            reward=reward,
            next_obs=next_state,
            done=terminated or truncated,
        )

        total_reward += reward
        state = next_state
        if terminated or truncated:
            break

    return total_reward


def main() -> None:
    env = gym.make("CartPole-v1")
    discretiser = Discretiser.build()

    agent = QLearningAgent(
        num_states=discretiser.num_states(),
        num_actions=env.action_space.n,
        learning_rate=0.1,
        epsilon=0.1,
        discount=0.99,
        seed=0,
        policy=EpsilonGreedyPolicy(epsilon=0.1),
    )

    episodes = 55_000
    rewards = []
    for episode in range(episodes):
        reward = run_episode(env, agent, discretiser)
        rewards.append(reward)

        if (episode + 1) % 1_000 == 0:
            moving_average = np.mean(rewards[-100:])
            print(
                f"Episode {episode + 1:03d} | "
                f"Reward {reward:4.0f} | "
                f"Moving Avg (20) {moving_average:5.2f}"
            )

        # Simple epsilon decay over time.
        new_epsilon = max(0.01, agent.epsilon * 0.995)
        agent.set_epsilon(new_epsilon)

    env.close()
    print(
        f"Training complete. Average reward over last 20 episodes: "
        f"{np.mean(rewards[-20:]):.2f}"
    )


if __name__ == "__main__":
    main()
