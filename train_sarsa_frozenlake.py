import gymnasium as gym
import numpy as np
import random
import os
import matplotlib.pyplot as plt


env = gym.make("FrozenLake-v1", is_slippery=True)


Q = np.zeros((env.observation_space.n, env.action_space.n))


alpha = 0.1
gamma = 0.999
epsilon = 1.0
episodes = 20000
epsilon_decay_type = "exp"   # "linear_fast", "linear_slow", "exp"


total_rewards = np.zeros(episodes)

for i in range(episodes):

    if i % 1000 == 0:
        print(f"Episode {i}, epsilon: {epsilon:.3f}")

    state, _ = env.reset()

    # Initial action (epsilon-greedy)
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])

    done = False
    total_reward = 0

    while not done:

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Choose next action (on-policy)
        if random.uniform(0, 1) < epsilon:
            next_action = env.action_space.sample()
        else:
            next_action = np.argmax(Q[next_state, :])

        # SARSA update (On-policy)
        Q[state, action] += alpha * (
            reward + gamma * Q[next_state, next_action] - Q[state, action]
        )

        state = next_state
        action = next_action
        total_reward += reward

    total_rewards[i] = total_reward


    if epsilon_decay_type == "linear_fast":
        epsilon = epsilon - 2 / episodes if epsilon > 0.01 else 0.01

    elif epsilon_decay_type == "linear_slow":
        epsilon = epsilon - 1 / episodes if epsilon > 0.01 else 0.01

    elif epsilon_decay_type == "exp":
        epsilon = max(0.01, epsilon * 0.999)


os.makedirs("results/frozenlake", exist_ok=True)

experiment_name = f"Frozen_SARSA_a{alpha}_g{gamma}_{epsilon_decay_type}"

np.save(f"results/frozenlake/{experiment_name}.npy", total_rewards)

plt.plot(total_rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title(experiment_name)
plt.savefig(f"results/frozenlake/{experiment_name}.png")
plt.close()

print("\nExperiment:", experiment_name)
print("Success Rate (last 1000 episodes):", np.mean(total_rewards[-1000:]))