import gymnasium as gym
import numpy as np
import random
import os
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v1", is_slippery=True)

Q = np.zeros((env.observation_space.n, env.action_space.n))

alpha = 0.1
gamma = 0.99
epsilon = 1.0
episodes = 20000

total_rewards = np.zeros(episodes)

for i in range(episodes):

    if i % 1000 == 0:
        print(f"Episode {i}, epsilon: {epsilon:.3f}")

    state, _ = env.reset()

    # Initial action
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

        # SARSA update
        Q[state, action] += alpha * (
            reward + gamma * Q[next_state, next_action] - Q[state, action]
        )

        state = next_state
        action = next_action
        total_reward += reward

    total_rewards[i] = total_reward

    epsilon = max(0.01, epsilon * 0.995)

os.makedirs("results/frozenlake", exist_ok=True)

np.save("results/frozenlake/score_SARSA_frozenlake.npy", total_rewards)

plt.plot(total_rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("SARSA - FrozenLake")
plt.savefig("results/frozenlake/SARSA_frozenlake.png")
plt.close()

print("\nSARSA FrozenLake Complete")
print("Success Rate (last 1000 episodes):", np.mean(total_rewards[-1000:]))