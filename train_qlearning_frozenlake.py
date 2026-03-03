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
    done = False
    total_reward = 0

    while not done:

        # Epsilon-greedy
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-learning update
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state, :]) - Q[state, action]
        )

        state = next_state
        total_reward += reward

    total_rewards[i] = total_reward

    # Exponential epsilon decay
    epsilon = max(0.01, epsilon * 0.995)


os.makedirs("results/frozenlake", exist_ok=True)

np.save("results/frozenlake/score_Qlearning_frozenlake.npy", total_rewards)

plt.plot(total_rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Q-Learning - FrozenLake")
plt.savefig("results/frozenlake/Qlearning_frozenlake.png")
plt.close()

print("\nQ-Learning FrozenLake Complete")
print("Success Rate (last 1000 episodes):", np.mean(total_rewards[-1000:]))