from auxFunctions import getState, createEmptyQTable, maxAction, save_obj
import gymnasium as gym
import random
import numpy as np
import os
import matplotlib.pyplot as plt

# Create environment
env = gym.make('MountainCar-v0')
env._max_episode_steps = 1000

# Create empty Q-table (dictionary)
Q = createEmptyQTable()

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 1.0
episodes = 50000

total_score = np.zeros(episodes)
score = 0

for i in range(episodes):

    if i % 500 == 0:
        print(f"episode: {i}, score: {score}, epsilon: {epsilon:.3f}")

    observation, _ = env.reset()
    state = getState(observation)

    # Initial action (epsilon-greedy)
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = maxAction(Q, state)

    done = False
    score = 0

    while not done:

        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = getState(next_observation)

        # Choose next action (on-policy)
        if random.uniform(0, 1) < epsilon:
            next_action = env.action_space.sample()
        else:
            next_action = maxAction(Q, next_state)

        score += reward

        # SARSA UPDATE (On-policy)
        Q[(state, action)] += alpha * (
            reward
            + gamma * Q[(next_state, next_action)]
            - Q[(state, action)]
        )

        state = next_state
        action = next_action

    total_score[i] = score

    epsilon = epsilon - 2 / episodes if epsilon > 0.01 else 0.01



os.makedirs("results/mountaincar", exist_ok=True)

save_obj(Q, "results/mountaincar/Q-table-SARSA")
np.save("results/mountaincar/total_score_sarsa.npy", total_score)

plt.plot(total_score)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("SARSA - MountainCar")
plt.savefig("results/mountaincar/sarsa_curve.png")
plt.close()

print("\nSARSA Training Complete")
print("Average reward (last 100 episodes):", np.mean(total_score[-100:]))