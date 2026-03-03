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

    done = False
    score = 0

    while not done:

        # Epsilon-greedy action
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = maxAction(Q, state)

        # Take step (Gymnasium API)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = getState(next_observation)

        score += reward

        # TRUE Q-LEARNING UPDATE (Off-policy)
        best_next_action = maxAction(Q, next_state)

        Q[(state, action)] += alpha * (
            reward
            + gamma * Q[(next_state, best_next_action)]
            - Q[(state, action)]
        )

        state = next_state

    total_score[i] = score

    # Linear epsilon decay
    epsilon = epsilon - 2 / episodes if epsilon > 0.01 else 0.01



os.makedirs("results/mountaincar", exist_ok=True)

save_obj(Q, "results/mountaincar/Q-table-Qlearning")
np.save("results/mountaincar/total_score_qlearning.npy", total_score)

plt.plot(total_score)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-Learning - MountainCar")
plt.savefig("results/mountaincar/qlearning_curve.png")
plt.close()

print("\nQ-Learning Training Complete")
print("Average reward (last 100 episodes):", np.mean(total_score[-100:]))