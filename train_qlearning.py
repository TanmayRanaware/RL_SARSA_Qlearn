from auxFunctions import getState, createEmptyQTable, maxAction, save_obj
import gymnasium as gym
import random
import numpy as np
import os
import matplotlib.pyplot as plt


env = gym.make('MountainCar-v0')
env._max_episode_steps = 1000


Q = createEmptyQTable()


alpha = 0.05
gamma = 0.99
epsilon = 1.0
episodes = 50000
epsilon_decay_type = "exp"


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

        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = maxAction(Q, state)

        # Take step
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = getState(next_observation)

        score += reward

        # Q-learning update (Off-policy)
        best_next_action = maxAction(Q, next_state)

        Q[(state, action)] += alpha * (
            reward
            + gamma * Q[(next_state, best_next_action)]
            - Q[(state, action)]
        )

        state = next_state

    total_score[i] = score

    # Exponential epsilon decay
    epsilon = max(0.01, epsilon * 0.995)


os.makedirs("results/mountaincar", exist_ok=True)

# Save Q-table
save_obj(Q, "results/mountaincar/Qlearning_best")

# Save score array (FOR COMPARISON)
np.save("results/mountaincar/score_Qlearning_best.npy", total_score)

# Save plot
plt.plot(total_score)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-Learning Best (alpha=0.05, gamma=0.99, exp decay)")
plt.savefig("results/mountaincar/Qlearning_best_curve.png")
plt.close()

# Final metric
final_avg = np.mean(total_score[-100:])

print("\nQ-Learning (Best) Training Complete")
print("Average reward (last 100 episodes):", final_avg)