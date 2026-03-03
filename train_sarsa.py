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

        # Step
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = getState(next_observation)

        # Choose next action (on-policy)
        if random.uniform(0, 1) < epsilon:
            next_action = env.action_space.sample()
        else:
            next_action = maxAction(Q, next_state)

        score += reward

        # SARSA update (On-policy)
        Q[(state, action)] += alpha * (
            reward
            + gamma * Q[(next_state, next_action)]
            - Q[(state, action)]
        )

        state = next_state
        action = next_action

    total_score[i] = score

    # Exponential epsilon decay
    epsilon = max(0.01, epsilon * 0.995)


os.makedirs("results/mountaincar", exist_ok=True)

# Save Q-table
save_obj(Q, "results/mountaincar/SARSA_best")

# Save score array (FOR COMPARISON)
np.save("results/mountaincar/score_SARSA_best.npy", total_score)

# Save plot
plt.plot(total_score)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("SARSA Best (alpha=0.05, gamma=0.99, exp decay)")
plt.savefig("results/mountaincar/SARSA_best_curve.png")
plt.close()

# Final metric
final_avg = np.mean(total_score[-100:])

print("\nSARSA (Best) Training Complete")
print("Average reward (last 100 episodes):", final_avg)