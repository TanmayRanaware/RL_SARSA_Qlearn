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

        # Epsilon-greedy action
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = maxAction(Q, state)

        # Take step
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = getState(next_observation)

        score += reward

        # TRUE Q-learning update
        best_next_action = maxAction(Q, next_state)

        Q[(state, action)] += alpha * (
            reward
            + gamma * Q[(next_state, best_next_action)]
            - Q[(state, action)]
        )

        state = next_state

    total_score[i] = score

    
    if epsilon_decay_type == "linear_fast":
        epsilon = epsilon - 2 / episodes if epsilon > 0.01 else 0.01

    elif epsilon_decay_type == "linear_slow":
        epsilon = epsilon - 1 / episodes if epsilon > 0.01 else 0.01

    elif epsilon_decay_type == "exp":
        epsilon = max(0.01, epsilon * 0.995)


os.makedirs("results/mountaincar", exist_ok=True)

experiment_name = f"Qlearning_a{alpha}_g{gamma}_{epsilon_decay_type}"

save_obj(Q, f"results/mountaincar/{experiment_name}")
np.save(f"results/mountaincar/score_{experiment_name}.npy", total_score)

plt.plot(total_score)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title(f"Q-Learning ({experiment_name})")
plt.savefig(f"results/mountaincar/{experiment_name}.png")
plt.close()


final_avg = np.mean(total_score[-100:])

print("\nQ-Learning Training Complete")
print("Experiment:", experiment_name)
print("Average reward (last 100 episodes):", final_avg)