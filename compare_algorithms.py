import numpy as np
import matplotlib.pyplot as plt

# Load BEST saved results
q_scores = np.load("results/mountaincar/score_Qlearning_best.npy")
sarsa_scores = np.load("results/mountaincar/score_SARSA_best.npy")

# Plot both curves
plt.plot(q_scores, label="Q-Learning")
plt.plot(sarsa_scores, label="SARSA")

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-Learning vs SARSA (Tuned - MountainCar)")
plt.legend()

plt.savefig("results/mountaincar/Q_vs_SARSA_best_comparison.png")
plt.show()

print("Q-Learning Final Avg (last 100):", np.mean(q_scores[-100:]))
print("SARSA Final Avg (last 100):", np.mean(sarsa_scores[-100:]))