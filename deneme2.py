import numpy as np
import matplotlib.pyplot as plt

# Function to simulate Explore & Commit Algorithm for a specific exploration length (N)
def explore_commit_algorithm(N, num_trials=1000, T=500, true_means=[0.4, 0.8]):
    total_rewards = np.zeros(num_trials)

    for trial in range(num_trials):
        # Placeholder code for Explore & Commit Algorithm
        arm_rewards = np.zeros(2)  # Placeholder for storing rewards for each arm
        chosen_arms = []  # Store the arms chosen during exploration phase

        for t in range(N):
            # Exploration phase
            arm = np.random.choice(2)  # Placeholder - randomly choose an arm
            chosen_arms.append(arm)
            reward = np.random.binomial(1, true_means[arm])  # Placeholder - sample reward from chosen arm
            arm_rewards[arm] += reward

        # Commit to the arm with the highest observed reward during exploration
        chosen_arm = np.argmax(arm_rewards)
        for t in range(N, T):
            # Exploitation phase
            reward = np.random.binomial(1, true_means[chosen_arm])  # Placeholder - sample reward from chosen arm
            total_rewards[trial] += reward

    avg_total_rewards = np.mean(total_rewards)
    return avg_total_rewards

# Specify exploration lengths (N)
exploration_lengths = [1, 6, 11, 16, 21, 26, 31, 41, 46]

# Simulate and plot results
average_rewards = []
for N in exploration_lengths:
    avg_rewards = explore_commit_algorithm(N)
    average_rewards.append(avg_rewards)

# Plotting
plt.plot(exploration_lengths, average_rewards, marker='o')
plt.xlabel('Exploration Length (N)')
plt.ylabel('Average Total Rewards')
plt.title('Explore & Commit Algorithm Performance')
plt.xticks(np.arange(1, 47, step=5))
plt.show()