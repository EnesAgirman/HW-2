import numpy as np
import matplotlib.pyplot as plt

# Implementation of Explore-Then-Commit algorithm for arm 1 and arm 2 with beurnoulli 
# distributions with 0.4 and 0.8 means respectively using 1000 trials and 1000 time steps
def explore_then_commit_2_arm(arm1_mean, arm2_mean, T, N):    
    """implement an Explore-Then-Commit algorithm for a 2-armed bandit problem with beurnoulli distributions
    Args:
        arm1_mean (_type_): mean value of the beurnoulli distribution for arm 1
        arm2_mean (_type_): mean value of the beurnoulli distribution for arm 2
        T (_type_): the total number of trials to be made  including both exploration and exploitation
        N (_type_): the total number of trials to be made for each arm in the exploration phase

    Returns:
        totalAverageReward: average of rewards received from both arms in the exploration and exploitation phases
    """
    
    K = 2   # number of arms
    
    # initialize the number of trials for each arm
    n = np.zeros(K)
    
    armMeans = np.zeros(K)
    
    # initialize the total reward for each arm
    totalReward = np.zeros(K)
    
    # initialize the total number of trials in total including both arms in exploration and exploitation phases
    numOfTrialsTotal = 0    
    
    # initialize the exploration means for each arm
    explorationMeans = np.zeros(K)
    
    # initialize the total average reward
    totalAverageReward = 0

    for i in range(N):
        arm_1_reward = np.random.binomial(1, arm1_mean)
        arm_2_reward = np.random.binomial(1, arm2_mean)
        
        totalReward[0] += arm_1_reward
        totalReward[1] += arm_2_reward

        numOfTrialsTotal += 2
    
    explorationMeans[0] = totalReward[0] / (N)
    explorationMeans[1] = totalReward[1] / (N)
    
    greedyArm = np.argmax(explorationMeans)
    
    for j in range(T-2*N):
        totalReward[greedyArm] += np.random.binomial(1, armMeans[greedyArm])
        numOfTrialsTotal += 1
    
    totalAverageReward = np.sum(totalReward) / numOfTrialsTotal
    
    return totalAverageReward

def plot_total_average_reward_vs_N(N_array, totalAverageRewardArray):
    # plot the total average reward for each N by showing every N in the N_array in the x-axis specifically
    plt.plot(N_array, totalAverageRewardArray)
    plt.xlabel("N")
    plt.ylabel("Total Average Reward")
    plt.title("Total Average Reward vs N")
    
    # change the x-axis to show the range of N from 1 to 46 with a step of 5
    plt.xticks(np.arange(1, 91, step=5))
    plt.show()

def main():
    
    N_array = [1, 6, 11, 16, 21, 26, 31, 41, 46, 90]
    
    # initialize the total average reward for each N
    totalAverageRewardArray = np.zeros(len(N_array))
    
    totalRewardsArray = np.zeros(len(N_array))
    
    numberOfExperiments = 1000
    
    for j in range(numberOfExperiments):
        for i in range(len(N_array)):
            totalRewardsArray[i] += explore_then_commit_2_arm(0.4, 0.8, 500, N_array[i])
    
    totalAverageRewardArray = totalRewardsArray / numberOfExperiments
        
    plot_total_average_reward_vs_N(N_array, totalAverageRewardArray)
    
    
    
    print("Execution Finished.")


if __name__ == "__main__":
    main()