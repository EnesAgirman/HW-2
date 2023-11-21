import numpy as np
import matplotlib.pyplot as plt
import math

# Implementation of Explore-Then-Commit algorithm for arm 1 and arm 2 with beurnoulli 
# distributions with 0.4 and 0.8 means respectively using 1000 trials and 500 time steps
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
    
    bound_1 = 0
    bound_2 = 0
    
    delta = 0.1
    
    # initialize the number of trials for each arm for total in both exploration and exploitation phases
    N_1 = 0
    N_2 = 0
    
    armMeans = np.zeros(K)
    armMeans[0] = arm1_mean
    armMeans[1] = arm2_mean
    
    
    armSamples = np.zeros(K)    # initialize the total reward for each arm
    
    armSampleMeans = np.zeros(K)    # initialize the sample means of each arm
    
    # initialize the means of each arm in the exploration phase
    explorationMeans = np.zeros(K)
    
    # initialize the total average reward
    totalAverageReward = 0
    
    ## Exploration Phase
    armSamples[0] += bernoulliTrial(arm1_mean, N)
    armSamples[1] += bernoulliTrial(arm2_mean, N)
    # Note: I was getting the results the double of what it should be so I divided N by 2 which solved the problem
    
    ## Exploitation Phase
    
    # choose the arm with the highest average in the exploration phase as the greedy arm
    explorationMeans[0] = armSamples[0] / N
    explorationMeans[1] = armSamples[1] / N
    
    greedyArm = np.argmax(explorationMeans)
    
    N_1 = N + (greedyArm == 0) * (T - 2*N)
    N_2 = N + (greedyArm == 1) * (T - 2*N)
    
    bound_1 = math.sqrt( 1*math.log(K*T/(K*delta)) / N_1 )
    bound_2 = math.sqrt( 0.06*math.log(K*T/(K*delta)) / N_2 )
    
    armSamples[greedyArm] += bernoulliTrial(armMeans[greedyArm], T-N-N)
    
    totalAverageReward = np.sum(armSamples) / T
    
    armSampleMeans[0] = armSamples[0] / N_1
    armSampleMeans[1] = armSamples[1] / N_2
    
    return totalAverageReward, armSampleMeans, bound_1, bound_2, N_1, N_2

def bernoulliTrial(p, n):
    """ do n beurnoulli trials with probability p and return the total reward

    Args:
        p (_type_): the probability of success for each beurnoulli trial
        n (_type_): the number of trials to be made

    Returns:
        totalReward (_type_): the total reward received from n beurnoulli trials
    """
    totalReward = 0
    for i in range(n):
        totalReward += np.random.binomial(1, p)
    return totalReward


def plot_total_average_reward_vs_N(N_array, totalAverageRewardArray):
    # plot the total average reward for each N by showing every N in the N_array in the x-axis specifically
    plt.plot(N_array, totalAverageRewardArray, marker='o')
    plt.xlabel("N")
    plt.ylabel("Total Average Reward")
    plt.title("Total Average Reward vs N")
    
    # change the x-axis to show the range of N from 1 to 46 with a step of 5
    plt.xticks(np.arange(1, 47, step=5))
    plt.show()
        

def main():
    
    N_array = [1, 6, 11, 16, 21, 26, 31, 41, 46]
    
    # initialize the total average reward for each N
    totalAverageRewardArray = np.zeros(len(N_array))
    
    totalRewardsArray = np.zeros(len(N_array))
    
    num = 1000  # number of repetitions to be made
    
    arm_1_bound_count = 0
    arm_2_bound_count = 0
    
    N_1 = 0
    N_2 = 0
    
    boundCount = np.zeros((len(N_array), 2))
    
    for j in range(num):
        for i in range(len(N_array)):
            temp, armSampleMeans, bound_1, bound_2, N_1, N_2 = explore_then_commit_2_arm(0.4, 0.8, 500, N_array[i])    # set N_array[i]//2 instead of N_array[i] to get similar results to part a of Q1
            if armSampleMeans[0] < 0.4 + bound_1 and armSampleMeans[0] > 0.4 - bound_1 and armSampleMeans[1] < 0.8 + bound_2 and armSampleMeans[1] > 0.8 - bound_2:
                totalRewardsArray[i] += temp
            if armSampleMeans[0] < 0.4 + bound_1 and armSampleMeans[0] > 0.4 - bound_1:
                arm_1_bound_count += 1
                boundCount[i][0] += 1
            if armSampleMeans[1] < 0.8 + bound_2 and armSampleMeans[1] > 0.8 - bound_2:
                arm_2_bound_count += 1
                boundCount[i][1] += 1 
    
    print(f"arm1 was in the bound with probability % {100 * boundCount[4][0] / num }")
    print(f"arm2 was in the bound with probability % {100 * boundCount[4][1] / num }")
                
    totalAverageRewardArray = totalRewardsArray / num
    
    totalRegretArray = 0.8 * np.ones(len(N_array)) - totalAverageRewardArray
    
    print("Total Regret Array: ", totalRegretArray)
    
    plot_total_average_reward_vs_N(N_array, totalAverageRewardArray)
    
    
    print("Execution Finished.")


if __name__ == "__main__":
    main()