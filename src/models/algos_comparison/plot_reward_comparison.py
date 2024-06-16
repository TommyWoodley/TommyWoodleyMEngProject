import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths to the CSV files
path_a2c = './approaching-a2c_2024_06_16_02_56/logs.monitor.csv'
path_ppo = './approaching-ppo_2024_06_16_02_39/logs.monitor.csv'
path_sac = './approaching-sac_2024_06_16_02_10/logs.monitor.csv'
path_td3 = './approaching-td3_2024_06_16_03_16/logs.monitor.csv'

# Function to read and smooth rewards
def read_and_smooth(path, window=70):
    data = pd.read_csv(path, skiprows=1)  # Skip the metadata row
    rewards = data['r'][0:500]
    smoothed_rewards = rewards.rolling(window=window, min_periods=1).mean()
    return smoothed_rewards

# Read and smooth rewards for each algorithm
rewards_a2c = read_and_smooth(path_a2c)
rewards_ppo = read_and_smooth(path_ppo)
rewards_sac = read_and_smooth(path_sac)
rewards_td3 = read_and_smooth(path_td3)

# Set the plot style
sns.set(style='whitegrid')

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(rewards_a2c, label='A2C')
plt.plot(rewards_ppo, label='PPO')
plt.plot(rewards_sac, label='SAC')
plt.plot(rewards_td3, label='TD3')

# Adding title and labels
plt.title('Comparison of RL Algorithms During The Approaching and Wrapping Stage', fontsize=15)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)
plt.savefig("algorithm_comparison.png", dpi=500)
plt.show()
