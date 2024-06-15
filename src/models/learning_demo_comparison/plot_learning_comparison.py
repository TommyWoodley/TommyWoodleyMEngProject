import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def smooth_data(data, window_size=100):
    return data.rolling(window=window_size).mean()

plotting_length = 1200

# Load data from CSV files
data_5_demos = pd.read_csv('training_5_demos_2024_06_14_00_16/logs.monitor.csv', skiprows=1)
data_1_demo = pd.read_csv('training_1_demo_2024_06_14_02_02/logs.monitor.csv', skiprows=1)
data_0_demos = pd.read_csv('training_0_demo_2024_06_14_00_26/logs.monitor.csv', skiprows=1)

# Extract the rewards and timesteps
rewards_5_demos = data_5_demos['r'][:plotting_length]
rewards_1_demo = data_1_demo['r'][:plotting_length]
rewards_0_demos = data_0_demos['r'][:plotting_length]


# Apply smoothing
smoothed_rewards_5_demos = smooth_data(rewards_5_demos)
smoothed_rewards_1_demo = smooth_data(rewards_1_demo)
smoothed_rewards_0_demos = smooth_data(rewards_0_demos)

# Set seaborn style
sns.set_theme(style="whitegrid")

# Plotting the data
plt.figure(figsize=(10, 6))

plt.plot(smoothed_rewards_5_demos, label='5 Demonstrations', linewidth=3)
plt.plot(smoothed_rewards_1_demo, label='1 Demonstration', linewidth=3)
plt.plot(smoothed_rewards_0_demos, label='0 Demonstrations', linewidth=3)
plt.xlim(100, 1200)

plt.xlabel('Timesteps')
plt.ylabel('Rewards')
plt.title('Comparative Learning of Tensile Perching from Demonstrations', fontsize=15)
plt.legend(loc='best')
plt.grid(True)

plt.savefig('comparative_learning.png', dpi=500)

plt.show()