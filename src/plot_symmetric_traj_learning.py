import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV files
file1 = '/Users/tomwoodley/Desktop/TommyWoodleyMEngProject/src/models/non-symmetric-approaching_2024_06_12_11_30/logs.monitor.csv'
file2 = '/Users/tomwoodley/Desktop/TommyWoodleyMEngProject/src/models/symmetric-approaching_2024_06_12_09_13/logs.monitor.csv'

# Read the CSV files
df_non_symmetric = pd.read_csv(file1, skiprows=1)
df_symmetric = pd.read_csv(file2, skiprows=1)

print("Non-Symmetric DataFrame:")
print(df_non_symmetric.head())

print("\nSymmetric DataFrame:")
print(df_symmetric.head())

# Extract the 'l' column (assuming it is named 'l')
trajectory_non_symmetric = df_non_symmetric['l']
trajectory_symmetric = df_symmetric['l']

rolling_non_symmetric = trajectory_non_symmetric.rolling(window=50).mean()
rolling_symmetric = trajectory_symmetric.rolling(window=50).mean()

# Plot the trajectories
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

plt.plot(rolling_non_symmetric, label='Non-Symmetric')
plt.plot(rolling_symmetric, label='Symmetric')

# Set y-axis limit
plt.ylim(30, 100)

# Set x-axis limit
plt.xlim(50, 400)

# Add labels and title
plt.xlabel('Episodes')
plt.ylabel('Episode Length (timesteps)')
plt.title('Comparison of Average Trajectory Length During Training')
plt.legend()

# Show the plot
save_path = '/Users/tomwoodley/Desktop/TommyWoodleyMEngProject/src/symmetric-learning.png'
plt.savefig(save_path, dpi=500)
plt.show()
