import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the CSV data
df = pd.read_csv('/Users/tomwoodley/Desktop/TommyWoodleyMEngProject/src/Data/FinalExperimentTrajectories/best_human/rosbag2_2024_06_07-21_02_12/rosbag2_2024_06_07-21_02_12.csv')

# Convert positions to meters
df['drone_x'] = - df['drone_x'] / 1000
df['drone_y'] = df['drone_y'] / 1000
df['drone_z'] = df['drone_z'] / 1000

# Apply a simple moving average for smoothing
window_size = 3
df['drone_x_smooth'] = df['drone_x'].rolling(window=window_size).mean()
df['drone_y_smooth'] = df['drone_y'].rolling(window=window_size).mean()
df['drone_z_smooth'] = df['drone_z'].rolling(window=window_size).mean()

start_index = df[df['drone_z'] > 2.0].index[0]
filtered_df = df.loc[start_index:]

# Extract timestamp and smoothed drone positions
timestamps = filtered_df['Timestamp']
drone_x = filtered_df['drone_x_smooth']
drone_y = filtered_df['drone_y_smooth']
drone_z = filtered_df['drone_z_smooth']

# Plot XZ Position
plt.figure()
plt.plot(drone_x, drone_z, label='XZ Position')
plt.xlabel('X Position (m)')
plt.ylabel('Z Position (m)')
plt.title('Actual Position of the Drone during Pracitcal Experiments')
plt.ylim(0, 3)
plt.legend()
plt.show()

# Plot X Position
plt.figure()
plt.plot(timestamps, drone_x, label='X Position')
plt.xlabel('Timestamp')
plt.ylabel('Position (m)')
plt.title('Drone X Position Over Time')
plt.legend()
plt.show()

# Plot Y Position
plt.figure()
plt.plot(timestamps, drone_y, label='Y Position')
plt.xlabel('Timestamp')
plt.ylabel('Position (m)')
plt.title('Drone Y Position Over Time')
plt.legend()
plt.show()

# Plot Z Position
plt.figure()
plt.plot(timestamps, drone_z, label='Z Position')
plt.xlabel('Timestamp')
plt.ylabel('Position (m)')
plt.title('Drone Z Position Over Time')
plt.legend()
plt.show()
