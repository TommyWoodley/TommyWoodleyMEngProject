import pandas as pd
import numpy as np
import json
import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# from Gym.bullet_drone_env import BulletDroneEnv

# bulletDroneEnv = BulletDroneEnv()


angle = "22.5"


# def calc_reward(state):
#     x, z, t = state
#     return bulletDroneEnv.calc_reward(np.array([x, 0.0, z]))


def transform_demo(csv_file):
    interval_seconds = 0.1
    # Load the CSV file
    df = pd.read_csv(f"2024_05_22_Flight_Data/processed/" + csv_file)

    df['delta_drone_x'] = df['drone_x'].diff().fillna(0)
    df['delta_drone_z'] = df['drone_z'].diff().fillna(0)
    df['distance'] = np.sqrt(df['delta_drone_x']**2 + df['delta_drone_z']**2)
    # Convert Timestamp to a datetime object for easier manipulation
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ns')

    waypoints = []
    waypoints.append((df.iloc[0]['drone_x'], df.iloc[0]['drone_z'] + 1000))
    previous_time = df['Timestamp'].iloc[0]

    for index, row in df.iterrows():
        if (row['Timestamp'] - previous_time).total_seconds() >= interval_seconds:
            waypoints.append((row['drone_x'], row['drone_z'] + 1000))
            previous_time = row['Timestamp']
    print("WAYPOINTS: ", waypoints)

    # Calculate state, action rewards
    x_original, _ = waypoints[0]
    mult = 1 if x_original >= 0 else -1
    print(f"Angle: {csv_file} is {mult}")

    state_action_reward = []
    memory = []
    x, y = waypoints[0]
    num_wraps = 0.0
    curr_x = x / 1000
    curr_y = y / 1000
    for i in range(len(waypoints) - 1):
        next_x, next_y = waypoints[i]
        next_x = next_x / 1000
        next_y = next_y / 1000
        action_x = curr_x - next_x
        action_y = curr_y - next_y
        
        state_action_reward.append(((curr_x, curr_y, num_wraps), (action_x, action_y), 0.0, (next_x, next_y)))

        curr_x = next_x
        curr_y = next_y

    # Print the state, action, reward list
    print("STATE,ACTION,REWARDS: ")
    for strs in state_action_reward:
        print(strs)

    state_action_reward_serializable = []

    # Convert data into a JSON serializable format
    for state, action, reward, next_state in state_action_reward:
        state_action_reward_serializable.append({
            "state": list(state),
            "action": list(action),
            "reward": reward,
            "next_state": list(next_state)
        })

    # Write the serializable list to a JSON file
    with open(f"rl_demos/rl_demo_approaching.json", 'w') as file:
        json.dump(state_action_reward_serializable, file, indent=4)

    print(f"Data saved to rl_demo_approaching.json")


csv_file = ["rosbag2_2024_05_22-17_00_56.csv",]
            # "rosbag2_2024_05_22-17_03_00.csv", "rosbag2_2024_05_22-17_20_43.csv",
            # "rosbag2_2024_05_22-17_26_15.csv", "rosbag2_2024_05_22-18_10_51.csv", "rosbag2_2024_05_22-18_16_45.csv"]
for file in csv_file:
    transform_demo(file)
