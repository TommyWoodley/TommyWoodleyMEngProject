import pandas as pd
import numpy as np
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Gym.bullet_drone_env import BulletDroneEnv

bulletDroneEnv = BulletDroneEnv()


angle = "22.5"


def calc_reward(state):
    x, z, t = state
    return bulletDroneEnv.calc_reward(np.array([x, 0.0, z]))


def transform_demo(csv_file):
    interval_seconds = 5
    # Load the CSV file
    df = pd.read_csv(f"2024_05_22_Flight_Data/processed/" + csv_file)

    df['delta_drone_x'] = df['drone_x'].diff().fillna(0)
    df['delta_drone_z'] = df['drone_z'].diff().fillna(0)
    df['distance'] = np.sqrt(df['delta_drone_x']**2 + df['delta_drone_z']**2)
    # Convert Timestamp to a datetime object for easier manipulation
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ns')

    waypoints = []
    waypoints.append((df.iloc[0]['dron_x'], df.iloc[0]['drone_z'] + 3))
    previous_time = df['Timestamp'].iloc[0]

    cumulative_distance = 0
    for index, row in df.iterrows():
        if (row['Timestamp'] - previous_time).total_seconds() >= interval_seconds:
            waypoints.append((row['drone_x'], row['drone_z'] + 3))
            previous_time = row['Timestamp']
    print("WAYPOINTS: ", waypoints)

    # Calculate state, action rewards
    x_original, _ = waypoints[0]
    mult = 1 if x_original >= 0 else -1
    print(f"Angle: {angle} is {mult}")

    state_action_reward = []
    memory = []
    for i in range(len(waypoints) - 1):
        x, y = waypoints[i]
        current_state = (x * mult, y, 0.0)

        if i == 0:
            memory = [current_state] * 3

        memory.append(current_state)
        if len(memory) > 3:
            memory.pop(0)

        augmented_current_state = tuple(item for state in memory for item in state)

        x, y = waypoints[i + 1]
        next_state = (x * mult, y, 0.0)

        # Add next state to the memory to calculate augmented next state
        memory.append(next_state)
        if len(memory) > 3:
            augmented_next_state = tuple(item for state in memory[-3:] for item in state)
        else:
            augmented_next_state = tuple(item for state in memory for item in state)
        memory.pop()  # Remove the next state from memory after using it for augmentation

        # Calculate action as difference between next and current state
        action = (next_state[0] - current_state[0], next_state[1] - current_state[1])

        reward = calc_reward(current_state)
        state_action_reward.append((augmented_current_state, action, reward, augmented_next_state))

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
    with open(f"rl_demos/rl_demo_approaching_angle_{angle}.json", 'w') as file:
        json.dump(state_action_reward_serializable, file, indent=4)

    print(f"Data saved to rl_demo_approaching_angle_{angle}.json")


csv_file = ["rosbag2_2024_05_22-17_00_56.csv",]
            # "rosbag2_2024_05_22-17_03_00.csv", "rosbag2_2024_05_22-17_20_43.csv",
            # "rosbag2_2024_05_22-17_26_15.csv", "rosbag2_2024_05_22-18_10_51.csv", "rosbag2_2024_05_22-18_16_45.csv"]
for file in csv_file:
    transform_demo(file)
