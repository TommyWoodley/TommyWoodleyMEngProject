import pandas as pd
import numpy as np
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Gym.bullet_drone_env import BulletDroneEnv

bulletDroneEnv = BulletDroneEnv()


angle = "22.5"


def calc_reward(state):
    x, z, t = state
    return bulletDroneEnv.calc_reward_and_done(np.array([x, 0.0, z]), num_wraps=t)


class WeightCalculator():
    def __init__(self) -> None:
        self.weight_prev_angle = None
        self.weight_cumulative_angle_change = 0
        self.weight_wraps = 0

    def compute_total_rotation(self, weight_x, weight_y):
        # ANGLE FOR WEIGHT
        weight_delta_x = weight_x - 0
        weight_delta_y = 2.7 - weight_y

        # Compute the angle using arctan2, which considers quadrant location
        weight_angle_radians = np.arctan2(weight_delta_x, weight_delta_y)  # swapped x and y to align with the vertical
        weight_angle_degrees = np.degrees(weight_angle_radians)

        if self.weight_prev_angle is not None:
            # Calculate angle change considering the wrap around at 180/-180
            weight_angle_change = weight_angle_degrees - self.weight_prev_angle
            if weight_angle_change > 180:
                weight_angle_change -= 360
            elif weight_angle_change < -180:
                weight_angle_change += 360

            # Update cumulative angle change
            self.weight_cumulative_angle_change += weight_angle_change

            # Update wraps as a float
            self.weight_wraps = self.weight_cumulative_angle_change / 360.0

        # Update the previous angle for the next call
        self.weight_prev_angle = weight_angle_degrees

        return abs(self.weight_cumulative_angle_change)


def transform_demo(version, csv_file):
    interval_seconds = 0.10
    # Load the CSV file
    df = pd.read_csv("2024_05_22_Flight_Data/processed/" + csv_file)

    df['delta_drone_x'] = df['drone_x'].diff().fillna(0)
    df['delta_drone_z'] = df['drone_z'].diff().fillna(0)
    df['distance'] = np.sqrt(df['delta_drone_x']**2 + df['delta_drone_z']**2)
    # Convert Timestamp to a datetime object for easier manipulation
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ns')

    waypoints = []
    previous_time = df['Timestamp'].iloc[0]

    start_adding_waypoints = False
    initial_movement_found = False
    has_hit = False
    for index, row in df.iterrows():
        if row['drone_x'] - row['round_bar_x'] < 0:
            has_hit = True
        if has_hit:
            num_wraps = 1.0
        else:
            num_wraps = 0.0
        if not start_adding_waypoints and row['drone_z'] > 2000:
            start_adding_waypoints = True
            prev_x = row['drone_x']
        if start_adding_waypoints and not initial_movement_found:
            delta_x = row['drone_x'] - prev_x
            if abs(delta_x) > 0.3 * 1000:
                print(f"Movement found {delta_x}")
                initial_movement_found = True
        if start_adding_waypoints and initial_movement_found:
            if (row['Timestamp'] - previous_time).total_seconds() >= interval_seconds:
                waypoints.append((row['drone_x'] - row['round_bar_x'],
                                  row['drone_z'] - row['round_bar_z'] + 2700,
                                  num_wraps))
                previous_time = row['Timestamp']
    print("WAYPOINTS: ", waypoints)

    # Calculate state, action rewards
    x_original, _, _ = waypoints[0]
    mult = 1 if x_original >= 0 else -1
    print(f"Angle: {csv_file} is {mult}")

    state_action_reward = []
    x, y, w = waypoints[0]
    curr_x = x / 1000
    curr_y = y / 1000
    curr_w = w
    max_action_magnitude = 0
    for i in range(len(waypoints) - 1):
        next_x, next_y, next_w = waypoints[i]
        next_x = next_x / 1000
        next_y = next_y / 1000
        action_x = next_x - curr_x
        action_y = next_y - curr_y

        action_magnitude = np.sqrt(action_x**2 + action_y**2)

        # Check if the magnitude exceeds 0.25 and print a warning if it does
        if action_magnitude > 0.25:
            print(f"Warning: Action magnitude exceeds 0.25 at index {i}. Magnitude: {action_magnitude}")

        if action_magnitude > max_action_magnitude:
            max_action_magnitude = action_magnitude

        reward, done = calc_reward((curr_x, curr_y, next_w))

        state_action_reward.append(((curr_x, curr_y, curr_w), (action_x, action_y), reward, (next_x, next_y, next_w)))

        curr_x = next_x
        curr_y = next_y
        curr_w = next_w

        if done:
            break

    # Print the state, action, reward list
    print("STATE,ACTION,REWARDS: ")
    for strs in state_action_reward:
        print(strs)
    print(f"Largest action magnitude: {max_action_magnitude}")
    print("NUM_WAYPOINTS: " + str(len(state_action_reward)))

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
    with open(f"rl_demos/rl_demo_{version}.json", 'w') as file:
        json.dump(state_action_reward_serializable, file, indent=4)

    print(f"Data saved to rl_demos/rl_demo_{version}.json")


csv_file = ["rosbag2_2024_05_22-17_20_43.csv",
            "rosbag2_2024_05_22-17_26_15.csv", "rosbag2_2024_05_22-18_10_51.csv", "rosbag2_2024_05_22-18_16_45.csv"]
for i in range(len(csv_file)):
    transform_demo(i + 3, csv_file[i])
    bulletDroneEnv.reset()
