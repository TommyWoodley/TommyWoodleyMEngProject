import pandas as pd
import numpy as np
import json


def calc_reward(state):
    target_position = np.array([0, 3])
    distance = np.linalg.norm(np.array(state) - target_position)
    return -distance


# Load the CSV file
df = pd.read_csv('Original/trajectory_data.csv')
df['delta_x'] = df['cycleX'].diff().fillna(0)
df['delta_z'] = df['cycleZ'].diff().fillna(0)
df['distance'] = np.sqrt(df['delta_x']**2 + df['delta_z']**2)

waypoints = []
cumulative_distance = 0
for index, row in df.iterrows():
    cumulative_distance += row['distance']
    if cumulative_distance >= 0.25:
        waypoints.append((row['cycleX'], row['cycleZ'] + 3))
        cumulative_distance = 0  # Reset the distance accumulator
print("WAYPOINTS: ", waypoints)

# Calculate state, action rewards
state_action_reward = []
for i in range(len(waypoints) - 1):
    current_state = waypoints[i]
    next_state = waypoints[i + 1]

    # Calculate action as difference between next and current state
    action = (next_state[0] - current_state[0], next_state[1] - current_state[1])

    reward = calc_reward(current_state)
    state_action_reward.append((current_state, action, reward, next_state))

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
with open('state_action_reward.json', 'w') as file:
    json.dump(state_action_reward_serializable, file, indent=4)

print("Data saved to 'state_action_reward.json'.")
