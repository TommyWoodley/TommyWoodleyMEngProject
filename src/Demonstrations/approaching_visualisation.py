import json
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.graphics.plot_trajectories import plot_trajectories_with_rewards

json_files = ["3", "4", "5", "6"]


def extract_and_plot(dir):
    trajectories = []
    traj_rewards = []

    for i, angle in enumerate(json_files):
        filename = f"{dir}/rl_demo_{angle}.json"
        # Load data from each JSON file
        with open(filename, 'r') as file:
            data = json.load(file)

        # Extracting state coordinates for plotting
        states = []
        rewards = []

        rewards.append(-3)
        for entry in data:
            states.append(np.array([entry["state"][0], entry["state"][1]]))
            rewards.append(float(entry["reward"]))

        states.append(np.array([entry["next_state"][0], entry["next_state"][1]]))

        # Plot each file's data with a different color
        trajectories.append(states)
        traj_rewards.append(rewards)

    plot_trajectories_with_rewards(trajectories, traj_rewards=traj_rewards)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: approaching_visualisation.py <traj dir>")
    else:
        dir = sys.argv[1]
        extract_and_plot(dir)
