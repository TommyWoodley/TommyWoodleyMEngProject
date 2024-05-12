import json
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.graphics.plot_trajectories import plot_trajectories

json_files = ["0.0", "22.5", "45.0", "67.5", "90.0", "112.5", "135.0", "157.5", "180.0", "202.5", "225.0", "247.5", "270.0", "292.5", "315.0", "337.5", "360.0"]


def extract_and_plot(dir):
    trajectories = []
    for i, angle in enumerate(json_files):
        filename = f"{dir}/rl_demo_approaching_angle_{angle}.json"
        # Load data from each JSON file
        with open(filename, 'r') as file:
            data = json.load(file)

        # Extracting state coordinates for plotting
        states = []

        for entry in data:
            states.append(np.array([entry["state"][0], entry["state"][1]]))

        states.append(np.array([entry["next_state"][0], entry["next_state"][1]]))

        # Plot each file's data with a different color
        trajectories.append(states)

    plot_trajectories(trajectories)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: approaching_visualisation.py <traj dir>")
    else:
        dir = sys.argv[1]
        extract_and_plot(dir)
