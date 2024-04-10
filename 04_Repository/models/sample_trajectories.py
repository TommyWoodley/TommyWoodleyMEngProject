import numpy as np
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Gym.bullet_drone_env import BulletDroneEnv
from Gym.Wrappers.two_dim_wrapper import TwoDimWrapper
from Gym.Wrappers.position_wrapper import PositionWrapper
from stable_baselines3 import SAC

dir = "approach-sac_2024_04_10_11_39"
model = SAC.load(f"{dir}/model.zip") # Model to load
env = PositionWrapper(TwoDimWrapper(BulletDroneEnv()))
model.set_env(env)

num_trajectories = 5
trajectory_length = 10
trajectory_states = []

for _ in range(num_trajectories):
    obs = model.env.reset()  # Reset the environment
    trajectory = []  # Store states for this trajectory
    trajectory.append(obs)
    for _ in range(trajectory_length):
        action, _ = model.predict(obs, deterministic=True)  # Get action from the policy
        obs, _, done, _ = model.env.step(action)  # Take action in the environment
        if done:
            break
        trajectory.append(obs)  # Append the state to the trajectory
    trajectory_states.append(trajectory)

# Plot each trajectory
for trajectory in trajectory_states:
    for state in trajectory:
        print(state)
    x_values = [state[0][0] for state in trajectory]  # Extract x coordinates
    y_values = [state[0][1] for state in trajectory]  # Extract y coordinates
    plt.plot(x_values, y_values)

# Add labels and title
plt.xlabel('X position')
plt.ylabel('Z position')
plt.title('Sample Trajectories')

plt.savefig(f"{dir}/sample_trajectories.png")
plt.show()
