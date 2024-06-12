import sys
import os
import csv
import gymnasium as gym
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Gym.bullet_drone_env import BulletDroneEnv
from Gym.Wrappers.two_dim_wrapper import TwoDimWrapper
from Gym.Wrappers.position_wrapper import PositionWrapper
from Gym.Wrappers.symmetric_wrapper import SymmetricWrapper
from Gym.Wrappers.hovering_wrapper import HoveringWrapper
from stable_baselines3 import SAC
from utils.graphics.plot_trajectories import plot_trajectories
import numpy as np

global_info = {}


class SampleTrajEnv(gym.Wrapper):
    def __init__(self, env, plotting_degrees):
        super().__init__(env)
        self.plotting_degrees = plotting_degrees
        self.iterator = 0
        self.fake_reset_done = True

    def reset(self, **kwargs):
        self.fake_reset_done = False
        obs, info = self.env.reset(degrees=self.plotting_degrees[self.iterator], **kwargs)
        global global_info
        global_info = info
        self.iterator = (self.iterator + 1) % len(self.plotting_degrees)
        return obs, info


def sample_trajectories(dir, show=True, human=False, phase="all"):
    file_name = f"{dir}/model.zip"
    output_filename = f"{dir}/sample_trajectories.png"
    sample_trajectories_from_file(file_name, output_filename, show, human, phase=phase, log_dir=dir)


def sample_trajectories_from_file(file, output_filename, show=True, human=False, phase="approaching", log_dir=None):
    plotting_degrees = [0, 45, 180, 135]

    model = SAC.load(file)
    render_mode = "console" if not human else "human"
    env = SampleTrajEnv(HoveringWrapper(PositionWrapper(TwoDimWrapper(SymmetricWrapper(
        BulletDroneEnv(render_mode=render_mode, phase=phase, log_dir="logs/"))))), plotting_degrees=plotting_degrees)
    model.set_env(env)

    num_trajectories = len(plotting_degrees)
    if human:
        print("Num Trajectories: ", num_trajectories)
    trajectory_length = 100
    trajectory_states = []
    hanging_states = []
    done = False

    for _ in range(num_trajectories):
        if not done:
            obs = model.env.reset()
            global global_info
        trajectory = []
        hanging = []
        x, _, z, _ = global_info["original_state"]
        trajectory.append(np.array([x, z]))
        hanging.append(False)

        for i in range(trajectory_length):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = model.env.step(action)
            # print(reward)
            if done:
                # TODO: Fix this to add the final state into visual
                # if human:
                print(f"Done: {i}")
                break
            x, _, z, _ = info[0]["original_state"]
            if reward < -2.0:
                hanging.append(False)
            else:
                hanging.append(True)
            trajectory.append(np.array([x, z]))
        trajectory_states.append(trajectory)
        hanging_states.append(hanging)
    env.close()

    plot_trajectories(trajectory_states, output_filename=output_filename, show_plot=show)

    if log_dir is not None:
        log_trajectories(trajectory_states, hanging_states, log_dir)


def log_trajectories(trajectories, hanging_states, output_dir):
    """
    Writes each trajectory to a separate CSV file in a subdirectory called 'sampled_trajectories'
    within the specified output directory.

    Parameters:
    - trajectories: List of trajectories, where each trajectory is a list of (x, z) positions.
    - output_dir: Directory name where the 'sampled_trajectories' subdirectory will be created.
    """
    # Define the subdirectory path
    subdirectory = os.path.join(output_dir, 'sampled_trajectories')

    # Ensure the subdirectory exists
    os.makedirs(subdirectory, exist_ok=True)

    for i, (trajectory, haning) in enumerate(zip(trajectories, hanging_states)):
        # Define the filename for each trajectory
        filename = os.path.join(subdirectory, f'trajectory_{i + 1}.csv')

        # Write the trajectory to a CSV file
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['x', 'y', 'z', 'h'])  # Write the header

            for (x, z), is_hang in zip(trajectory, haning):
                csvwriter.writerow([x, 0, z, is_hang])  # Write the (x, y, z) position with y always being 0


if __name__ == "__main__":
    if len(sys.argv) == 2:
        dir = sys.argv[1]
        human = False
    elif len(sys.argv) == 3 and sys.argv[2] == "-h":
        dir = sys.argv[1]
        human = True
    else:
        print("Usage: python sample_trajectories_from_model.py <model dir>")
        exit()
    if dir.endswith(".zip"):
        sample_trajectories_from_file(dir, None, show=True, human=human)
    else:
        sample_trajectories(dir, show=True, human=human)
