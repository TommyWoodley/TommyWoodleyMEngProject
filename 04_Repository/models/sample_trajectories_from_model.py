import sys
import os
import gymnasium as gym
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Gym.bullet_drone_env import BulletDroneEnv
from Gym.Wrappers.two_dim_wrapper import TwoDimWrapper
from Gym.Wrappers.position_wrapper import PositionWrapper
from Gym.Wrappers.symmetric_wrapper import SymmetricWrapper
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


def sample_trajectories(dir, show=True, human=False):
    file_name = f"{dir}/model.zip"
    output_filename = f"{dir}/sample_trajectories.png"
    sample_trajectories_from_file(file_name, output_filename, show, human)


def sample_trajectories_from_file(file, output_filename, show=True, human=False):
    plotting_degrees = [0, 45, 90, 135, 180, 225, 270, 315]

    model = SAC.load(file)
    render_mode = "console" if not human else "human"
    env = SampleTrajEnv(PositionWrapper(TwoDimWrapper(SymmetricWrapper(BulletDroneEnv(render_mode=render_mode)))),
                        plotting_degrees=plotting_degrees)
    model.set_env(env)

    num_trajectories = len(plotting_degrees)
    if human:
        print("Num Trajectories: ", num_trajectories)
    trajectory_length = 60
    trajectory_states = []
    done = False

    for _ in range(num_trajectories):
        if not done:
            obs = model.env.reset()
            global global_info
        trajectory = []
        x, _, z = global_info["original_state"]
        trajectory.append(np.array([x, z]))
        for i in range(trajectory_length):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = model.env.step(action)
            # print(info)
            if done:
                # TODO: Fix this to add the final state into visual
                if human:
                    print(f"Done: {i}")
                break
            x, _, z = info[0]["original_state"]
            trajectory.append(np.array([x, z]))
        trajectory_states.append(trajectory)
    env.close()

    plot_trajectories(trajectory_states, output_filename=output_filename, show_plot=show)


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
