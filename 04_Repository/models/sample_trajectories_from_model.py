import sys
import os
import gymnasium as gym
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Gym.bullet_drone_env import BulletDroneEnv
from Gym.Wrappers.two_dim_wrapper import TwoDimWrapper
from Gym.Wrappers.position_wrapper import PositionWrapper
from stable_baselines3 import SAC
from utils.graphics.plot_trajectories import plot_trajectories


class SampleTrajEnv(gym.Wrapper):
    def __init__(self, env, plotting_degrees):
        super().__init__(env)
        self.plotting_degrees = plotting_degrees
        self.iterator = 0
        self.fake_reset_done = True

    def reset(self, **kwargs):
        self.fake_reset_done = False
        print("Plotting degrees: ", self.plotting_degrees[self.iterator])
        obs = self.env.reset(degrees=self.plotting_degrees[self.iterator], **kwargs)
        self.iterator = (self.iterator + 1) % len(self.plotting_degrees)
        return obs


def sample_trajectories(dir, show=True, human=False):
    plotting_degrees = [0, 45, 90, 135, 180, 225, 270, 315]
    # plotting_degrees = [135, 180, 0]
    
    model = SAC.load(f"{dir}/model.zip")
    render_mode = "console" if not human else "human"
    env = SampleTrajEnv(PositionWrapper(TwoDimWrapper(BulletDroneEnv(render_mode=render_mode))),
                        plotting_degrees=plotting_degrees)
    model.set_env(env)

    num_trajectories = len(plotting_degrees)
    print("Num Trajectories: ", num_trajectories)
    trajectory_length = 40
    trajectory_states = []
    done = False

    for _ in range(num_trajectories):
        if not done:
            obs = model.env.reset()
        trajectory = []
        trajectory.append(obs[0])
        for _ in range(trajectory_length):
            action, _ = model.predict(obs, deterministic=True)
            print(action)
            obs, _, done, _ = model.env.step(action)
            if done:
                trajectory.append(trajectory[-1] + 0.5 * action[0])
                print("Done")
                break
            trajectory.append(obs[0])
        trajectory_states.append(trajectory)
    env.close()

    plot_trajectories(trajectory_states, output_filename=f"{dir}/sample_trajectories.png", show_plot=show)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        dir = sys.argv[1]
        sample_trajectories(dir, show=True)
    elif len(sys.argv) == 3 and sys.argv[2] == "-h":
        dir = sys.argv[1]
        sample_trajectories(dir, show=True, human=True)
    else:
        print("Usage: python sample_trajectories_from_model.py <model dir>")
