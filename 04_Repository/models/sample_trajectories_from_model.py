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
        if self.fake_reset_done:
            self.fake_reset_done = False
            obs = self.env.reset(degrees=self.plotting_degrees[self.iterator], **kwargs)
            self.iterator = (self.iterator + 1) % len(self.plotting_degrees)
            return obs
        else:
            self.fake_reset_done = True
            return self.env.reset(degrees=self.plotting_degrees[self.iterator], **kwargs)


def sample_trajectories(dir):
    plotting_degrees = [0, 45, 90, 135, 180, 225, 270, 315]

    model = SAC.load(f"{dir}/model.zip")
    env = SampleTrajEnv(PositionWrapper(TwoDimWrapper(BulletDroneEnv(render_mode="console"))),
                        plotting_degrees=plotting_degrees)
    model.set_env(env)

    num_trajectories = len(plotting_degrees)
    print("Num Trajectories: ", num_trajectories)
    trajectory_length = 10
    trajectory_states = []

    for _ in range(num_trajectories):
        obs = model.env.reset()
        trajectory = []
        trajectory.append(obs[0])
        for _ in range(trajectory_length):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = model.env.step(action)
            if done:
                break
            trajectory.append(obs[0])
        trajectory_states.append(trajectory)
    env.close()

    plot_trajectories(trajectory_states)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sample_trajectories_from_model.py <model dir>")
    else:
        dir = sys.argv[1]
        sample_trajectories(dir)
