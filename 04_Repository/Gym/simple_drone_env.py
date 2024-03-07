import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SimpleDroneEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """

    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {"render_modes": ["console"]}

    # Define constants for clearer code
    LEFT = 0
    RIGHT = 1

    def __init__(self, x_range=(-10, 10), z_range=(-10, 10), render_mode="console"):
        super(SimpleDroneEnv, self).__init__()
        self.render_mode = render_mode

        # Define action and observation space
        # Action space is a Box, representing movements in x and z directions
        self.action_space = spaces.Box(low=np.array([-2, -2]), high=np.array([2, 2]), dtype=np.float32)
        
        # Observation space represents the x and z coordinates of the agent
        self.observation_space = spaces.Box(low=np.array([x_range[0], z_range[0]]),
                                            high=np.array([x_range[1], z_range[1]]),
                                            dtype=np.float32)

        # Initial position will be set in reset method
        self.state = None

    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)
        self.state = self.observation_space.sample()
        return self.state, {}

    def step(self, action):
        # Update the agent's position based on the action taken
        self.state = np.clip(self.state + action, self.observation_space.low, self.observation_space.high)

        # Assuming no specific goal for simplicity, so no reward or termination logic
        reward = 0
        terminated = False
        truncated = False
        info = {}

        return self.state, reward, terminated, truncated, info

    def render(self):
        # agent is represented as a cross, rest as a dot
        if self.render_mode == "console":
            print(f'Agent position: x={self.state[0]}, z={self.state[1]}')

    def close(self):
        pass