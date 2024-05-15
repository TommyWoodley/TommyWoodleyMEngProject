import gymnasium as gym
from gymnasium import spaces
import numpy as np

class HoveringWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Assuming the original observation space is a Box of some shape
        low = np.append(env.observation_space.low, -np.inf)
        high = np.append(env.observation_space.high, np.inf)
        
        # Update observation space to include one additional integer for the hover count
        # self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space = env.observation_space
        
        # Initialize hover count
        self.hover_count = 0

    def reset(self, seed = None, options = None, degrees = None, position = None):
        # Reset the environment and hover count
        obs, info = self.env.reset(seed, options, degrees, position)
        self.hover_count = 0
        # Return the augmented observation
        augmented_obs = np.append(obs, self.hover_count)
        return augmented_obs, info

    def step(self, action):
        # Take a step using the underlying environment
        obs, reward, done, truncated, info = self.env.step(action)
        # Update the hover count if the action size is less than 0.01
        if np.linalg.norm(action) < 0.1:
            self.hover_count += 1
        print(self.hover_count, np.linalg.norm(action))
        # Augment the observation with the hover count
        augmented_obs = np.append(obs, self.hover_count)
        return augmented_obs, reward, done, truncated, info
