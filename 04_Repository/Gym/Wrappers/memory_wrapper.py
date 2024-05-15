import numpy as np
from gym import spaces

class MemoryWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.memory = [np.zeros_like(env.observation_space.sample()) for _ in range(3)]
        # Adjust observation space for the memory-augmented state
        self.observation_space = spaces.Box(low=np.tile(env.observation_space.low, 3),
                                            high=np.tile(env.observation_space.high, 3),
                                            dtype=env.observation_space.dtype)

    def reset(self):
        initial_obs = self.env.reset()
        self.memory = [initial_obs for _ in range(3)]
        return self._get_augmented_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Update memory
        self.memory.append(obs)
        if len(self.memory) > 3:
            self.memory.pop(0)
        return self._get_augmented_obs(), reward, done, info

    def _get_augmented_obs(self):
        # If there's less than 3 observations, repeat the oldest
        if len(self.memory) < 3:
            return np.concatenate([self.memory[0]] * (3 - len(self.memory)) + self.memory)
        return np.concatenate(self.memory)
