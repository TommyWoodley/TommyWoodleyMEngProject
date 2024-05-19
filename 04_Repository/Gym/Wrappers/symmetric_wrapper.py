import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple
import numpy as np


class SymmetricWrapper(gym.Wrapper):
    MAGNITUDE = 0.005
    MAX_STEP = 0.5
    MAX = 6
    MIN = -3
    NUM_ACTIONS_PER_STEP = 25

    def __init__(self, env) -> None:
        super().__init__(env)

        # Position Based Action Space
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.positive = True

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        # Adjust action
        if self.positive:
            new_action = action
        else:
            x, y, z = action
            new_action = np.array([-1 * x, y, z])

        state, reward, terminated, truncated, info = self.env.step(new_action)

        info["original_state"] = state

        if self.positive:
            new_state = state
        else:
            x, y, z, t = state
            new_state = (-1 * x, y, z, t)

        return new_state, reward, terminated, truncated, info

    def reset(self, seed: int = None, options: Dict[Any, Any] = None,
              degrees: int = None, position=None) -> Tuple[np.ndarray, Dict[Any, Any]]:
        state, info = self.env.reset(seed, options, degrees, position)
        x, y, z, t = state # Do we need this line?
        if x >= 0:
            self.positive = True
        else:
            self.positive = False

        info["original_state"] = state

        if self.positive:
            new_state = state
        else:
            x, y, z, t = state
            new_state = (-1 * x, y, z, t)
        return new_state, info
