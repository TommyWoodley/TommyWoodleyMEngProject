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
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.MIN, high=self.MAX, shape=(2,), dtype=np.float32)
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
            x, y, z = state
            new_state = (-1 * x, y, z)
        
        # print(f"Old State: {state}, New State: {new_state}, Old Action {action}, New Action {new_action}")

        return new_state, reward, terminated, truncated, info

    def reset(self, seed: int = None, options: Dict[Any, Any] = None,
              degrees: int = None, position=None) -> Tuple[np.ndarray, Dict[Any, Any]]:
        state, info = self.env.reset(seed, options, degrees, position)
        x, y, z = state
        if x >= 0:
            self.positive = True
        else:
            self.positive = False

        info["original_state"] = state
        
        if self.positive:
            new_state = state
        else:
            x, y, z = state
            new_state = (-1 * x, y, z)
      
        print(f"Old State: {state}, New State: {new_state}")
        return new_state, info

