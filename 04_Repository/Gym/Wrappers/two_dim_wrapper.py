import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple
import numpy as np


class TwoDimWrapper(gym.Wrapper):
    def __init__(self, env, x_range: Tuple[int, int] = (-10, 10), z_range: Tuple[int, int] = (-10, 10),
                 render_mode: str = "console") -> None:
        super().__init__(env)
        self.action_space = spaces.Box(low=np.array([-0.001, -0.001]), high=np.array([0.001, 0.001]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        state, reward, terminated, truncated, info = self.env.step(self._convert_2d_to_3d(action))
        new_state = self._convert_3d_to_2d(state)
        return new_state, reward, terminated, truncated, info

    def reset(self, seed: int = None, options: Dict[Any, Any] = None,
              degrees: int = None, position=None) -> Tuple[np.ndarray, Dict[Any, Any]]:
        state, info = self.env.reset(seed, options, degrees, position)
        new_state = self._convert_3d_to_2d(state)
        return new_state, info

    def _convert_2d_to_3d(self, arr: np.ndarray) -> np.ndarray:
        new_arr = np.zeros(3, dtype=np.float32)
        new_arr[0] = arr[0]
        new_arr[2] = arr[1]
        return new_arr

    def _convert_3d_to_2d(self, arr: np.ndarray) -> np.ndarray:
        new_arr = np.zeros(2, dtype=np.float32)
        new_arr[0] = arr[0]
        new_arr[1] = arr[2]
        return new_arr
