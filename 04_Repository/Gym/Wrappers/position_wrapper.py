import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple
import numpy as np


class PositionWrapper(gym.Wrapper):
    MAGNITUDE = 0.001
    MAX_STEP = 0.5

    def __init__(self, env) -> None:
        super().__init__(env)

        # Position Based Action Space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.current_state = None
        env.unwrapped.should_render = False
        self.num_steps = 0

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        action = action * self.MAX_STEP
        self.num_steps += 1
        waypoint = self.current_state + action

        state, reward, terminated, truncated, info = self._take_single_step(waypoint)
        while not PositionWrapper._is_close_enough(self.current_state, waypoint):
            state, reward, terminated, truncated, info = self._take_single_step(waypoint)

        return state, reward, terminated, truncated, info

    def reset(self, seed: int = None, options: Dict[Any, Any] = None,
              degrees: int = None) -> Tuple[np.ndarray, Dict[Any, Any]]:
        state, info = self.env.reset(seed, options, degrees)
        self.render()
        self.current_state = state
        self.num_steps = 0
        return state, info

    def render(self):
        print(f'Agent position: {self.current_state}')

    def _is_close_enough(curr_pos, target_pos, threshold=0.01) -> bool:
        distance = np.linalg.norm(curr_pos - target_pos)
        return distance <= threshold

    def _take_single_step(self, target: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        # Calculate direction vector
        direction = target - self.current_state
        action = np.clip(direction, -self.MAGNITUDE, self.MAGNITUDE)

        state, reward, terminated, truncated, info = self.env.step(action)

        self.current_state = state

        return state, reward, terminated, self.num_steps >= 10, info
