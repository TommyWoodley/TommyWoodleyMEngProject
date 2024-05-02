import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple
import numpy as np


class PositionWrapper(gym.Wrapper):
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
        self.current_state = None
        env.unwrapped.should_render = False
        self.num_steps = 0

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        action = action * self.MAX_STEP
        self.num_steps += 1

        action = action / self.NUM_ACTIONS_PER_STEP
        total_reward = 0
        actual_steps_taken = 0
        
        for i in range(self.NUM_ACTIONS_PER_STEP):
            state, reward, terminated, truncated, info = self._take_single_step(action)
            total_reward += reward
            actual_steps_taken += 1
            if terminated or truncated:
                break
        
        avg_reward = total_reward / actual_steps_taken
        return state, avg_reward - 1, terminated, truncated, info

    def reset(self, seed: int = None, options: Dict[Any, Any] = None,
              degrees: int = None, position=None) -> Tuple[np.ndarray, Dict[Any, Any]]:
        state, info = self.env.reset(seed, options, degrees, position)
        self.current_state = state
        self.num_steps = 0
        return state, info

    def render(self):
        print(f'Agent position: {self.current_state}')

    def _is_close_enough(curr_pos, target_pos, threshold=0.001) -> bool:
        distance = np.linalg.norm(curr_pos - target_pos)
        return distance <= threshold

    def _take_single_step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        # Calculate direction vector

        state, reward, terminated, truncated, info = self.env.step(action)

        self.current_state = state

        return state, reward, terminated, truncated or self.num_steps >= 60, info
