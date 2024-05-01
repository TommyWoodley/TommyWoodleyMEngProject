import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple
import numpy as np


class PositionWrapper(gym.Wrapper):
    MAGNITUDE = 0.005
    MAX_STEP = 0.5

    def __init__(self, env) -> None:
        super().__init__(env)

        # Position Based Action Space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-3, high=6, shape=(2,), dtype=np.float32)
        self.current_state = None
        env.unwrapped.should_render = False
        self.num_steps = 0

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        action = action * self.MAX_STEP
        self.num_steps += 1

        num = 25
        action = action / num
        total_reward = 0
        # print(np.linalg.norm(action))

        state, reward, terminated, truncated, info = self._take_single_step(action)
        
        while num > 0:
            total_reward += reward
            num = num - 1
            state, reward, terminated, truncated, info = self._take_single_step(action)
        
        total_reward += reward
        reward = total_reward / 25.0 #TODO: Watch out for this
        # print(f"Num: {self.num_steps}, num: {num}")
        
        # print(f"Terminated: {terminated}, Truncated: {truncated}, numSteps: {self.num_steps}")

        return state, reward - 1, terminated, truncated, info

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
