import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple
import numpy as np


class PositionWrapper(gym.Wrapper):
    MAGNITUDE=0.001
    def __init__(self, env) -> None:
        super().__init__(env)

        # Position Based Action Space
        self.action_space = spaces.Box(low=np.array([-5, 0]), high=np.array([5, 5]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.current_state = None

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        print("Action: ", action)
        state, reward, terminated, truncated, info = self.take_single_step(action)

        while not PositionWrapper.is_close_enough(self.current_state, action):
            state, reward, terminated, truncated, info = self.take_single_step(action)
        
        return state, reward, terminated, truncated, info

    def reset(self, seed: int = None, options: Dict[Any, Any] = None) -> Tuple[np.ndarray, Dict[Any, Any]]:
        state, info = self.env.reset(seed, options)
        self.current_state = state
        return state, info
    
    def is_close_enough(curr_pos, target_pos, threshold=0.01) -> bool:
        distance = np.linalg.norm(curr_pos - target_pos)
        return distance <= threshold
    
    def take_single_step(self, target:np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        # Calculate direction vector
        direction = target - self.current_state
        action = np.clip(direction, -self.MAGNITUDE, self.MAGNITUDE)

        state, reward, terminated, truncated, info = self.env.step(action)

        self.current_state = state

        return state, reward, terminated, truncated, info
