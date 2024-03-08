import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple

from TetherModel.Environment.tethered_drone_simulator import TetheredDroneSimulator


class BulletDroneEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """

    metadata = {"render_modes": ["console"]}
    reset_pos = np.array([2, 3], dtype=np.float32)
    goal_state = np.array([0.0, 3.0])  # Goal state

    def __init__(self, render_mode: str = "console") -> None:
        super(BulletDroneEnv, self).__init__()
        self.simulator = TetheredDroneSimulator(drone_pos=self._convert_2d_to_3d(self.reset_pos))
        self.action_space = spaces.Box(low=np.array([-0.001, -0.001]), high=np.array([0.001, 0.001]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.render_mode = render_mode
        self.num_steps = 0

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[Any, Any]]:
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """

        super().reset(seed=seed, options=options)
        self.simulator.reset(self._convert_2d_to_3d(self.reset_pos))
        state = self.reset_pos
        self.num_steps = 0
        return state, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        self.simulator.step(self._convert_2d_to_3d(action))
        r_state = self.simulator.drone_pos
        state = self._convert_3d_to_2d(r_state)

        self.num_steps += 1

        reward, terminated, truncated = self.reward_fun(state)
        info = {"distance_to_goal": -reward}

        return state, reward, terminated, truncated, info

    def render(self) -> None:
        # agent is represented as a cross, rest as a dot
        if self.render_mode == "console":
            print(f'Agent position: {self.simulator.drone_pos}')

    def close(self) -> None:
        if hasattr(self, 'simulator'):
            self.simulator.close()

    def reward_fun(self, state: np.ndarray) -> Tuple[float, bool, bool]:
        # Implement how reward is calculated based on the state
        distance = np.linalg.norm(state - self.goal_state)
        return - distance, bool(distance < 0.1), bool(self.num_steps > 1000)

    def _convert_2d_to_3d(self, arr: np.ndarray) -> np.ndarray:
        new_arr = np.zeros(3, dtype=np.float32)
        new_arr[0] = arr[0]
        new_arr[2] = arr[1]
        return new_arr

    def _convert_3d_to_2d(self, arr: np.ndarray) -> np.ndarray:
        new_arr = np.zeros(2, dtype=np.float32)
        new_arr[0] = arr[0]
        new_arr[1] = arr[2]
