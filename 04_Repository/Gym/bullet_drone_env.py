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
        pos = np.zeros(3, dtype=np.float32)
        pos[0] = self.reset_pos[0]
        pos[2] = self.reset_pos[1]
        self.simulator = TetheredDroneSimulator(drone_pos=pos)
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
        pos = np.zeros(3, dtype=np.float32)
        pos[0] = self.reset_pos[0]
        pos[2] = self.reset_pos[1]
        self.simulator.reset(pos)
        state = self.reset_pos
        self.num_steps = 0
        return state, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        new_action = np.zeros(3, dtype=np.float32)
        new_action[0] = action[0]
        new_action[2] = action[1]

        self.simulator.step(new_action)
        r_state = self.simulator.drone_pos
        state = np.zeros(2, dtype=np.float32)
        state[0] = r_state[0]
        state[1] = r_state[2]

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
