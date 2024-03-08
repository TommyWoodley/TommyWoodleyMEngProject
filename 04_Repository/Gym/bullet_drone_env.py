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
    reset_pos = [2, 0, 3]
    goal_state = np.array([0.0, 3.0])  # Goal state

    def __init__(self, render_mode: str = "console") -> None:
        super(BulletDroneEnv, self).__init__()
        self.simulator = TetheredDroneSimulator(drone_pos=self.reset_pos)
        self.action_space = spaces.Box(low=np.array([-0.001, -0.001]), high=np.array([0.001, 0.001]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.render_mode = render_mode
        self.num_steps = 0

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> None:
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """

        super().reset(seed=seed, options=options)
        self.simulator.reset(self.reset_pos)
        state = np.array([0.0, 1.0])
        self.num_steps = 0
        return np.array(state, dtype=np.float32), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        action_list = [action[0], 0.0, action[1]]
        self.simulator.step(action_list)
        state_list = self.simulator.drone_pos
        state = np.array([state_list[0], state_list[2]], dtype=np.float32)
        self.num_steps += 1

        distance_to_goal = np.linalg.norm(state - self.goal_state)
        reward = -distance_to_goal
        terminated = distance_to_goal < 0.1  # Example threshold
        truncated = self.num_steps > 1000
        info = {"distance_to_goal": distance_to_goal}

        return state, reward, terminated, truncated, info

    def render(self) -> None:
        # agent is represented as a cross, rest as a dot
        if self.render_mode == "console":
            print(f'Agent position: {self.simulator.drone_pos}')

    def close(self) -> None:
        if hasattr(self, 'simulator'):
            self.simulator.close()

    def reward_fun(self, state: np.ndarray) -> float:
        # Implement how reward is calculated based on the state
        return 0.0

    def check_if_done(self, state: np.ndarray) -> bool:
        # Implement to check if the episode is done based on the state
        return False
