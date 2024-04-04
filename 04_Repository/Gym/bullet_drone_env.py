import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple

from TetherModel.Environment.tethered_drone_simulator import TetheredDroneSimulator


class BulletDroneEnv(gym.Env):
    """
    Custom PyBullet Drone Environment that follows gym interface.
    Render Modes
      - Console: Uses PyBullet Direct - supports multiple environments in parallel.
      - Human: Uses PyBullet GUI - note that this has limitations - GUI console cannot be quit
        additionally only environment can be built at a time.
    """

    metadata = {"render_modes": ["console", "human"]}
    reset_pos = [2, 0, 3]
    goal_state = np.array([0.0, 0.0, 3.0])  # Goal state
    reset_pos_distance = 2.0

    def __init__(self, render_mode: str = "human") -> None:
        super(BulletDroneEnv, self).__init__()
        self.simulator = TetheredDroneSimulator(drone_pos=self._generate_reset_position(42),
                                                gui_mode=(render_mode == "human"))
        self.action_space = spaces.Box(low=np.array([-0.001, -0.001, -0.001]),
                                       high=np.array([0.001, 0.001, 0.001]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.render_mode = render_mode
        self.num_steps = 0
        self.should_render = True

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[Any, Any]]:
        super().reset(seed=seed, options=options)
        reset_pos = self._generate_reset_position(seed)
        self.simulator.reset(reset_pos)
        self.num_steps = 0
        return reset_pos, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        self.simulator.step(action)
        self.render()
        state = self.simulator.drone_pos

        self.num_steps += 1

        reward, terminated, truncated = self.reward_fun(state)
        info = {"distance_to_goal": -reward}

        return state, reward, terminated, truncated, info

    def render(self) -> None:
        if self.should_render:
            self._render()

    def _render(self) -> None:
        # agent is represented as a cross, rest as a dot
        if self.render_mode == "console":
            print(f'Agent position: {self.simulator.drone_pos}')

    def close(self) -> None:
        if hasattr(self, 'simulator'):
            self.simulator.close()

    def reward_fun(self, state: np.ndarray) -> Tuple[float, bool, bool]:
        # Implement how reward is calculated based on the state
        distance = np.linalg.norm(state - self.goal_state)
        return - distance, bool(distance < 0.1), False

    def _generate_reset_position(self, seed):
        """
        Uses a ring method around the target to generate a reset position.
        """
        if seed is not None:
            np.random.seed(seed)
        angle = np.random.uniform(0, 2 * np.pi)

        x_offset = self.reset_pos_distance * np.cos(angle)
        y_offset = self.reset_pos_distance * np.sin(angle)

        reset_pos = self.goal_state + np.array([x_offset, 0, y_offset], dtype=np.float32)

        return reset_pos.astype(np.float32)
