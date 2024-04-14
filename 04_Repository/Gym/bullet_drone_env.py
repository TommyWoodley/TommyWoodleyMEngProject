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
    goal_state = np.array([0.0, 0.0, 3.2])  # Drone 
    branch_position = np.array([0.0, 0.0, 2.7]) # Branch position
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

    def reset(self, seed: int = None, options: Dict[str, Any] = None,
              degrees: int = None) -> Tuple[np.ndarray, Dict[Any, Any]]:
        super().reset(seed=seed, options=options)
        if degrees is not None:
            reset_pos = self._generate_reset_position_from_degrees(degrees)
        else:
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
        reward = - distance + self.calculate_drone_hit_branch_reward(state=state)
        return reward, bool(distance < 0.1), False
    
    def calculate_drone_hit_branch_reward(self, state: np.ndarray) -> float:
        """
        Calculate reward for drone hitting the branch: Ring based
        - Inner: -10, Outer: 0, Between: 0:-5
        """
        dist_to_branch = np.linalg.norm(state - self.branch_position)
        if dist_to_branch < 0.1:  # A collision
            return -5.0
        elif dist_to_branch < 0.3: # Quite close
            return BulletDroneEnv.interpolate_distance(dist_to_branch, 0.1, -5, min_value=0.3)
        else:
            return 0
    
    def interpolate_distance(distance, max_value, max_reward, min_value=0, min_reward=0):
        return min_reward + ((max_reward - min_reward) * (distance - min_value)) / (max_value - min_value)

    def _generate_reset_position(self, seed):
        """
        Uses a ring method around the target to generate a reset position from seed
        """
        if seed is not None:
            np.random.seed(seed)
        angle = np.random.uniform(0, 2 * np.pi)

        return self._generate_reset_position_from_radians(angle)

    def _generate_reset_position_from_degrees(self, degrees):
        return self._generate_reset_position_from_radians(np.radians(degrees))

    def _generate_reset_position_from_radians(self, radians):

        x_offset = self.reset_pos_distance * np.cos(radians)
        y_offset = self.reset_pos_distance * np.sin(radians)

        reset_pos = self.goal_state + np.array([x_offset, 0, y_offset], dtype=np.float32)
        return reset_pos.astype(np.float32)

    # Visualisation funtion
    def calc_reward(self, state):
        reward, _, _ = self.reward_fun(state)
        return reward
