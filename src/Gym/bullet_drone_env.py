import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple

from TetherModel.Environment.tethered_drone_simulator import TetheredDroneSimulator
from Gym.Rewards.reward_system import RewardSystem


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
    centre_pos = np.array([0.0, 0.0, 3.0])  # Goal state
    reset_pos_distance = 2.0

    def __init__(self, render_mode: str = "human", phase: str = "all") -> None:
        super(BulletDroneEnv, self).__init__()
        self.simulator = TetheredDroneSimulator(drone_pos=self._generate_reset_position(42),
                                                gui_mode=(render_mode == "human"))
        self.action_space = spaces.Box(low=np.array([-0.001, -0.001, -0.001]),
                                       high=np.array([0.001, 0.001, 0.001]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.render_mode = render_mode
        self.num_steps = 0
        self.should_render = True
        self.reward = RewardSystem(phase)

    def reset(self, seed: int = None, options: Dict[str, Any] = None,
              degrees: int = None, position=None) -> Tuple[np.ndarray, Dict[Any, Any]]:
        super().reset(seed=seed, options=options)
        if position is not None:  # position and degrees are here for testing and visualisation purposes
            reset_pos = position
        elif degrees is not None:
            reset_pos = self._generate_reset_position_from_degrees(degrees)
        else:
            reset_pos = self._generate_reset_position(seed)
        self.simulator.reset(reset_pos)
        self.num_steps = 0

        aug_state = np.append(reset_pos, 0.0).astype(np.float32)
        return aug_state, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        has_collided, dist_tether_branch, dist_drone_branch, dist_drone_ground, num_wraps = self.simulator.step(action)
        self.render()
        state = self.simulator.drone_pos

        self.num_steps += 1
        aug_state = np.append(state, num_wraps).astype(np.float32)

        reward, terminated = self.reward.calculate(state, has_collided, dist_tether_branch, dist_drone_branch, num_wraps)

        info = {"distance_to_goal": -reward, "has_crashed": bool(dist_drone_branch < 0.1), "num_wraps": num_wraps}

        return aug_state, reward, terminated, False, info

    def render(self) -> None:
        if self.should_render:
            self._render()

    def _render(self) -> None:
        # agent is represented as a cross, rest as a dot
        if self.render_mode == "console":
            print(f'Agent position: {self.simulator.drone_pos}')

    def close(self) -> None:
        self.reward.end()
        if hasattr(self, 'simulator'):
            self.simulator.close()

    def _generate_reset_position(self, seed):
        """
        Uses a ring method around the target to generate a reset position from seed
        """
        if seed is not None:
            np.random.seed(seed)
        angle = np.random.uniform(- np.pi / 3, np.pi / 3)

        return self._generate_reset_position_from_radians(angle)

    def _generate_reset_position_from_degrees(self, degrees):
        return self._generate_reset_position_from_radians(np.radians(degrees))

    def _generate_reset_position_from_radians(self, radians):

        x_offset = self.reset_pos_distance * np.cos(radians)
        y_offset = self.reset_pos_distance * np.sin(radians)

        reset_pos = self.centre_pos + np.array([x_offset, 0, y_offset], dtype=np.float32)
        return reset_pos.astype(np.float32)

    # Visualisation funtion
    def calc_reward(self, state):
        branch_pos = np.array([0.0, 0.0, 2.7])  # Branch position
        tether_pos = state - np.array([0, 0, 0.5])
        dist_tether_branch = np.linalg.norm(tether_pos - branch_pos)
        dist_drone_branch = np.linalg.norm(state - branch_pos)
        has_collided = bool(dist_tether_branch < 0.1)

        reward, _ = self.reward.calculate(state, has_collided, dist_tether_branch, dist_drone_branch,
                                              num_wraps=1.0)
        return reward
