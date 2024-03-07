import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SimpleDroneEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """

    metadata = {"render_modes": ["console"]}

    def __init__(self, x_range=(-10, 10), z_range=(-10, 10), render_mode="console"):
        super(SimpleDroneEnv, self).__init__()
        self.render_mode = render_mode
        self.num_steps = 0

        # Action, Observation and Goal State
        self.goal_state = np.array([0.0, 0.0])  # Goal state
        self.action_space = spaces.Box(low=np.array([-2, -2]), high=np.array([2, 2]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([x_range[0], z_range[0]]),
                                            high=np.array([x_range[1], z_range[1]]),
                                            dtype=np.float32)

        self.state = None  # Initial position will be set in reset method

    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)
        self.state = self.observation_space.sample()
        self.num_steps = 0
        return self.state, {}

    def step(self, action):
        # Update the agent's position based on the action taken
        self.state = np.clip(self.state + action, self.observation_space.low, self.observation_space.high)
        self.num_steps += 1

        # Assuming no specific goal for simplicity, so no reward or termination logic
        distance_to_goal = np.linalg.norm(self.state - self.goal_state)
        reward = -distance_to_goal
        terminated = distance_to_goal < 2.0  # Example threshold
        truncated = self.num_steps > 100
        info = {"distance_to_goal": distance_to_goal}

        return self.state, reward, terminated, truncated, info

    def render(self):
        # agent is represented as a cross, rest as a dot
        if self.render_mode == "console":
            print(f'Agent position: x={self.state[0]}, z={self.state[1]}')

    def close(self):
        pass