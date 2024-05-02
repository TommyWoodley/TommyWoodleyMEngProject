import sys
import pytest
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Gym.bullet_drone_env import BulletDroneEnv
from Gym.Wrappers.two_dim_wrapper import TwoDimWrapper
from Gym.Wrappers.position_wrapper import PositionWrapper
from stable_baselines3.common.env_checker import check_env


@pytest.fixture
def env():
    # Create the base environment and wrap it with PositionWrapper
    env = PositionWrapper(TwoDimWrapper(BulletDroneEnv(render_mode="console")))
    env.reset()
    yield env

    env.close()


def test_bullet_env():
    util_test_valid_env(BulletDroneEnv(render_mode="console"))


def test_two_dim_bullet_env():
    util_test_valid_env(TwoDimWrapper(BulletDroneEnv(render_mode="console")))


def test_position_wrapped_two_dim_bullet_env():
    util_test_valid_env(PositionWrapper(TwoDimWrapper(BulletDroneEnv(render_mode="console"))))


def test_environment_truncation(env):
    truncated = False

    for _ in range(60):
        assert not truncated, "The environment truncated too early."

        action = env.action_space.sample()  # Sample a random action
        _, _, _, truncated, _ = env.step(action)

    assert truncated, "The environment did not return truncated after 60 steps."


def test_position_wrapper_action_space(env):
    assert env.action_space.low.min() == -1, "Action space lower bound should be -1"
    assert env.action_space.high.max() == 1, "Action space upper bound should be 1"


def util_test_valid_env(env):
    check_env(env, warn=True)
    env.close()
