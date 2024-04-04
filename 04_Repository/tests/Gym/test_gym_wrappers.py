import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Gym.bullet_drone_env import BulletDroneEnv
from Gym.Wrappers.two_dim_wrapper import TwoDimWrapper
from Gym.Wrappers.position_wrapper import PositionWrapper
from stable_baselines3.common.env_checker import check_env


def test_bullet_env():
    util_test_valid_env(BulletDroneEnv(render_mode="console"))


def test_two_dim_bullet_env():
    util_test_valid_env(TwoDimWrapper(BulletDroneEnv(render_mode="console")))


def test_position_wrapped_two_dim_bullet_env():
    util_test_valid_env(PositionWrapper(TwoDimWrapper(BulletDroneEnv(render_mode="console"))))


def util_test_valid_env(env):
    check_env(env, warn=True)
    env.close()
