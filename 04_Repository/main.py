# %%
from Gym.simple_drone_env import SimpleDroneEnv
from Gym.bullet_drone_env import BulletDroneEnv
from Gym.Wrappers.two_dim_wrapper import TwoDimWrapper
from Gym.Wrappers.position_wrapper import PositionWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

# %%
# Check BulletDroneEnv in 2d wrapper confors to gym.Env
env = PositionWrapper(TwoDimWrapper(BulletDroneEnv()))
check_env(env, warn=True)
env.close()

# %%

env = PositionWrapper(TwoDimWrapper(BulletDroneEnv()))
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.0001 * np.ones(n_actions))

default_model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    seed=0,
    batch_size=64,
    action_noise=action_noise,
    policy_kwargs=dict(net_arch=[64, 64]),
).learn(5)

env.close()

# %%
