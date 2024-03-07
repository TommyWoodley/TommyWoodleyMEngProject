

# %%
# Check simple model conforms to gym env
from stable_baselines3.common.env_checker import check_env
from simple_drone_env import SimpleDroneEnv

env = SimpleDroneEnv()
check_env(env, warn=True)

# %%
