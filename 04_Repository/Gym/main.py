

# %%
# Check simple model conforms to gym env
from stable_baselines3.common.env_checker import check_env
from simple_drone_env import SimpleDroneEnv

env = SimpleDroneEnv()
check_env(env, warn=True)

# %%
from simple_drone_env import SimpleDroneEnv

env = SimpleDroneEnv()

obs, _ = env.reset()
env.render()

print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())

n_steps = 20
for step in range(n_steps):
    print(f"Step {step + 1}")
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
    print("obs=", obs, "reward=", reward, "done=", done)
    env.render()
    if done:
        print("Goal reached!", "reward=", reward)
        break
# %%
from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

env = SimpleDroneEnv()
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.4 * np.ones(n_actions))

default_model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    seed=0,
    batch_size=64,
    action_noise=action_noise,
    policy_kwargs=dict(net_arch=[64, 64]),
).learn(50_000)

# %%
# Test the trained agent
# using the vecenv
obs, _ = env.reset()
n_steps = 20
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    print(f"Step {step + 1}")
    print("Action: ", action)
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print("obs=", obs, "reward=", reward, "done=", done)
    env.render()
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Goal reached!", "reward=", reward)
        break
# %%
