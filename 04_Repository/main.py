
from Gym.bullet_drone_env import BulletDroneEnv
from Gym.Wrappers.two_dim_wrapper import TwoDimWrapper
from Gym.Wrappers.position_wrapper import PositionWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
import numpy as np
import argparse
import datetime
import os

def main(algorithm, num_steps, filename):
    print(f"Algorithm: {algorithm}")
    print(f"Number of Steps: {num_steps}")
    if filename != None:
        dir_name = get_dir_name(filename)
    print(f"File Name: {dir_name}")

    os.mkdir(f"models/{dir_name}")
    env = PositionWrapper(TwoDimWrapper(BulletDroneEnv(render_mode="console")))
    if filename != None:
        env = Monitor(env, f"models/{dir_name}/logs")
    if algorithm == "SAC":
        model = train_sac(env, num_steps)
    else:
        print("ERROR: Not yet implemented")
    if filename != None:
        model.save(f"models/{dir_name}/model")
    env.close()

def train_sac(env, num_steps):
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        seed=0,
        batch_size=32,
        policy_kwargs=dict(net_arch=[64, 64]),
    ).learn(num_steps, log_interval=10)
    
    return model


def get_dir_name(prefix):
    # Get the current date and time
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M")
    dir_name = f"{prefix}_{formatted_datetime}"

    return dir_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input parameters.")
    parser.add_argument("-a", "--algorithm", type=str, choices=['SAC', 'SACfD'], required=True,
                        help="Choose the algorithm: 'SAC' or 'SACfD'")
    parser.add_argument("-n", "--num_steps", type=int, required=True,
                        help="Specify the number of steps e.g., 4000")
    parser.add_argument("-f", "--filename", type=str,
                        default=f"simple_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.py",
                        help="Optional: Specify the file name. Defaults to 'simple_YYYYMMDD_HHMM.py'")

    args = parser.parse_args()
    main(args.algorithm, args.num_steps, args.filename)

