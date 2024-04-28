from Gym.bullet_drone_env import BulletDroneEnv
from Gym.Wrappers.two_dim_wrapper import TwoDimWrapper
from Gym.Wrappers.position_wrapper import PositionWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
import argparse
import datetime
import os
import numpy as np
import json
import glob


def main(algorithm, num_steps, filename, render_mode):
    print_green(f"Algorithm: {algorithm}")
    print_green(f"Number of Steps: {num_steps}")
    save_data = filename is not None
    if save_data:
        dir_name = get_dir_name(filename)
        os.mkdir(f"models/{dir_name}")
        print_green(f"File Name: {dir_name}")
    else:
        print_red("WARNING: No output or logs will be generated, the model will not be saved!")

    env = PositionWrapper(TwoDimWrapper(BulletDroneEnv(render_mode=render_mode)))
    if save_data:
        env = Monitor(env, f"models/{dir_name}/logs")
    if algorithm == "SAC":
        model = train_sac(env, num_steps)
    elif algorithm == "SACfD":
        model = train_sacfd(env, num_steps)
    else:
        print_red("ERROR: Not yet implemented")
    print_green("TRAINING COMPLETE!")
    if save_data:
        model.save(f"models/{dir_name}/model")
        print_green("Model Saved")
    env.close()

    generate_graphs(directory=f"models/{dir_name}")


def train_sac(env, num_steps):
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        seed=0,
        batch_size=32,
        policy_kwargs=dict(net_arch=[64, 64]),
    ).learn(num_steps, log_interval=10, progress_bar=True)

    return model


def train_sacfd(env, num_steps):
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        seed=0,
        batch_size=32,
        policy_kwargs=dict(net_arch=[64, 64]),
        learning_starts=0,
    )
    from stable_baselines3.common.logger import configure
    tmp_path = "/tmp/sb3_log/"
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    data = get_buffer_data(env)
    model.learning_rate = 0.003
    print("Buffer Size: ", model.replay_buffer.size())

    for obs, next_obs, action, reward, done, info in data:
        model.replay_buffer.add(obs, next_obs, action, reward, done, info)
    print("Buffer Size: ", model.replay_buffer.size())
    model.train(5000, 32)
    model.learning_rate = 0.0003
    print_green("Pretraining Complete!")
    model.learn(num_steps, log_interval=10, progress_bar=True)

    return model


def get_buffer_data(env):
    dir = "Data/PreviousWorkTrajectories/rl_demos"
    return load_all_data(env, dir)


def load_all_data(env, directory):
    pattern = f"{directory}/rl_demo_approaching_angle_*.json"
    files = glob.glob(pattern)
    all_data = []
    for file in files:
        json_data = load_json(file)
        transformed_data = convert_data(env, json_data)
        all_data.extend(transformed_data)
    return all_data


def generate_graphs(directory):
    from models.generate_reward_graph_from_logs import read_csv_file
    from models.visualise_reward import plot_reward_visualisation
    from models.sample_trajectories_from_model import sample_trajectories

    # visualise reward function used
    print_green("Generating Reward Visualisation")
    plot_reward_visualisation(directory, show=False)

    # visualise training rewards
    print_green("Generating Reward Logs")
    read_csv_file(f"{directory}/logs.monitor.csv", show=False)

    # visualise sample trajectories
    print_green("Generating Sample Trajectories")
    sample_trajectories(directory, show=False)


def get_dir_name(prefix):
    # Get the current date and time
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M")
    dir_name = f"{prefix}_{formatted_datetime}"

    return dir_name


# Load the JSON data from a file
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def convert_data(env, json_data):
    dataset = []
    for item in json_data:
        obs = np.array(item['state'])
        _next_obs = item['next_state']
        x, z = _next_obs
        next_obs = np.array(_next_obs)

        action = np.array(item['action'])
        reward = np.array(env.unwrapped.calc_reward([x, 0, z]))
        done = np.array([False])
        info = [{}]
        dataset.append((obs, next_obs, action, reward, done, info))
    return dataset


def print_red(text):
    print(f"\033[31m{text}\033[0m")


def print_green(text):
    print(f"\033[32m{text}\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input parameters.")
    parser.add_argument("-a", "--algorithm", type=str, choices=['SAC', 'SACfD'], required=True,
                        help="Choose the algorithm: 'SAC' or 'SACfD'")
    parser.add_argument("-n", "--num_steps", type=int, required=True,
                        help="Specify the number of steps e.g., 4000")
    parser.add_argument("-f", "--filename", type=str,
                        default=None,
                        help="Optional: Specify the file name. Defaults to 'simple_YYYYMMDD_HHMM.py'")
    parser.add_argument("-v", "--visualise", type=bool,
                        default=False,
                        help="Optional: Visualise the training - This is significantly slower.")

    args = parser.parse_args()
    main(args.algorithm, args.num_steps, args.filename, "console" if not args.visualise else "human")
