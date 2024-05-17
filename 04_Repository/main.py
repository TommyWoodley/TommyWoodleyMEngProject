from Gym.bullet_drone_env import BulletDroneEnv
from Gym.Wrappers.two_dim_wrapper import TwoDimWrapper
from Gym.Wrappers.position_wrapper import PositionWrapper
from Gym.Wrappers.symmetric_wrapper import SymmetricWrapper
from Gym.Wrappers.memory_wrapper import MemoryWrapper
from Gym.Wrappers.hovering_wrapper import HoveringWrapper
from Gym.Algorithms.sacfd import SACfD
from stable_baselines3 import SAC
from Gym.Wrappers.custom_monitor import CustomMonitor
from Gym.Callbacks.CheckpointCallback import CheckpointCallback
import argparse
import datetime
import os
import numpy as np
import json
import glob


def main(algorithm, num_steps, filename, render_mode):
    print_green(f"Algorithm: {algorithm}")
    print_green(f"Number of Steps: {num_steps}")
    print_green(f"Render Mode: {render_mode}")

    save_data = filename is not None
    if save_data:
        dir_name = get_dir_name(filename)
        os.mkdir(f"/Users/tomwoodley/Desktop/TommyWoodleyMEngProject/04_Repository/models/{dir_name}")
        print_green(f"File Name: {dir_name}")
    else:
        print_red("WARNING: No output or logs will be generated, the model will not be saved!")

    env = HoveringWrapper(MemoryWrapper(PositionWrapper(TwoDimWrapper(
        SymmetricWrapper(BulletDroneEnv(render_mode=render_mode))))))
    if save_data:
        env = CustomMonitor(env,
                            f"/Users/tomwoodley/Desktop/TommyWoodleyMEngProject/04_Repository/models/{dir_name}/logs")

        checkpoint_callback = CheckpointCallback(
            save_freq=5000,
            save_path=f"/models/{dir_name}/training_logs/",
            name_prefix="checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )
    else:
        checkpoint_callback = None

    if algorithm == "SAC":
        model = train_sac(env, num_steps, checkpoint_callback)
    elif algorithm == "SACfD":
        model = train_sacfd(env, num_steps, checkpoint_callback)
    else:
        print_red("ERROR: Not yet implemented",)
    print_green("TRAINING COMPLETE!")
    if save_data:
        model.save(f"/Users/tomwoodley/Desktop/TommyWoodleyMEngProject/04_Repository/models/{dir_name}/model")
        print_green("Model Saved")
    env.close()

    if save_data:
        generate_graphs(directory=f"/Users/tomwoodley/Desktop/TommyWoodleyMEngProject/04_Repository/models/{dir_name}")


def train_sac(env, num_steps, callback=None):
    model = SAC(
        "MlpPolicy",
        env,
        seed=0,
        batch_size=32,
        learning_rate=linear_schedule(0.0002),
        policy_kwargs=dict(net_arch=[64, 64]),
    ).learn(num_steps, log_interval=10, progress_bar=True, callback=callback)

    return model


def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return 0.00005 + progress_remaining * initial_value

    return func


def train_sacfd(env, num_steps, callback=None):
    # from utils.graphics.plot_actor_policy import visualize_policy

    model = SACfD(
        "MlpPolicy",
        env,
        seed=0,
        batch_size=32,
        policy_kwargs=dict(net_arch=[128, 128, 64]),
        learning_starts=0,
        gamma=0.96,
        learning_rate=linear_schedule(0.0002),
    )
    from stable_baselines3.common.logger import configure
    tmp_path = "/tmp/sb3_log/"
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    data = get_buffer_data(env)
    # show_in_env(env=env, transformed_data=data)
    print("Buffer Size: ", model.replay_buffer.size())

    for obs, next_obs, action, reward, done, info in data:
        model.replay_buffer.add(obs, next_obs, action, reward, done, info)
    print("Buffer Size: ", model.replay_buffer.size())

    # model.train_actor()
    # visualize_policy(model, data, action_scale=1.0)
    print_green("Pretraining Complete!")

    model.learn(num_steps, log_interval=10, progress_bar=True, callback=callback)

    return model


def get_buffer_data(env):
    dir = "/Users/tomwoodley/Desktop/TommyWoodleyMEngProject/04_Repository/Data/PreviousWorkTrajectories/rl_demos"
    return load_all_data(env, dir)


def load_all_data(env, directory):
    pattern = f"{directory}/rl_demo_approaching_angle_*.json"
    files = glob.glob(pattern)
    all_data = []
    for file in files:
        json_data = load_json(file)
        transformed_data = convert_data(env, json_data)
        # show_in_env(env, transformed_data)
        all_data.extend(transformed_data)
    return all_data


# Shows the demonstration data in the enviornment - useful for verification purposes
# TODO: Add a setting to enable this
def show_in_env(env, transformed_data):
    start, _, _, _, _, _ = transformed_data[0]
    x, z = start

    state = env.reset(position=np.array([x, 0.0, z]))
    done = False

    # Run through each action in the provided list
    for _, _, action, _, _, _ in transformed_data:
        state, reward, done, truncated, _ = env.step(action)

        if done or truncated:
            print("Episode finished")
            break

    while not done and not truncated:
        _, _, done, truncated, _ = env.step(np.array([0.0, 0.0]))
        if done:
            print("Episode finished")
        if truncated:
            print("Episode Truncated")

    env.reset()

    print(state)


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
    num = 0
    for item in json_data:
        obs = np.append(np.array(item['state']), num / 100.0)
        _next_obs = item['next_state']
        _, _, _, _, x, z = _next_obs
        next_obs = np.append(np.array(_next_obs), (num + 1) / 100.0)

        # Normalised action TODO: Define this relative to the env so it's consistent
        action = np.array(item['action']) * 4.0
        reward = np.array(env.unwrapped.calc_reward([x, 0, z]))
        done = np.array([False])
        info = [{}]
        dataset.append((obs, next_obs, action, reward, done, info))
        num = num + 1
    for _ in range(1):  # Adds an extra action on the end which helps with wrapping.
        dataset.append((next_obs, next_obs, np.array([0.0, 0.0]), reward, done, info))
    dataset.append((next_obs, next_obs, np.array([0.0, 0.0]), reward, np.array([True]), info))
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
    parser.add_argument("-v", "--visualise", action="store_true",
                        help="Optional: Visualise the training - This is significantly slower.")

    args = parser.parse_args()
    main(args.algorithm, args.num_steps, args.filename, "console" if not args.visualise else "human")
