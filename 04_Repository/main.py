from Gym.bullet_drone_env import BulletDroneEnv
from Gym.Wrappers.two_dim_wrapper import TwoDimWrapper
from Gym.Wrappers.position_wrapper import PositionWrapper
from Gym.Algorithms.sacfd import SACfD
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
import argparse
import datetime
import os
import numpy as np
import json
import glob
import matplotlib.pyplot as plt
import torch


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
    model = SACfD(
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
    model.learning_rate = 0.0003
    print("Buffer Size: ", model.replay_buffer.size())

    for obs, next_obs, action, reward, done, info in data:
        model.replay_buffer.add(obs, next_obs, action, reward, done, info)
    print("Buffer Size: ", model.replay_buffer.size())
    model.logger.set_level(10)
    model.train_actor()
    model.learning_rate = 0.0003
  
    visualize_policy(model, data, action_scale=1.0)
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
        # show_in_env(env, transformed_data)
        all_data.extend(transformed_data)
    return all_data


def show_in_env(env, transformed_data):
    start, _, _, _, _, _ = transformed_data[0]
    x, z = start

    state = env.reset(position=np.array([x, 0.0, z]))
    done = False

    # Run through each action in the provided list
    for _, _, action, _, _, _ in transformed_data:
        state, reward, done, _, _ = env.step(action)

        if done:
            print("Episode finished")
            break
    
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
    for item in json_data:
        obs = np.array(item['state'])
        _next_obs = item['next_state']
        x, z = _next_obs
        next_obs = np.array(_next_obs)

        action = np.array(item['action']) * 2.0
        reward = np.array(env.unwrapped.calc_reward([x, 0, z])) + 10
        done = np.array([False])
        info = [{}]
        dataset.append((obs, next_obs, action, reward, done, info))
    for _ in range(1):
        dataset.append((next_obs, next_obs, np.array([0.0, 0.0]), reward, done, info))
    dataset.append((next_obs, next_obs, np.array([0.0, 0.0]), reward, np.array([True]), info))
    return dataset


def print_red(text):
    print(f"\033[31m{text}\033[0m")


def print_green(text):
    print(f"\033[32m{text}\033[0m")


def visualize_policy(model, buffer, action_scale=1.0):
    # Prepare a 2D grid of states
    x_vals = np.linspace(-3, 3, 20)
    z_vals = np.linspace(0, 6, 20)
    X, Z = np.meshgrid(x_vals, z_vals)

    # Arrays to store policy outputs
    U = np.zeros_like(X)
    V = np.zeros_like(Z)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # Formulate the current state
            state = np.array([X[i, j], Z[i, j]])

            # Get the action from the actor network
            action, _ = model.predict(state)

            # Scale the action if necessary
            print("Action:", action)
            U[i, j] = action[0] * action_scale
            V[i, j] = action[1] * action_scale
    
    # Separate coordinates and actions from the buffer
    coords = np.array([pos for pos, _, _, _, _, _ in buffer])
    actions = np.array([rew for _, _, rew, _, _, _ in buffer])

    # Extract X and Z coordinates from the buffer
    X_buf, Z_buf = coords[:, 0], coords[:, 1]

    # Extract U and V action components from the buffer
    U_buf, V_buf = actions[:, 0], actions[:, 1]

    plt.scatter(X_buf, Z_buf, color='blue', label="Buffer Coordinates")

    plt.figure(figsize=(12, 10))

    # Plot the buffer's actions as a quiver plot
    plt.quiver(X_buf, Z_buf, U_buf, V_buf, angles="xy", color='green', label="Buffer Actions")

    # Plot the model's policy as a quiver plot
    plt.quiver(X, Z, U, V, angles="xy", color='red', label="SAC Policy")

    plt.title("Policy Visualization")
    plt.xlabel("X axis")
    plt.ylabel("Z axis")
    plt.xlim([-3.2, 3.2])
    plt.ylim([-0.2, 6.2])
    plt.grid(True)

    plt.show()


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
