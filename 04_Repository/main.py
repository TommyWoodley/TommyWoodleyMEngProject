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
from utils.util_graphics import print_green, print_red
from utils.util_file import load_json, make_dir
from utils.args_parsing import StoreDict
import argparse
import numpy as np
import glob

DEFAULT_DEMO_PATH="/Users/tomwoodley/Desktop/TommyWoodleyMEngProject/04_Repository/Data/PreviousWorkTrajectories/rl_demos"
DEFAULT_CHECKPOINT=5000

# ---------------------------------- RL UTIL ----------------------------------

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


# Shows the demonstration data in the enviornment - useful for verification purpose
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


# ----------------------------------- DATA ------------------------------------

def get_buffer_data(env, directory, show_demos_in_env):
    pattern = f"{directory}/rl_demo_approaching_angle_*.json"
    files = glob.glob(pattern)
    all_data = []
    for file in files:
        json_data = load_json(file)
        transformed_data = convert_data(env, json_data)
        if show_demos_in_env:
            show_in_env(env, transformed_data)
        all_data.extend(transformed_data)
    return all_data


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

# ---------------------------- ENVIRONMENT & AGENT ----------------------------

def get_checkpointer(should_save, dir_name, checkpoint):
    if should_save and checkpoint is not None:
        checkpoint_callback = CheckpointCallback(
                save_freq=checkpoint,
                save_path=f"/models/{dir_name}/training_logs/",
                name_prefix="checkpoint",
                save_replay_buffer=False,
                save_vecnormalize=True,
            )
        return checkpoint_callback
    return None


def get_env(dir_name, render_mode):
    env = HoveringWrapper(MemoryWrapper(PositionWrapper(TwoDimWrapper(
        SymmetricWrapper(BulletDroneEnv(render_mode=render_mode))))))

    if dir_name is not None:
        env = CustomMonitor(env, f"/models/{dir_name}/logs")

    return env


def get_agent(algorithm, env, demo_path, show_demos_in_env, hyperparams):
    _policy = "MlpPolicy"
    _seed = 0
    _batch_size = hyperparams.get("batch_size", 32)
    _policy_kwargs = dict(net_arch=[128, 128, 64])
    _lr_schedular = linear_schedule(hyperparams.get("lr", 0.0002))

    print_green(f"Hyperparamters: seed={_seed}, batch_size={_batch_size}, policy_kwargs={_policy_kwargs}" + (
                f"lr={_lr_schedular}"))

    if algorithm == "SAC":
        agent = SAC(
            _policy,
            env,
            seed=_seed,
            batch_size=_batch_size,
            learning_rate=_lr_schedular,
            policy_kwargs=_policy_kwargs,
        )
    elif algorithm == "SACfD":
        agent = SACfD(
            _policy,
            env,
            seed=_seed,
            batch_size=_batch_size,
            policy_kwargs=_policy_kwargs,
            learning_starts=0,
            gamma=0.96,
            learning_rate=_lr_schedular
        )
        pre_train(agent, env, demo_path, show_demos_in_env)
        
    else:
        print_red("ERROR: Not yet implemented",)
    return agent

def pre_train(agent, env, demo_path, show_demos_in_env):
    from stable_baselines3.common.logger import configure
    tmp_path = "/tmp/sb3_log/"
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    agent.set_logger(new_logger)

    data = get_buffer_data(env, demo_path, show_demos_in_env)
    print("Buffer Size: ", agent.replay_buffer.size())

    for obs, next_obs, action, reward, done, info in data:
        agent.replay_buffer.add(obs, next_obs, action, reward, done, info)
    print("Buffer Size: ", agent.replay_buffer.size())
    print_green("Pretraining Complete!")


# ----------------------------------- MAIN ------------------------------------


def main(algorithm, timesteps, filename, render_mode, demo_path, should_show_demo, checkpoint, hyperparams):
    save_data = filename is not None
    dir_name = make_dir(filename)

    env = get_env(dir_name, render_mode)
    checkpoint_callback = get_checkpointer(save_data, dir_name, checkpoint)

    agent = get_agent(algorithm, env, demo_path, should_show_demo, hyperparams)

    agent.learn(timesteps, log_interval=10, progress_bar=True, callback=checkpoint_callback)

    print_green("TRAINING COMPLETE!")

    if save_data:
        agent.save(f"/models/{dir_name}/model")
        print_green("Model Saved")
    env.close()

    if save_data:
        generate_graphs(directory=f"/models/{dir_name}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Reinforcement Learning Training for Tethered Drone Perching")

    # Number of timesteps
    parser.add_argument('-t', '--timesteps', type=int, required=True, help='Number of timesteps for training (e.g., 40000)')

    # Choice of algorithm
    parser.add_argument('-algo', '--algorithm', type=str, choices=['SAC', 'SACfD'], required=True, help='Choice of algorithm: SAC or SACfD')

    # Output filename for logs
    parser.add_argument('-o', '--output-filename', type=str, default=None, help='Filename for storing logs')

    # Graphical user interface
    parser.add_argument('-gui', '--gui', action='store_true', help='Enable graphical user interface')

    # Demonstration path
    parser.add_argument('--demo-path', type=str, default=DEFAULT_DEMO_PATH, help='Path to demonstration files (default: /path/to/default/directory)')

    # Show demonstrations in visual environment
    parser.add_argument('--show-demo', action='store_true', help='Show demonstrations in visual environment')

    # Checkpoint episodes
    parser.add_argument('--checkpoint-episodes', type=int, default=DEFAULT_CHECKPOINT, help='Frequency of checkpoint episodes (default: 5000)')
    parser.add_argument('--no-checkpoint', action='store_true', help='Perform NO checkpointing during training.')

    parser.add_argument("-params", "--hyperparams", type=str, nargs="+", action=StoreDict, help="Overwrite hyperparameter (e.g. lr:0.01 batch_size:10)",)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    algorithm = args.algorithm
    timesteps = args.timesteps
    filename = args.output_filename
    render_mode = "human" if args.gui else "console"
    demo_path = args.demo_path
    should_show_demo = args.show_demo
    checkpoint = args.checkpoint_episodes
    if args.no_checkpoint:
        checkpoint = None

    if algorithm != "SACfD" and demo_path is not None:
        print_red("WARNING: Demo path provided will NOT be used by this algorithm!")


    print_green(f"Algorithm: {algorithm}")
    print_green(f"Timesteps: {timesteps}")
    print_green(f"Render Mode: {render_mode}")
    if filename is None:
        print_red("WARNING: No output or logs will be generated, the model will not be saved!")
    else:
        print_green(f"File Name: {filename}")

    if algorithm == "SACfD":
        print_green(f"Demo Path: {demo_path}")
    print_green(f"Checkpointing: {checkpoint}")

    accpetable_hp = ["lr", "batch_size"]
    hyperparams = args.hyperparams if args.hyperparams is not None else dict()
    for key, val in hyperparams.items():
        if key in accpetable_hp:
            print_green(f"\t{key}: {val}")
        else:
            print_red(f"\nUnknown Hyperparameter: {key}")

    main(algorithm, timesteps, filename, render_mode, demo_path, should_show_demo, checkpoint, hyperparams)
