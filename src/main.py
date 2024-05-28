from utils.util_graphics import print_green, print_red
from utils.util_file import load_json, make_dir
from utils.args_parsing import StoreDict
from utils.rl.lr_schedular import LinearLearningRateSchedule
import argparse
import numpy as np
import glob
import os

DEMO_PATH = "/Users/tomwoodley/Desktop/TommyWoodleyMEngProject/src/Data/PreviousWorkTrajectories/rl_demos"
DEFAULT_CHECKPOINT = 5000

# ---------------------------------- RL UTIL ----------------------------------


def generate_graphs(directory, phase="all"):
    from plot_graphs import plot_reward_visualisation, read_csv_file
    from models.sample_trajectories_from_model import sample_trajectories

    # visualise reward function used
    print_green("Generating Reward Visualisation")
    plot_reward_visualisation(directory, show=False)

    # visualise training rewards
    print_green("Generating Reward Logs")
    read_csv_file(f"{directory}/logs.monitor.csv", show=False)

    # visualise sample trajectories
    print_green("Generating Sample Trajectories")
    sample_trajectories(directory, show=False, phase=phase)


# Shows the demonstration data in the enviornment - useful for verification purpose
def show_in_env(env, transformed_data):
    start, _, _, _, _, _ = transformed_data[0]
    x, z, _, _, _, _, _, _, _, _ = start

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
        _, _, _, _, _, _, x, z, t = item['state']
        obs = np.append(np.array([x, z, t]), num / 100.0)
        _next_obs = item['next_state']
        _, _, _, _, _, _, x, z, t = _next_obs
        next_obs = np.append(np.array(np.array([x, z, t])), (num + 1) / 100.0)

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


def get_checkpointer(should_save, dir_name, checkpoint, phase="all"):
    from Gym.Callbacks.CheckpointCallback import CheckpointCallback

    if should_save and checkpoint is not None:
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint,
            save_path=f"models/{dir_name}/training_logs/",
            name_prefix="checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=True,
            phase=phase)
        return checkpoint_callback
    return None


def get_env(dir_name, render_mode, phase):
    from Gym.bullet_drone_env import BulletDroneEnv
    from Gym.Wrappers.two_dim_wrapper import TwoDimWrapper
    from Gym.Wrappers.position_wrapper import PositionWrapper
    from Gym.Wrappers.symmetric_wrapper import SymmetricWrapper
    from Gym.Wrappers.hovering_wrapper import HoveringWrapper
    from Gym.Wrappers.custom_monitor import CustomMonitor

    env = HoveringWrapper(PositionWrapper(TwoDimWrapper(
        SymmetricWrapper(BulletDroneEnv(render_mode=render_mode, phase=phase)))))

    if dir_name is not None:
        env = CustomMonitor(env, f"models/{dir_name}/logs")

    return env


def get_agent(algorithm, env, demo_path, show_demos_in_env, hyperparams, timesteps):
    from Gym.Algorithms.sacfd import SACfD
    from Gym.Algorithms.dual_buffer import DualReplayBuffer
    from stable_baselines3 import SAC

    _policy = "MlpPolicy"
    _seed = 0
    _batch_size = hyperparams.get("batch_size", 64)
    _policy_kwargs = dict(net_arch=[128, 128, 64])
    def linear_decay_weighting(current_step: int, initial_weighting: int = 0.5, final_weighting: int = 0.01, decay_steps: int = timesteps) -> float:
        progress = min(1.0, current_step / decay_steps)
        return initial_weighting - progress * (initial_weighting - final_weighting)
    _replay_buffer_kwargs = dict(weighting_function=lambda step: linear_decay_weighting(step),)
    _lr_schedular = LinearLearningRateSchedule(hyperparams.get("lr", 0.0002))

    print_green(f"Hyperparamters: seed={_seed}, batch_size={_batch_size}, policy_kwargs={_policy_kwargs}, " + (
                f"lr={_lr_schedular}"))

    if algorithm == "SAC":
        agent = SAC(
            _policy,
            env,
            seed=_seed,
            batch_size=_batch_size,
            learning_rate=_lr_schedular,
            gamma=0.96,
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
            learning_rate=_lr_schedular,
            replay_buffer_class=DualReplayBuffer,
            replay_buffer_kwargs=_replay_buffer_kwargs,
        )
        pre_train(agent, env, demo_path, show_demos_in_env)

    else:
        print_red("ERROR: Not yet implemented",)
    return agent


def get_existing_agent(existing_agent_path, env):
    from Gym.Algorithms.sacfd import SACfD

    try:
        # Check if the file path ends with .zip
        if not existing_agent_path.endswith('.zip'):
            raise ValueError("The file path must end with .zip")

        # Load the SAC model directly from the zip file
        model = SACfD.load(existing_agent_path)
        model.set_env(env)

        # Check for the replay buffer file in the same directory
        replay_buffer_path = os.path.join(os.path.dirname(existing_agent_path), 'replay_buffer.pkl')
        if os.path.exists(replay_buffer_path):
            print("A replay buffer is being loaded.")
            model.load_replay_buffer(replay_buffer_path)

        print(f"Buffer Size: {model.replay_buffer.size()}")
        return model
    except Exception as e:
        print(f"ERROR: {str(e)}")
        exit(1)


def pre_train(agent, env, demo_path, show_demos_in_env):
    from stable_baselines3.common.logger import configure
    tmp_path = "/tmp/sb3_log/"
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    agent.set_logger(new_logger)

    data = get_buffer_data(env, demo_path, show_demos_in_env)
    print("Buffer Size: ", agent.replay_buffer.size())

    for i in range(5):
        for obs, next_obs, action, reward, done, info in data:
            agent.replay_buffer._add_offline(obs, next_obs, action, reward, done, info)
    print("Online Buffer Size: ", agent.replay_buffer.online_replay_buffer.size())
    print("Offline Buffer Size: ", agent.replay_buffer.offline_replay_buffer.size())
    print_green("Pretraining Complete!")


# ----------------------------------- MAIN ------------------------------------


def main(algorithm, timesteps, filename, render_mode, demo_path, should_show_demo, checkpoint, hyperparams,
         existing_agent, save_replay_buffer, phase):
    save_data = filename is not None
    dir_name = make_dir(filename)

    env = get_env(dir_name, render_mode, phase)
    checkpoint_callback = get_checkpointer(save_data, dir_name, checkpoint, phase)

    if existing_agent is None:
        agent = get_agent(algorithm, env, demo_path, should_show_demo, hyperparams, timesteps)
    else:
        agent = get_existing_agent(existing_agent, env)

    agent.learn(timesteps, log_interval=10, progress_bar=True, callback=checkpoint_callback)

    print_green("TRAINING COMPLETE!")

    if save_data:
        agent.save(f"models/{dir_name}/model")
        if save_replay_buffer:
            agent.save_replay_buffer(f"models/{dir_name}/replay_buffer")
        print_green("Model Saved")
    env.close()

    if save_data:
        generate_graphs(directory=f"models/{dir_name}", phase=phase)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Reinforcement Learning Training for Tethered Drone Perching")

    # Number of timesteps
    parser.add_argument('-t', '--timesteps', type=int, required=True,
                        help='Number of timesteps for training (e.g., 40000)')

    # Choice of algorithm
    parser.add_argument('-algo', '--algorithm', type=str, choices=['SAC', 'SACfD'], required=True,
                        help='Choice of algorithm: SAC or SACfD')

    # Output filename for logs
    parser.add_argument('-o', '--output-filename', type=str, default=None, help='Filename for storing logs')
    parser.add_argument('--save-replay-buffer', action='store_true', help='Saves the replay model from the buffer.')

    # Graphical user interface
    parser.add_argument('-gui', '--gui', action='store_true', help='Enable graphical user interface')

    # Demonstration path
    parser.add_argument('--demo-path', type=str, default=DEMO_PATH,
                        help=f"Path to demonstration files (default: {DEMO_PATH}")

    # Show demonstrations in visual environment
    parser.add_argument('--show-demo', action='store_true', help='Show demonstrations in visual environment')

    # Checkpoint episodes
    parser.add_argument('--checkpoint-episodes', type=int, default=DEFAULT_CHECKPOINT,
                        help='Frequency of checkpoint episodes (default: 5000)')
    parser.add_argument('--no-checkpoint', action='store_true', help='Perform NO checkpointing during training.')

    parser.add_argument("-params", "--hyperparams", type=str, nargs="+", action=StoreDict,
                        help="Overwrite hyperparameter (e.g. lr:0.01 batch_size:10)",)

    # Continue Training
    parser.add_argument("-i", "--trained-agent", help="Path to a pretrained agent to continue training",
                        default=None, type=str, required=False)

    # Train particular phase
    parser.add_argument("-p", "--phase", type=str, choices=["approaching", "all"], default="all",
                        help="Train a particular phase of a the system.")

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
    trained_agent = args.trained_agent
    save_replay_buffer = args.save_replay_buffer
    phase = args.phase

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

    if trained_agent is not None:
        print_green(f"Using pre-trained agent: {trained_agent}")

    accpetable_hp = ["lr", "batch_size"]
    hyperparams = args.hyperparams if args.hyperparams is not None else dict()
    for key, val in hyperparams.items():
        if key in accpetable_hp:
            print_green(f"\t{key}: {val}")
        else:
            print_red(f"\nUnknown Hyperparameter: {key}")

    main(algorithm, timesteps, filename, render_mode, demo_path, should_show_demo, checkpoint, hyperparams,
         trained_agent, save_replay_buffer, phase)
