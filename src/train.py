import optuna
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from main import get_env, get_agent
import numpy as np

DEMO_PATH = "/Users/tomwoodley/Desktop/TommyWoodleyMEngProject/04_Repository/Data/PreviousWorkTrajectories/rl_demos"
DEFAULT_CHECKPOINT = 5000

# Define the objective function
def objective(trial):
    # Define the hyperparameters to tune
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    gamma = trial.suggest_loguniform('gamma', 0.9, 0.9999)
    tau = trial.suggest_loguniform('tau', 1e-3, 1)
    target_update_interval = trial.suggest_int('target_update_interval', 1, 1000)
    
    # Define the network architecture
    n_layers = trial.suggest_int('n_layers', 2, 4)  # Number of layers
    net_arch = [trial.suggest_int(f'layer_{i}_units', 64, 256) for i in range(n_layers)]

    policy_kwargs = dict(
        net_arch=net_arch
    )

    # Create the env and model
    env = get_env(None, "console")
    
    model = get_agent("SACfD", env, DEMO_PATH, show_demos_in_env=False, hyperparams={"policy_kwargs": policy_kwargs, "lr": learning_rate},
                      gamma=gamma, tau=tau, target_update_interval=target_update_interval)

    # Train the model
    # TODO: Change this to be 50k timesteps
    model.learn(total_timesteps=1_000, progress_bar=True, log_interval=100_000)

    # Evaluate the model
    rewards, lengths = evaluate_policy(model, env, n_eval_episodes=5, return_episode_rewards=True)
    avg_reward = np.mean(rewards)
    avg_length = np.mean(lengths)

    performance = avg_reward - avg_length
    print(f"Over {len(rewards)} episodes: Avg Reward: {avg_reward}, Avg Length: {avg_length}, Perf={performance}")

    # TODO: Log this to a file :)
    return performance

# Create the Optuna study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

# Print the best hyperparameters
print('Best hyperparameters: ', study.best_params)