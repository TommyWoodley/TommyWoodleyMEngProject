import optuna
import gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from main import get_env

env = get_env(None, "console")

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

    # Create the model
    model = SAC('MlpPolicy', env, learning_rate=learning_rate, gamma=gamma, tau=tau,
                target_update_interval=target_update_interval, policy_kwargs=policy_kwargs, verbose=0)

    # Train the model
    eval_callback = EvalCallback(env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=1000, n_eval_episodes=5)
    model.learn(total_timesteps=10000, callback=eval_callback)

    # Evaluate the model
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)

    return mean_reward

# Create the Optuna study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Print the best hyperparameters
print('Best hyperparameters: ', study.best_params)