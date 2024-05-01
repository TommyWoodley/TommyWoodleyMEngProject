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
import time

def get_buffer_data(env):
    return load_all_data(env)


def load_all_data(env):
    json_data = load_json("Data/PreviousWorkTrajectories/rl_demos/rl_demo_approaching_angle_180.0.json")
    transformed_data = convert_data(env, json_data)
    return transformed_data


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

        action = np.array(item['action']) * 2
        reward = np.array(env.unwrapped.calc_reward([x, 0, z]))
        done = np.array([False])
        info = [{}]
        dataset.append((obs, next_obs, action, reward, done, info))
    return dataset

def run_simulation(env, actions, start):
    # Create the environment

    # Reset the environment to get the initial state
    state = env.reset(position=start)
    done = False

    # Run through each action in the provided list
    for action in actions:
        # Render the environment (optional)
        env.render()

        # Take the action, get the new state, reward, done flag, and additional info
        state, reward, done, _, _ = env.step(action)

        # Check if the episode is done
        if done:
            print("Episode finished")
            break
    
    for i in range(10):
        state, reward, done, _, _ = env.step(action=np.array([0.0, 0.0]))
    
    print(state)

    # Close the environment
    env.close()

def convert_to_actions(buffer_data):
    actions = []
    for data in buffer_data:
        _, _, action, _, _, _ = data
        print(action)
        actions.append(action)
    
    start, _, _, _, _, _ = buffer_data[0]
    x, z = start
    return actions, np.array([x, 0.0, z])

env = PositionWrapper(TwoDimWrapper(BulletDroneEnv(render_mode="human")))
actions, start = convert_to_actions(get_buffer_data(env))
time.sleep(2.0)

run_simulation(env, actions, start)