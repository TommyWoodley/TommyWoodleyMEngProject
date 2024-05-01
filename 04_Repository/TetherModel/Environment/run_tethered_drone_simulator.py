from tethered_drone_simulator import TetheredDroneSimulator
from typing import List
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Gym.Rewards.Approaching import CircularApproachingReward
import matplotlib.pyplot as plt


class TetheredDroneSimulatorRunner:
    def __init__(self, xs: List[float], zs: List[float]) -> None:
        self.prev_pos = np.array([xs[0], 0, zs[0] + 3], dtype=np.float32)
        self.simulator = TetheredDroneSimulator(self.prev_pos)
        self.xs = xs
        self.zs = zs
        self.iteration = 0
        self.circular_reward = CircularApproachingReward()
        self.rewards = []

    def run(self) -> None:
        plt.ion()  # Turn on the interactive mode in matplotlib
        fig, ax = plt.subplots()

        already_moved = False
        action_size = None
        action_mags = []
        while True:
            it = min(self.iteration, (len(self.xs) - 1))
            x = self.xs[it]
            z = self.zs[it] + 3
            drone_pos = np.array([x, 0, z], dtype=np.float32)
            self.iteration += 395

            action = drone_pos - self.prev_pos
            action_mags.append(np.linalg.norm(action))
            if self.iteration < len(self.xs) * 2:
                has_collided, dist_tether_branch, dist_drone_branch, full = self.simulator.step(action)
            elif not already_moved:
                has_collided, dist_tether_branch, dist_drone_branch, full = self.simulator.step(np.array([-0.2, 0, 0], dtype=np.float32))
                already_moved = True
                action_size = np.mean(action_mags)
            else:
                has_collided, dist_tether_branch, dist_drone_branch, full = self.simulator.step()
            self.prev_pos = drone_pos
            state = self.simulator.drone_pos
            reward, _, _ = self.circular_reward.reward_fun(state, has_collided, dist_tether_branch, dist_drone_branch, full)
            print("x: ", x, " z: ", z, "action_mag: ", action_size, "reward: ", reward)

            # Append reward to the list
            self.rewards.append(reward)

            # Plot or update the graph
            ax.clear()
            ax.plot(self.rewards)
            ax.set_title('Reward Over Time')
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Reward')
            # plt.pause(0.05)  # Pause a bit for the plot to update
