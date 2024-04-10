import matplotlib.pyplot as plt
import pandas as pd

def plot_reward_graph(rewards, output_filename, window_size=10):

    running_avg = rewards.rolling(window=window_size, min_periods=1).mean()
    running_std = rewards.rolling(window=window_size, min_periods=1).std()

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(rewards.index, running_avg, color='red', linestyle='-', linewidth=2, label='Running Average')
    plt.fill_between(rewards.index, running_avg + running_std, running_avg - running_std,
                     color='gray', alpha=0.3, label='Variance')
    plt.title('Running Rewards and Variance Over Training')
    plt.xlabel('Episodes')
    plt.ylabel('Reward Value')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_filename)
    plt.show()