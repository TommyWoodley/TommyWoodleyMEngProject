import matplotlib.pyplot as plt
import pandas as pd

def plot_rl_reward_graph(rewards, output_filename=None, window_size=10, title='Running Rewards and Variance Over Training',
                      show_plot=True):
    """
    Plot a graph of rewards with running average and variance.

    Parameters:
        rewards (pd.Series): A Pandas Series containing rewards.
        output_filename (str): The filename to save the plot. If None, plot will be displayed but not saved.
        window_size (int): The window size for calculating running average and variance.
        title (str): The title of the plot.
        show_plot (bool): Whether to display the plot or not.

    Returns:
        None
    """

    # Input validation
    if not isinstance(rewards, pd.Series):
        raise ValueError("rewards must be a Pandas Series.")
    if not isinstance(window_size, int):
        raise ValueError("window_size must be an integer.")
    if window_size < 1:
        raise ValueError("window_size must be greater than or equal to 1.")
    if not isinstance(title, str):
        raise ValueError("title must be a string.")
    if not isinstance(show_plot, bool):
        raise ValueError("show_plot must be a boolean.")
    
    _plot_reward_graph(rewards, output_filename=output_filename, window_size=window_size, title=title, show_plot=show_plot)
    
def _plot_reward_graph(rewards, output_filename, window_size, title, show_plot):

    running_avg = rewards.rolling(window=window_size, min_periods=1).mean()
    running_std = rewards.rolling(window=window_size, min_periods=1).std()

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(rewards.index, running_avg, color='red', linestyle='-', linewidth=2, label='Running Average')
    plt.fill_between(rewards.index, running_avg + running_std, running_avg - running_std,
                     color='gray', alpha=0.3, label='Variance')
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Reward Value')
    plt.grid(True)
    plt.legend()

    if output_filename:
        plt.savefig(output_filename)
    if show_plot:
        plt.show()