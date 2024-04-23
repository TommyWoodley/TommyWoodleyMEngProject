import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.graphics.plot_rl_rewards_training import plot_rl_reward_graph


def read_csv_file(filename, num_episodes=None, show=True):
    """
    Reads a CSV file and prints its contents.

    Args:
    filename (str): The path to the CSV file to be read.
    """
    try:
        # Load the CSV file into a DataFrame
        data = pd.read_csv(filename, comment='#', header=0)

        directory = os.path.dirname(filename)
        output_filename = os.path.join(directory, 'rewards_graph.png')

        rewards = data['r']
        lengths = data['l']
        if num_episodes is not None:
            rewards = rewards.iloc[0:num_episodes]
            lengths = lengths.iloc[0:num_episodes]
        plot_rl_reward_graph(rewards, episode_lens=lengths, output_filename=output_filename, show_plot=show)
    except FileNotFoundError:
        print("Error: File not found. Please check the filename and try again.")
    except pd.errors.EmptyDataError:
        print("Error: File is empty.")
    except pd.errors.ParserError:
        print("Error: File could not be parsed. Check if the file is a valid CSV.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Check if the filename is given as a command-line argument
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        read_csv_file(filename, show=True)
    elif len(sys.argv) == 3:
        filename = sys.argv[1]
        num_episodes = int(sys.argv[2])
        read_csv_file(filename, num_episodes=num_episodes, show=True)
    else:
        print("Usage: python generate_reward_graph_from_logs.py <filename> <num>")
        
