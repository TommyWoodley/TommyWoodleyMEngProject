import pandas as pd
import sys
import matplotlib.pyplot as plt
import os


def read_csv_file(filename):
    """
    Reads a CSV file and prints its contents.

    Args:
    filename (str): The path to the CSV file to be read.
    """
    try:
        # Load the CSV file into a DataFrame
        data = pd.read_csv(filename, comment='#', header=0)
        print("CSV File Contents:")
        print(data)

        directory = os.path.dirname(filename)
        output_filename = os.path.join(directory, 'rewards_graph.png')

        plot_reward_graph(data, output_filename)
    except FileNotFoundError:
        print("Error: File not found. Please check the filename and try again.")
    except pd.errors.EmptyDataError:
        print("Error: File is empty.")
    except pd.errors.ParserError:
        print("Error: File could not be parsed. Check if the file is a valid CSV.")
    except Exception as e:
        print(f"An error occurred: {e}")


def plot_reward_graph(data, output_filename, window_size=10):
    rewards = data['r']

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


if __name__ == "__main__":
    # Check if the filename is given as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <filename>")
    else:
        filename = sys.argv[1]
        read_csv_file(filename)
