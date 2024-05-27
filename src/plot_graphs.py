import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import glob

limits = {
    'x': (-3, 3),
    'y': (-3, 3),
    'z': (0, 6),
    'roll': (-1, 1),
    'pitch': (-1, 1),
    'yaw': (-1, 1)}


# Function to read the CSV file and plot each column over time
def plot_columns_over_time(input_file, output_dir):
    # Read the CSV file
    data = pd.read_csv(input_file)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    first_phase_one_index = data[data['phase'] == 1].index[0] if 1 in data['phase'].values else None

    def plot_data(sub_data, suffix):
        # Plot each column over time
        for column, (min_limit, max_limit) in limits.items():
            plt.figure(figsize=(10, 6))
            plt.plot(sub_data['timestep'], sub_data[column], label=f'{column} over time', color='blue')

            actual_min = min(sub_data[column].min(), min_limit)
            actual_max = max(sub_data[column].max(), max_limit)
            plt.ylim(actual_min, actual_max)

            # Add labels and title
            plt.xlabel('Time')
            plt.ylabel(column)
            plt.title(f'{column} over Time')
            plt.grid(True)

            # Save the plot
            base_filename = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(output_dir, f'{base_filename}_{column}_over_time_{suffix}.png')
            plt.savefig(output_file)
            plt.close()

    plot_data(data, 'full')

    if first_phase_one_index is not None:
        plot_data(data.iloc[:first_phase_one_index], "phase0")

        plot_data(data.iloc[first_phase_one_index:], "phase1")


# Main function to handle command-line arguments
def main():
    parser = argparse.ArgumentParser(description='Plot CSV columns over time.')
    parser.add_argument('-i', '--input', help='Path to the input CSV file')
    parser.add_argument('-d', '--directory', help='Path to the directory containing CSV files')
    parser.add_argument('-o', '--output', required=True, help='Directory to save the output plots')

    args = parser.parse_args()

    if args.input:
        plot_columns_over_time(args.input, args.output)
    elif args.directory:
        csv_files = glob.glob(os.path.join(args.directory, '*.csv'))
        for csv_file in csv_files:
            plot_columns_over_time(csv_file, args.output)
    else:
        print("Please provide either an input file with -i or a directory with -d.")


if __name__ == '__main__':
    main()
