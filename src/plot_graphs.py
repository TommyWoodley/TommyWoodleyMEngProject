import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import glob

# Function to read the CSV file and plot each column over time
def plot_columns_over_time(input_file, output_dir):
    # Read the CSV file
    data = pd.read_csv(input_file)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # List of columns to plot
    columns = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
    
    # Plot each column over time
    for column in columns:
        plt.figure(figsize=(10, 6))
        plt.plot(data['timestep'], data[column], label=f'{column} over time', color='blue')
        
        # Add labels and title
        plt.xlabel('Time')
        plt.ylabel(column)
        plt.title(f'{column} over Time')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f'{base_filename}_{column}_over_time.png')
        plt.savefig(output_file)
        plt.close()

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
