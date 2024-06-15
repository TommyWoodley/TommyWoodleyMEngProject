import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def extract_positions(csv_file, flight_start, flight_end):
    # Load the CSV file
    data = pd.read_csv(csv_file)

    # Extract positions and convert from mm to m
    timesteps = data['Timestamp'][flight_start:flight_end]
    drone_x = data['drone_x'][flight_start:flight_end] / 1000.0
    drone_y = data['drone_y'][flight_start:flight_end] / 1000.0
    drone_z = data['drone_z'][flight_start:flight_end] / 1000.0
    payload_x = data['payload_x'][flight_start:flight_end] / 1000.0
    payload_y = data['payload_y'][flight_start:flight_end] / 1000.0
    payload_z = data['payload_z'][flight_start:flight_end] / 1000.0

    return timesteps, drone_x, drone_y, drone_z, payload_x, payload_y, payload_z

def plot_3d_positions(csv_file, flight_start, flight_end):
    timesteps, drone_x, drone_y, drone_z, payload_x, payload_y, payload_z = extract_positions(csv_file, flight_start, flight_end)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot drone positions
    ax.scatter(drone_x, drone_y, drone_z, c='b', marker='o', label='Drone')

    # Plot payload positions
    ax.scatter(payload_x, payload_y, payload_z, c='r', marker='^', label='Payload')

    # Set labels with units in meters
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Plot of Drone and Payload Positions (Timesteps {} to {})'.format(flight_start, flight_end))
    ax.legend()

    # Show plot
    plt.show()

def plot_single_axis(csv_file, flight_start, flight_end):
    timesteps, drone_x, drone_y, drone_z, payload_x, payload_y, payload_z = extract_positions(csv_file, flight_start, flight_end)

    # Create subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))

    # Plot X component
    axs[0, 0].plot(timesteps, drone_x, 'b', label='Drone X')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('X Position (m)')
    axs[0, 0].set_title('Drone X Position Over Time')
    axs[0, 0].legend()

    axs[0, 1].plot(timesteps, payload_x, 'r', label='Payload X')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('X Position (m)')
    axs[0, 1].set_title('Payload X Position Over Time')
    axs[0, 1].legend()

    # Plot Y component
    axs[1, 0].plot(timesteps, drone_y, 'b', label='Drone Y')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Y Position (m)')
    axs[1, 0].set_title('Drone Y Position Over Time')
    axs[1, 0].legend()

    axs[1, 1].plot(timesteps, payload_y, 'r', label='Payload Y')
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Y Position (m)')
    axs[1, 1].set_title('Payload Y Position Over Time')
    axs[1, 1].legend()

    # Plot Z component
    axs[2, 0].plot(timesteps, drone_z, 'b', label='Drone Z')
    axs[2, 0].set_xlabel('Time')
    axs[2, 0].set_ylabel('Z Position (m)')
    axs[2, 0].set_title('Drone Z Position Over Time')
    axs[2, 0].legend()

    axs[2, 1].plot(timesteps, payload_z, 'r', label='Payload Z')
    axs[2, 1].set_xlabel('Time')
    axs[2, 1].set_ylabel('Z Position (m)')
    axs[2, 1].set_title('Payload Z Position Over Time')
    axs[2, 1].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot 3D positions of drone and payload from tether flight data.')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to the tether flight data CSV file.')
    parser.add_argument('--start', '-s', type=int, required=True, help='Start timestep for the flight data.')
    parser.add_argument('--end', '-e', type=int, required=True, help='End timestep for the flight data.')
    parser.add_argument('--type', '-t', type=str, choices=['3d', 'single'], required=True, help='Type of plot to generate: "3d" for 3D plot, "single" for single axis plots.')

    args = parser.parse_args()

    if args.type == '3d':
        plot_3d_positions(args.input, args.start, args.end)
    elif args.type == 'single':
        plot_single_axis(args.input, args.start, args.end)
