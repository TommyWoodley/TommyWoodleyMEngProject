import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from TetherModel.Environment.tethered_drone_simulator import TetheredDroneSimulator
import numpy as np
import seaborn as sns

def calculate_error_metrics(real, simulated):
    # Calculate differences
    differences = real - simulated

    # Calculate MBE
    mbe = np.mean(differences)
    
    # Calculate MAE
    mae = np.mean(np.abs(differences))
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean(differences ** 2))
    
    # Calculate SD
    sd = np.std(differences)
    
    return mbe, mae, rmse, sd

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

def extract_sim_positions(csv_file):
    # Load the CSV file
    data = pd.read_csv(csv_file)

    # Extract positions and convert from mm to m
    drone_x = data['drone_x'][:6000]
    drone_y = data['drone_y'][:6000]
    drone_z = data['drone_z'][:6000]
    payload_x = data['payload_x'][:6000]
    payload_y = data['payload_y'][:6000]
    payload_z = data['payload_z'][:6000]
    sim_payload_x = data['sim_payload_x'][:6000]
    sim_payload_y = data['sim_payload_y'][:6000]
    sim_payload_z = data['sim_payload_z'][:6000]

    return drone_x, drone_y, drone_z, payload_x, payload_y, payload_z, sim_payload_x, sim_payload_y, sim_payload_z

def plot_3d_positions(csv_file, flight_start, flight_end):
    timesteps, drone_x, drone_y, drone_z, payload_x, payload_y, payload_z = extract_positions(csv_file, flight_start, flight_end)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot drone positions
    ax.plot(drone_x, drone_y, drone_z, c='b', label='Drone')

    # Plot payload positions
    ax.plot(payload_x, payload_y, payload_z, c='r', label='Payload')

    # Set labels with units in meters
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Plot of Drone and Payload Positions \nDuring Random Flight'.format(flight_start, flight_end))
    ax.legend()

    # Show plot
    plt.savefig("./rosbag2_2024_06_14-12_06_23_tether_flight_data/RandomFlightPlot.png", dpi=500)
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

def plot_dual_axis(csv_file, flight_start, flight_end):
    timesteps, drone_x, drone_y, drone_z, payload_x, payload_y, payload_z = extract_positions(csv_file, flight_start, flight_end)

    # Create subplots for dual axis
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True)

    # Plot X component
    axs[0].plot(timesteps, drone_x, 'b', label='Drone X')
    axs[0].plot(timesteps, payload_x, 'r', label='Payload X')
    axs[0].set_ylabel('X Position (m)')
    axs[0].set_title('Drone and Payload X Positions Over Time')
    axs[0].legend()

    # Plot Y component
    axs[1].plot(timesteps, drone_y, 'b', label='Drone Y')
    axs[1].plot(timesteps, payload_y, 'r', label='Payload Y')
    axs[1].set_ylabel('Y Position (m)')
    axs[1].set_title('Drone and Payload Y Positions Over Time')
    axs[1].legend()

    # Plot Z component
    axs[2].plot(timesteps, drone_z, 'b', label='Drone Z')
    axs[2].plot(timesteps, payload_z, 'r', label='Payload Z')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Z Position (m)')
    axs[2].set_title('Drone and Payload Z Positions Over Time')
    axs[2].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()

def smooth_list(data, window_size=10):
    return pd.Series(data).rolling(window=window_size, min_periods=1).mean().tolist()

def plot_one_axis(csv_file, flight_start, flight_end):
    timesteps, drone_x, drone_y, drone_z, payload_x, payload_y, payload_z = extract_positions(csv_file, flight_start, flight_end)

    # Create a single plot
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(10, 6))

    # Plot X component
    avg_drone_x = sum(drone_x) / len(drone_x)
    adjusted_drone_x = [x - avg_drone_x for x in drone_x]
    plt.plot(timesteps, smooth_list(adjusted_drone_x), 'b', label='x')

    # Plot Y component
    avg_drone_y = sum(drone_y) / len(drone_y)
    adjusted_drone_y = [y - avg_drone_y for y in drone_y]
    plt.plot(timesteps, smooth_list(adjusted_drone_y), 'g', label='y')

    # Plot Z component
    avg_drone_z = sum(drone_z) / len(drone_z)
    adjusted_drone_z = [z - avg_drone_z for z in drone_z]
    plt.plot(timesteps, smooth_list(adjusted_drone_z), 'r', label='z')

    # Labels and title
    plt.xlabel('Time')
    plt.ylabel('Offset from fixed target position (m)')
    plt.title('Drone Positions Over Time When Hovering at a Fixed Point', fontsize=15)
    plt.legend()

    # Show plot
    plt.tight_layout()
    plt.savefig("hoverin_position_plot.png", dpi=500)
    plt.show()

def print_all_metrics(payload_x, payload_y, payload_z, sim_payload_x, sim_payload_y, sim_payload_z):
    # Convert to numpy arrays for calculation
    payload_x = np.array(payload_x)
    payload_y = np.array(payload_y)
    payload_z = np.array(payload_z)
    sim_payload_x = np.array(sim_payload_x)
    sim_payload_y = np.array(sim_payload_y)
    sim_payload_z = np.array(sim_payload_z)

    # Calculate error metrics for each dimension
    mbe_x, mae_x, rmse_x, sd_x = calculate_error_metrics(payload_x, sim_payload_x)
    mbe_y, mae_y, rmse_y, sd_y = calculate_error_metrics(payload_y, sim_payload_y)
    mbe_z, mae_z, rmse_z, sd_z = calculate_error_metrics(payload_z, sim_payload_z)

    total_real = np.sqrt(payload_x**2 + payload_y**2 + payload_z**2)
    total_simulated = np.sqrt(sim_payload_x**2 + sim_payload_y**2 + sim_payload_z**2)
    mbe_totoal, mae_total, rmse_total, sd_total = calculate_error_metrics(total_real, total_simulated)

    results = pd.DataFrame({
        'Metric': ['MBE', 'MAE', 'RMSE', 'SD'],
        'X': [mbe_x, mae_x, rmse_x, sd_x],
        'Y': [mbe_y, mae_y, rmse_y, sd_y],
        'Z': [mbe_z, mae_z, rmse_z, sd_z],
        'Total': [mbe_totoal, mae_total, rmse_total, sd_total]
    })

    # Display the results
    print(results)

    # Optionally, you can save the results to a CSV file
    results.to_csv('error_metrics.csv', index=False)

def plot_dual_sim_axis(csv_file):
    drone_x, drone_y, drone_z, payload_x, payload_y, payload_z, sim_payload_x, sim_payload_y, sim_payload_z = extract_sim_positions(csv_file)

    # Create subplots for dual axis
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    window_size=500
    # Plot X component
    # axs[0].plot(drone_x, 'b', label='Drone X')
    axs[0].plot(payload_x.rolling(window=window_size, min_periods=1).mean(), 'r', label='Actual Payload X',linewidth=2)
    axs[0].plot(sim_payload_x.rolling(window=window_size, min_periods=1).mean(), label='Simulated Payload X',linewidth=2)
    axs[0].set_ylabel('X Position (m)')
    # axs[0].set_ylim(-1.0, 0.3)
    axs[0].set_title('Drone and Payload X Positions Over Time')
    axs[0].legend(loc='best')

    # Plot Y component
    # axs[1].plot(drone_y, 'b', label='Drone Y')
    axs[1].plot(payload_y.rolling(window=window_size, min_periods=1).mean(), 'r', label='Payload Y',linewidth=2)
    axs[1].plot(sim_payload_y.rolling(window=window_size, min_periods=1).mean(), label='Simulated Payload Y',linewidth=2)
    axs[1].set_ylabel('Y Position (m)')
    # axs[1].set_ylim(-1.0, 0.3)
    axs[1].set_title('Drone and Payload Y Positions Over Time')
    axs[1].legend(loc='lower left')

    # Plot Z component
    # axs[2].plot(drone_z, 'b', label='Drone Z')
    axs[2].plot(payload_z.rolling(window=window_size, min_periods=1).mean(), 'r', label='Payload Z',linewidth=2)
    axs[2].plot(sim_payload_z.rolling(window=window_size, min_periods=1).mean(), label='Simulated Payload Z',linewidth=2)
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Z Position (m)')
    axs[2].set_title('Drone and Payload Z Positions Over Time')
    # axs[2].set_ylim(0.8, 2.0)
    axs[2].legend(loc='lower right')

    # Adjust layout
    plt.tight_layout()
    plt.savefig("position_plots.png", dpi=500)
    plt.show()

def plot_dual_sim_axis_individual_save(csv_file):
    _, _, _, payload_x, payload_y, payload_z, sim_payload_x, sim_payload_y, sim_payload_z = extract_sim_positions(csv_file)
    print_values = print_all_metrics(payload_x, payload_y, payload_z, sim_payload_x, sim_payload_y, sim_payload_z)
    window_size = 500

    # Plot X component
    fig_x, ax_x = plt.subplots(figsize=(5, 4))
    ax_x.plot(payload_x.rolling(window=window_size, min_periods=1).mean(), 'r', label='Actual Payload X', linewidth=2)
    ax_x.plot(sim_payload_x.rolling(window=window_size, min_periods=1).mean(), label='Simulated Payload X', linewidth=2)
    ax_x.set_ylabel('X Position (m)')
    ax_x.set_title('Drone and Payload X Positions Over Time')
    ax_x.legend(loc='best')
    plt.tight_layout()
    fig_x.savefig("position_plot_x.png", dpi=500)

    # Plot Y component
    fig_y, ax_y = plt.subplots(figsize=(5, 4))
    ax_y.plot(payload_y.rolling(window=window_size, min_periods=1).mean(), 'r', label='Payload Y', linewidth=2)
    ax_y.plot(sim_payload_y.rolling(window=window_size, min_periods=1).mean(), label='Simulated Payload Y', linewidth=2)
    ax_y.set_ylabel('Y Position (m)')
    ax_y.set_title('Drone and Payload Y Positions Over Time')
    ax_y.legend(loc='lower left')
    plt.tight_layout()
    fig_y.savefig("position_plot_y.png", dpi=500)

    # Plot Z component
    fig_z, ax_z = plt.subplots(figsize=(5, 4))
    ax_z.plot(payload_z.rolling(window=window_size, min_periods=1).mean(), 'r', label='Payload Z', linewidth=2)
    ax_z.plot(sim_payload_z.rolling(window=window_size, min_periods=1).mean(), label='Simulated Payload Z', linewidth=2)
    ax_z.set_xlabel('Time')
    ax_z.set_ylabel('Z Position (m)')
    ax_z.set_title('Drone and Payload Z Positions Over Time')
    ax_z.legend(loc='lower right')
    plt.tight_layout()
    fig_z.savefig("position_plot_z.png", dpi=500)

    plt.show()


def generate_from_sim(input_file, start_time, end_time):
    # Function to generate data from simulation based on input parameters
    timesteps, drone_x, drone_y, drone_z, payload_x, payload_y, payload_z = extract_positions(input_file, start_time, end_time)
    
      # Calculate actions (movements) based on position changes
    actions_x, actions_y, actions_z = [], [], []
    payload_x_list, payload_y_list, payload_z_list = [], [], []
    drone_x_list, drone_y_list, drone_z_list = [], [], []
    step = 1
    for i in range(1, len(drone_x), step):
        # Calculate movement from previous position
        payload_x_list.append(payload_x.iloc[i])
        payload_y_list.append(payload_y.iloc[i])
        payload_z_list.append(payload_z.iloc[i])
        drone_x_list.append(drone_x.iloc[i])
        drone_y_list.append(drone_y.iloc[i])
        drone_z_list.append(drone_z.iloc[i])
        
        movement_x = drone_x.iloc[i] - drone_x.iloc[i-step]
        movement_y = drone_y.iloc[i] - drone_y.iloc[i-step]
        movement_z = drone_z.iloc[i] - drone_z.iloc[i-step]
        
        # Append movements to action lists
        actions_x.append(movement_x)
        actions_y.append(movement_y)
        actions_z.append(movement_z)

    # Use the simulator to generate data 
    starting_pos = np.array([drone_x.iloc[0], drone_y.iloc[0], drone_z.iloc[0]])
    print(f"Using starting position: {starting_pos}")
    simulator = TetheredDroneSimulator(starting_pos, branch_enabled=False, gui_mode=False)

      # List to store tether payload positions
    payload_positions = []

    for action_x, action_y, action_z in zip(actions_x, actions_y, actions_z):
        action = np.array([action_x, action_y, action_z])
        simulator.step(action)  # Take a step with the calculated action

        # Extract tether payload position on each step
        payload_position = simulator.weight.get_position()
        payload_positions.append(payload_position)

    # Plot x, y, z comparisons (after simulation loop)
    plt.figure(figsize=(10, 6))

    # Extract x, y, z components from payload positions
    sim_x = [pos[0] for pos in payload_positions]
    sim_y = [pos[1] for pos in payload_positions]
    sim_z = [pos[2] for pos in payload_positions]
# drone_x,drone_y,drone_z,payload_x,payload_y,payload_z,round_bar_x

    all_positions_df = pd.DataFrame({
      'drone_x': drone_x_list,
      'drone_y': drone_y_list,
      'drone_z': drone_z_list,
      'payload_x': payload_x_list,
      'payload_y': payload_y_list,
      'payload_z': payload_z_list,
      'sim_payload_x': [pos[0] for pos in payload_positions],  # Extract X from sim positions
      'sim_payload_y': [pos[1] for pos in payload_positions],  # Extract Y from sim positions
      'sim_payload_z': [pos[2] for pos in payload_positions],  # Extract Z from sim positions
  })
    
    input_file_dir = os.path.dirname(input_file)
    filename = f"simulated_tether_positions_{start_time}_{end_time}.csv"
    output_file = os.path.join(input_file_dir, filename)

    all_positions_df.to_csv(output_file, index=False)  # Don't include index row

    print(f"Simulated positions exported to: {output_file}")


    plt.subplot(311)
    plt.plot(payload_x_list, label='Extracted X')
    plt.plot(sim_x, label='Simulated X')
    plt.xlabel('Step')
    plt.ylabel('X Position')
    plt.legend()

    plt.subplot(312)
    plt.plot(payload_y_list, label='Extracted Y')
    plt.plot(sim_y, label='Simulated Y')
    plt.xlabel('Step')
    plt.ylabel('Y Position')
    plt.legend()

    plt.subplot(313)
    plt.plot(payload_z_list, label='Extracted Z')
    plt.plot(sim_z, label='Simulated Z')
    plt.xlabel('Step')
    plt.ylabel('Z Position')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tool for plotting and generating data.')
    
    subparsers = parser.add_subparsers(dest='command', required=True, help='Choose command to execute.')

    # Subparser for plotting command
    plot_parser = subparsers.add_parser('plot', help='Plot positions of drone and payload.')
    plot_parser.add_argument('--input', '-i', type=str, required=True, help='Path to the tether flight data CSV file.')
    plot_parser.add_argument('--start', '-s', type=int, required=True, help='Start timestep for the flight data.')
    plot_parser.add_argument('--end', '-e', type=int, required=True, help='End timestep for the flight data.')
    plot_parser.add_argument('--type', '-t', type=str, choices=['3d', 'single', 'dual', 'one'], required=True, help='Type of plot to generate.')

    # Subparser for generating command
    generate_parser = subparsers.add_parser('generate', help='Generate data from simulation.')
    generate_parser.add_argument('--input', '-i', type=str, required=True, help='Input file for simulation data.')
    generate_parser.add_argument('--start', '-s', type=int, required=True, help='Start timestep for the generation.')
    generate_parser.add_argument('--end', '-e', type=int, required=True, help='End timestep for the generation.')

    # Subparser for sim plot command
    sim_plot_parser = subparsers.add_parser('sim_plot', help='Generate data from simulation.')
    sim_plot_parser.add_argument('--input', '-i', type=str, required=True, help='Input file for simulation data.')

    args = parser.parse_args()

    if args.command == 'plot':
        if args.type == '3d':
            plot_3d_positions(args.input, args.start, args.end)
        elif args.type == 'single':
            plot_single_axis(args.input, args.start, args.end)
        elif args.type == 'dual':
            plot_dual_axis(args.input, args.start, args.end)
        elif args.type == 'one':
            plot_one_axis(args.input, args.start, args.end)
    elif args.command == 'generate':
        generate_from_sim(args.input, args.start, args.end)
    elif sim_plot_parser:
        plot_dual_sim_axis_individual_save(args.input)
