import matplotlib.pyplot as plt
import json

json_files = ['0.0', '45.0', '90.0', '135.0', '180.0', '225.0', '270.0', '315.0', '360.0']

# Create a plot
plt.figure(figsize=(10, 6))

# Colors for each line
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Add more colors if you have more files

for i, angle in enumerate(json_files):
    filename = f"rl_demos/rl_demo_approaching_angle_{angle}.json"
    # Load data from each JSON file
    with open(filename, 'r') as file:
        data = json.load(file)

    # Extracting state coordinates for plotting
    x_values = []
    y_values = []

    for entry in data:
        x_values.append(entry["state"][0])
        y_values.append(entry["state"][1])
    
    x_values.append(data[-1]["next_state"][0])
    y_values.append(data[-1]["next_state"][1])

    # Plot each file's data with a different color
    plt.plot(x_values, y_values, marker='o', linestyle='-', label=f'Angle: {angle}')

# Set axis limits
plt.xlim(-3, 3)
plt.ylim(0, 6)

# Add a grid
plt.grid(True)

# Highlight the center point (0, 3)
plt.scatter([0], [3], color='red', zorder=5)  # Zorder for making the point appear on top of the line
plt.annotate('Center (0,3)', xy=(0, 3), xytext=(10, -10), textcoords='offset points', ha='center', color='red')

# Adding legend to distinguish lines
plt.legend()

# Title and labels
plt.title('Plot of States in Reinforcement Learning Scenario Across Multiple Files')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

# Show the plot
plt.savefig("rl_demos/demo_visualisation.png")
plt.show()