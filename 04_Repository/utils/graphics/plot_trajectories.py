import matplotlib.pyplot as plt

def plot_trajectories(trajectories, output_filename=None, window_size=10,
                         title='Sample Trajectories', show_plot=True):
    for trajectory in trajectories:
        x_values = [state[0] for state in trajectory]
        y_values = [state[1] for state in trajectory]
        plt.plot(x_values, y_values, marker='o', linestyle='-', label=f'label', color='blue')

    # Add labels and title
    plt.xlabel('X position')
    plt.ylabel('Z position')
    plt.title(title)

    # Set axis limits
    plt.xlim(-3, 3)
    plt.ylim(0, 6)

    # Add a grid
    plt.grid(True)

    # Highlight the center point (0, 3)
    plt.scatter([0], [3], color='red', zorder=5)  # Zorder for making the point appear on top of the line
    plt.annotate('Branch', xy=(0, 3), xytext=(0, -15), textcoords='offset points', ha='center', color='red')
    if output_filename is not None:
        plt.savefig(output_filename)
    if show_plot:
        plt.show()