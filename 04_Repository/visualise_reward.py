from Gym.bullet_drone_env import BulletDroneEnv
import matplotlib.pyplot as plt
import numpy as np

env = BulletDroneEnv(render_mode="console")

# Set the range for x and z
x_values = np.linspace(-3, 3, 100)
z_values = np.linspace(0, 6, 100)

# Create a grid of x, y=0, z values
x_grid, z_grid = np.meshgrid(x_values, z_values)

# Compute the rewards for each position
rewards = np.array([[env.calc_reward([x, 0, z]) for x, z in zip(x_row, z_row)] for x_row, z_row in zip(x_grid, z_grid)])

# Branch coordinates
branch_x = 0  # Example x-coordinate
branch_z = 2.7  # Example z-coordinate

# Plotting
plt.figure(figsize=(10, 6))
heatmap = plt.imshow(rewards, extent=[-3, 3, 0, 6], origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(heatmap, label='Reward')
plt.title('Reward Function Visualization')
plt.xlabel('X coordinate')
plt.ylabel('Z coordinate')

# Add the branch point
plt.scatter(branch_x, branch_z, color='red', label='Branch', s=20)  # 's' adjusts the size of the point
plt.legend()

plt.savefig("Visuals/rewards/position_branch_drone_collision.png")
plt.show()