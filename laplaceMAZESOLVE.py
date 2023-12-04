import numpy as np
import matplotlib.pyplot as plt
import datetime

def generate_complex_maze(dim=16, complexity=2, density=2):
    shape = ((dim // 2) * 2 + 1, (dim // 2) * 2 + 1)
    complexity = int(complexity * (5 * (shape[0] + shape[1])))
    density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))
    Z = np.zeros(shape, dtype=bool)
    Z[0, :] = Z[-1, :] = Z[:, 0] = Z[:, -1] = 1  # Add the border
    for i in range(density):
        x, y = np.random.randint(0, shape[1] // 2) * 2, np.random.randint(0, shape[0] // 2) * 2
        Z[y, x] = 1
        for j in range(complexity):
            neighbours = []
            if x > 1:             neighbours.append((y, x - 2))
            if x < shape[1] - 2:  neighbours.append((y, x + 2))
            if y > 1:             neighbours.append((y - 2, x))
            if y < shape[0] - 2:  neighbours.append((y + 2, x))
            if len(neighbours):
                y_, x_ = neighbours[np.random.randint(0, len(neighbours))]
                if Z[y_, x_] == 0:
                    Z[y_, x_] = 1
                    Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                    x, y = x_, y_
    Z[0, 1] = 0  # Remove the block from the start and end points of the maze
    Z[-1, -2] = 0
    return Z

def solve_laplace_complex_maze(maze):
    # Initialize potential with zeros
    phi = np.zeros_like(maze, dtype=float)
    # Assign high potential to the starting point
    phi[0, 1] = 10
    # Assign low potential to the exit point
    phi[-1, -2] = 0
    # Identify open spaces and walls
    mask = maze == 0
    walls = maze == 1

    # Iterate to solve for the potential, respecting walls
    while True:
        phi_old = phi.copy()
        for y in range(1, phi.shape[0] - 1):
            for x in range(1, phi.shape[1] - 1):
                if mask[y, x]:  # Update potential only for open spaces
                    neighbors = [(y, x+1), (y, x-1), (y+1, x), (y-1, x)]
                    valid_neighbors = [phi_old[ny, nx] for ny, nx in neighbors if mask[ny, nx]]
                    phi[y, x] = np.mean(valid_neighbors) if valid_neighbors else phi_old[y, x]

        # Apply the boundary condition for walls
        phi[walls] = 1

        # Check for convergence
        if np.max(np.abs(phi[mask] - phi_old[mask])) < 1e-4:
            break

    return phi

def plot_complex_maze_solution(maze, phi):
    plt.figure(figsize=(10, 10))
    plt.imshow(maze, cmap='gray', interpolation='none')
    gy, gx = np.gradient(-phi)  # negative sign to get the gradient in the direction of decreasing potential
    plt.streamplot(np.arange(phi.shape[1]), np.arange(phi.shape[0]), gx, gy, color=phi, cmap='Grays', density=0.2, linewidth=2, arrowstyle='-|>', arrowsize=2)
    plt.axis('off')
    # save the plot with a unique timestamped filename
    plt.savefig('maze_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.png', dpi=300)
    plt.show()

maze_complex = generate_complex_maze(dim=4, complexity=2, density=2)
phi_complex = solve_laplace_complex_maze(maze_complex)
plot_complex_maze_solution(maze_complex, phi_complex)
