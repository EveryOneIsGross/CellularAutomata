import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.signal as signal

# Define the parameters
N = 100 # Size of the grid
p = 0.2 # Probability of a cell being alive at the start
n = 64 # Number of generations

# Initialize the grid randomly
grid = np.random.choice([0, 1], size=(N, N), p=[1-p, p])

# Define the neighborhood
neighbors = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1] if (i, j) != (0, 0)]

# Define the rule
def update(grid):
    new_grid = grid.copy()
    for i in range(N):
        for j in range(N):
            # Count the number of alive neighbors
            alive = 0
            for di, dj in neighbors:
                # Use periodic boundary conditions
                ni = (i + di) % N
                nj = (j + dj) % N
                alive += grid[ni, nj]
            # Apply the rule
            if grid[i, j] == 1:
                if alive < 2 or alive > 3:
                    new_grid[i, j] = 0 # Die
            else:
                if alive == 3:
                    new_grid[i, j] = 1 # Become alive
    return new_grid

# Define the entropy function
def entropy(grid):
    # Flatten the grid into a 1D array
    grid = grid.flatten()
    # Count the frequency of each state
    freq = np.bincount(np.round(np.real(grid)).astype(int)) / len(grid)

    # Compute the entropy using the formula -sum(p * log(p))
    ent = -np.sum(freq * np.log2(freq + 1e-9))

    return ent

# Define the complexity function
def complexity(grid):
    # Compute the power spectrum of the grid using FFT
    ps = np.abs(np.fft.fft2(grid))**2
    # Compute the entropy of the power spectrum
    ent = entropy(ps)
    # Compute the complexity using the formula ent * log(N)
    comp = ent * np.log2(N)
    return comp

# Define the fractal dimension function
def fractal_dimension(grid):
    # Define the box sizes to use
    sizes = [2, 4, 5, 10, 20, 25, 50]

    # Initialize the list of box counts
    counts = []
    # Loop over the box sizes
    for size in sizes:
        # Divide the grid into boxes of the given size
        boxes = np.reshape(grid, (N // size, size, N // size, size))
        # Count the number of boxes that have at least one alive cell
        count = np.count_nonzero(np.sum(boxes, axis=(1, 3)) > 0)
        # Append the count to the list
        counts.append(count)
    # Convert the lists to numpy arrays
    sizes = np.array(sizes)
    counts = np.array(counts)
    # Compute the log-log slope of the box counts vs the box sizes
    slope, _, _, _, _ = stats.linregress(np.log2(sizes), np.log2(counts))
    # Compute the fractal dimension using the formula -slope
    fd = -slope
    return fd

# Initialize the lists of entropy, complexity, and fractal dimension
entropies = []
complexities = []
fractal_dimensions = []

# Loop over the generations
for i in range(n):
    # Update the grid
    grid = update(grid)
    # Compute the entropy, complexity, and fractal dimension
    ent = entropy(grid)
    comp = complexity(grid)
    fd = fractal_dimension(grid)
    # Append the values to the lists
    entropies.append(ent)
    complexities.append(comp)
    fractal_dimensions.append(fd)

# Plot the results
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.plot(entropies, 'b')
plt.xlabel('Generation')
plt.ylabel('Entropy')
plt.title('Entropy over time')
plt.subplot(1,3,2)
plt.plot(complexities, 'r')
plt.xlabel('Generation')
plt.ylabel('Complexity')
plt.title('Complexity over time')
plt.subplot(1,3,3)
plt.plot(fractal_dimensions, 'g')
plt.xlabel('Generation')
plt.ylabel('Fractal dimension')
plt.title('Fractal dimension over time')
plt.show()
