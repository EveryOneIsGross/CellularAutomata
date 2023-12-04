import numpy as np
import matplotlib.pyplot as plt

def generate_rule_pattern(rule_number, n):
    """Generate a cellular automaton pattern based on the given rule number."""
    rule_bin = format(rule_number, '08b')
    rules = {
        (1, 1, 1): int(rule_bin[0]),
        (1, 1, 0): int(rule_bin[1]),
        (1, 0, 1): int(rule_bin[2]),
        (1, 0, 0): int(rule_bin[3]),
        (0, 1, 1): int(rule_bin[4]),
        (0, 1, 0): int(rule_bin[5]),
        (0, 0, 1): int(rule_bin[6]),
        (0, 0, 0): int(rule_bin[7])
    }

    grid = np.zeros((n, n), dtype=int)
    grid[0, n // 2] = 1

    for i in range(1, n):
        for j in range(n):
            left = grid[i-1, (j-1) % n]
            center = grid[i-1, j]
            right = grid[i-1, (j+1) % n]
            grid[i, j] = rules[(left, center, right)]

    return grid

def xor_patterns(base_pattern, modulator_pattern):
    """Apply XOR operation between the base and modulator patterns."""
    return np.bitwise_xor(base_pattern, modulator_pattern)

def main_3d():
    carrier = int(input("Enter the Carrier rule number (0-255): "))
    n = int(input("Enter the size n (for n x n grid): "))
    carrier_pattern = generate_rule_pattern(carrier, n)  # Move this line outside the loop

    modulators = []
    while True:
        modulator = input("Enter a Modulator rule number (0-255) or press Enter to continue: ")
        if modulator == "":
            break
        modulators.append(int(modulator))

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    total_layers = 1 + 4 * len(modulators)
    z_values = np.linspace(0, n*total_layers, n*total_layers, endpoint=False)

    # Carrier pattern in green
    x, y = np.where(carrier_pattern == 1)
    z = np.full_like(x, z_values[0:n].mean())
    ax.scatter(x, y, z, color='green', s=5)

    for idx, mod_num in enumerate(modulators):
        mod_pattern = generate_rule_pattern(mod_num, n)
        xor_result = xor_patterns(carrier_pattern, mod_pattern)
        reacted_nodes = xor_result != carrier_pattern

        # Modulator patterns in yellow
        x, y = np.where(mod_pattern == 1)
        z = np.full_like(x, z_values[n*(4*idx+1):n*(4*idx+2)].mean())
        ax.scatter(x, y, z, color='yellow', s=5)

        # Reacted nodes in red
        x, y = np.where(reacted_nodes == 1)
        z = np.full_like(x, z_values[n*(4*idx+2):n*(4*idx+3)].mean() - 0.5)
        ax.scatter(x, y, z, color='red', s=5, marker='x')

        # XOR results (activation) in blue
        x, y = np.where(xor_result == 1)
        z = np.full_like(x, z_values[n*(4*idx+3):n*(4*idx+4)].mean())
        ax.scatter(x, y, z, color='blue', s=5)

        # Update the carrier pattern for the next iteration
        carrier_pattern = xor_result

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Generation')
    ax.view_init(elev=30, azim=-60)
    plt.savefig(f"rule{carrier}_modulators{'_'.join(map(str, modulators))}_3d.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main_3d()
