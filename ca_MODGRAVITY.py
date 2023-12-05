import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio.v2 as imageio

import string
import matplotlib

matplotlib.use('agg')

ALLOWED_CHARS = string.ascii_letters + string.digits + string.punctuation + string.whitespace

def generate_rule_pattern(rule_number, n, initial_condition=None):
    """Generates a cellular automaton pattern based on the given rule number and initial condition."""
    
    # Convert the rule number to its binary representation
    rule_bin = format(rule_number, '08b')
    
    # Define the rule using the binary representation
    rules = {(1, 1, 1): int(rule_bin[0]),
             (1, 1, 0): int(rule_bin[1]),
             (1, 0, 1): int(rule_bin[2]),
             (1, 0, 0): int(rule_bin[3]),
             (0, 1, 1): int(rule_bin[4]),
             (0, 1, 0): int(rule_bin[5]),
             (0, 0, 1): int(rule_bin[6]),
             (0, 0, 0): int(rule_bin[7])}
    
    # Initialize the grid with zeros
    grid = np.zeros((n, n), dtype=int)
    
    # If initial_condition is provided, use it. Else, set the middle cell of the top row to '1'
    if initial_condition:
        start_pos = n // 2 - len(initial_condition) // 2
        for idx, val in enumerate(initial_condition):
            grid[0, start_pos + idx] = int(val)
    else:
        grid[0, n // 2] = 1
    
    for i in range(1, n):
        for j in range(n):
            # Using modulo for wrapping boundary conditions
            left = grid[i-1, (j-1) % n]
            center = grid[i-1, j]
            right = grid[i-1, (j+1) % n]
            
            grid[i, j] = rules[(left, center, right)]
    
    return grid

def modulate_with_multiple_modulators(carrier, modulators, n, initial_condition=None):
    """Modulate the carrier rule using multiple modulators."""
    pattern = generate_rule_pattern(carrier, n, initial_condition)  # Start with the carrier pattern
    
    for modulator in modulators:
        modulator_pattern = generate_rule_pattern(modulator, n, initial_condition)
        
        # Apply modulation using XOR operation
        pattern = np.bitwise_xor(pattern, modulator_pattern)  # Update the pattern with the result of the modulation
    
    return pattern

def apply_gravity(pattern):
    """Apply gravity to the pattern so that the black blocks "fall" downwards."""
    for col in range(pattern.shape[1]):
        # Flatten the column and reverse it
        column = pattern[:, col][::-1]
        
        # Identify the positions of the black blocks
        black_blocks = np.where(column == 1)[0]
        
        # If no black blocks, skip
        if not black_blocks.size:
            continue
        
        # Starting from the bottom, place the black blocks
        for idx, block_pos in enumerate(black_blocks):
            column[idx] = 1
        
        # Set the rest to white
        column[len(black_blocks):] = 0
        
        # Set the modified column back
        pattern[:, col] = column[::-1]
        
    return pattern


def animate_gravity(pattern):
    """Animate the application of gravity over a specified number of steps."""
    frames = [pattern.copy()]
    
    changes_made = True
    while changes_made:
        changes_made = False  # Reset for each full pass over the pattern
        for col in range(pattern.shape[1]):
            column = pattern[:, col]
            black_blocks = np.where(column == 1)[0]
            
            if not black_blocks.size:
                continue
            
            for idx in range(len(black_blocks) - 1, -1, -1):  # Go from bottom-most black block to top
                block_pos = black_blocks[idx]
                if block_pos < pattern.shape[0] - 1 and pattern[block_pos + 1, col] == 0:  # If there's room below to fall
                    pattern[block_pos, col], pattern[block_pos + 1, col] = 0, 1  # Let the block fall by 1 cell
                    changes_made = True  # Mark that changes were made in this pass
                    frames.append(pattern.copy())  # Append the modified pattern to frames

    return frames


import io

def create_gif(frames, filename="automaton_gravity.gif"):
    """
    Create an animated GIF from the given frames using in-memory binary streams.
    """
    with imageio.get_writer(filename, mode='I', duration=0.2) as writer:  # Add the duration parameter here
        for frame in frames:
            buf = io.BytesIO()  # Create an in-memory binary stream
            plt.figure(figsize=(10, 10))
            plt.imshow(frame, cmap='binary', origin='upper')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(buf, format="png", dpi=300)
            plt.close()
            buf.seek(0)  # Move to the start of the buffer
            writer.append_data(imageio.imread(buf))




def main():
    # Get user input from the terminal
    carrier = int(input("Enter the Carrier Rule Number (between 0 and 255): "))
    n = int(input("Enter the grid size n (for n x n grid): "))
    modulators_input = input("Enter Modulator Rule Numbers (comma-separated, leave empty for no modulation): ")
    modulators = [int(m.strip()) for m in modulators_input.split(",") if m.strip()]

    # Construct the filename based on the generation conditions
    filename = f"automaton_carrier{carrier}_grid{n}x{n}"
    if modulators:
        modulators_str = "_".join(map(str, modulators))
        filename += f"_modulators{modulators_str}"
    filename += ".gif"

    # Generate the automaton pattern
    if modulators:
        pattern = modulate_with_multiple_modulators(carrier, modulators, n)
    else:
        pattern = generate_rule_pattern(carrier, n)

    # Animate the gravity effect
    frames = animate_gravity(pattern)

    # Create the GIF using the constructed filename
    create_gif(frames, filename)
    print(f"Generated GIF named '{filename}'.")

if __name__ == "__main__":
    main()
