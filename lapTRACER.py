'''
Modulates rules together to make complex states then laplace traces the path to generate a playable .wav from the two CA generations"
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import json
import os
import datetime
from typing import List
import numpy as np
from scipy.io.wavfile import write
from scipy.interpolate import interp1d


# Global variables
rule_1 = 115    
rule_2 = 61

matrix_width = 64
matrix_height = 8000 # 8000, 16000, 22050, 24000, 32000, 44100, 48000, 96000, 192000

output_folder = "output_files"
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

def create_cellular_automaton(rule, size=matrix_width, steps=matrix_height):
    """Create a 1D cellular automaton based on a given rule."""
    rule_bits = np.array([rule >> i & 1 for i in range(8)], dtype=np.uint8)
    ca = np.zeros((steps, size), dtype=np.uint8)
    ca[0, size // 2] = 1  # Initial condition: single cell in the middle

    for i in range(1, steps):
        for j in range(size):
            # Considering the neighborhood of each cell
            neighborhood = (ca[i - 1, (j - 1) % size] << 2) | (ca[i - 1, j] << 1) | ca[i - 1, (j + 1) % size]
            ca[i, j] = rule_bits[neighborhood]

    return ca

def laplace_filter(ca):
    """Apply Laplacian filter for edge detection."""
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return convolve2d(ca, kernel, mode='same', boundary='wrap')

def dither_image(image):
    """Apply dithering to the image using a simple threshold."""
    threshold = image.mean()
    return (image > threshold).astype(np.uint8)

def ca_to_continuous_path(ca):
    """Convert the cellular automaton array to a continuous path."""
    directions = []
    rows, cols = ca.shape
    start_row = 1
    start_column = np.argmax(ca[0]) + 1  # Starting at the first active cell in row 1
    end_row, end_column = None, None

    directions.append(f"Start at row {start_row}, column {start_column}")
    current_position = (0, start_column - 1)

    for i in range(rows):
        for j in range(cols):
            if ca[i, j] == 1:
                if current_position != (i, j):
                    # Move to a new row
                    if current_position[0] != i:
                        directions.append(f"Move to row {i + 1}")
                    # Move to a new column
                    if current_position[1] != j:
                        direction = "right" if j > current_position[1] else "left"
                        directions.append(f"Move {direction} to column {j + 1}")
                    current_position = (i, j)
                    end_row, end_column = i + 1, j + 1

    directions.append(f"End at row {end_row}, column {end_column}")
    return directions, rows, cols

def save_figure(figure, filename):
    """Save the figure with a unique timestamp."""
    figure_path = os.path.join(output_folder, f"{filename}_{timestamp}.png")
    print(f"Saving figure to {figure_path}")
    figure.savefig(figure_path)
    plt.close(figure)  # Close the plot to free memory

def save_json(data, filename):
    """Save data as JSON with a unique timestamp."""
    json_path = os.path.join(output_folder, f"{filename}_{timestamp}.json")
    with open(json_path, 'w') as file:
        json.dump(data, file, indent=4)
    return json_path

def visualize_path_from_json(json_file, filename):
    """Load path information from JSON file and visualize it."""
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    grid_size = data["grid_size"]
    instructions = data["path"]
    grid = np.zeros((grid_size["rows"], grid_size["columns"]))

    start_instruction = instructions[0].split()
    start_row = int(start_instruction[3].strip(','))
    start_column = int(start_instruction[5])
    current_row, current_column = start_row, start_column
    grid[current_row - 1][current_column - 1] = 1

    for instruction in instructions[1:]:
        if "End at row" in instruction:
            break
        parts = instruction.split()
        direction = parts[1]
        target = int(parts[-1])

        if direction == "right":
            for c in range(current_column, target + 1):
                grid[current_row - 1][c - 1] = 1
            current_column = target
        elif direction == "left":
            for c in range(target, current_column + 1):
                grid[current_row - 1][c - 1] = 1
            current_column = target
        elif direction == "to":
            row_move = target
            for r in range(min(current_row, row_move), max(current_row, row_move) + 1):
                grid[r - 1][current_column - 1] = 1
            current_row = row_move

    # Plot the grid as a maze
    plt.figure(figsize=(256, 256)) # if i want it bigger i can change the numbers here
    plt.imshow(grid, cmap='Greys', interpolation='none')
    #plt.title("Generated Maze Map")
    plt.axis('off')
    save_figure(plt.gcf(), filename)

def generate_waveform_from_array(ca_array, stretch_factor=10):
    # Use the sum or average of each row to represent its intensity
    waveform = np.mean(ca_array, axis=1)
    # Normalize the waveform to the range [0, 1]
    waveform_normalized = (waveform - np.min(waveform)) / (np.max(waveform) - np.min(waveform))
    # Normalize to peak amplitude
    peak_amplitude = np.max(np.abs(waveform_normalized))
    if peak_amplitude > 0:  # Avoid division by zero
        waveform_normalized /= peak_amplitude

    # Stretch the waveform
    original_length = len(waveform_normalized)
    stretched_length = int(original_length * stretch_factor)
    x_original = np.linspace(0, 1, original_length)
    x_stretched = np.linspace(0, 1, stretched_length)
    stretched_waveform = interp1d(x_original, waveform_normalized, kind='linear')(x_stretched)

    # Scale to 16-bit audio range
    audio_waveform = np.interp(stretched_waveform, (0, 1), (-32768, 32767))
    return audio_waveform.astype(np.int16)


def save_waveform_as_wav(waveform, sample_rate, file_path):
    write(file_path, sample_rate, waveform)

def save_grid_state_as_png(ca, step, filename):
    """Save a specific step of the cellular automaton as a PNG file."""
    grid_state = ca[step:step + matrix_width, :matrix_width]  # Extracting the square grid
    fig = plt.figure(figsize=(8, 8))  # Adjust figure size as needed
    plt.imshow(grid_state, cmap='binary', interpolation='nearest')
    plt.axis('off')
    figure_path = os.path.join(output_folder, f"{filename}_{timestamp}.png")
    fig.savefig(figure_path)
    plt.close(fig)
    print(f"Saved grid state to {figure_path}")

# Generate cellular automata
ca_01 = create_cellular_automaton(rule_1)
ca_02 = create_cellular_automaton(rule_2)
modulated_ca = np.bitwise_xor(ca_01, ca_02)

# Example of saving specific steps
save_grid_state_as_png(ca_01, 100, "ca_01_step_100")  # Saving 100th step of ca_01
save_grid_state_as_png(ca_02, 100, "ca_02_step_100")  # Saving 100th step of ca_02
save_grid_state_as_png(modulated_ca, 100, "modulated_ca_step_100")  # Saving 100th step of modulated_ca

laplace_ca = laplace_filter(modulated_ca)
dithered_ca = dither_image(laplace_ca)

# Convert the Laplacian filtered CA to a continuous path
continuous_path_directions, rows, cols = ca_to_continuous_path(laplace_ca)

# Save the first result
fig = plt.figure(figsize=(100, 100))
plt.imshow(dithered_ca, cmap='binary', interpolation='nearest')
plt.axis('off')
save_figure(fig, f"ca_result_rule{rule_1}_rule{rule_2}")

# Convert the Laplacian filtered CA to a continuous path and save as JSON
continuous_path_directions, rows, cols = ca_to_continuous_path(laplace_ca)
json_filename = f"ca_path_directions_rule{rule_1}_rule{rule_2}"
json_path = save_json({"grid_size": {"rows": rows, "columns": cols}, "path": continuous_path_directions}, json_filename)

# Visualize the path and save the figure
visualize_path_from_json(json_path, json_filename)

# Generate and save the first audio waveform
waveform_1 = generate_waveform_from_array(laplace_ca)
wav_file_path_1 = os.path.join(output_folder, f"ca_result_rule{rule_1}_rule{rule_2}_{timestamp}.wav")
save_waveform_as_wav(waveform_1, 44100, wav_file_path_1)
print(f"Saved waveform to {wav_file_path_1}")
