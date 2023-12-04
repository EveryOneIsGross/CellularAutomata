import numpy as np
import matplotlib.pyplot as plt

def generate_rule_pattern(rule_number, n):
    """Generates a cellular automaton pattern based on the given rule number."""
    rule_bin = format(rule_number, '08b')
    rules = {(1, 1, 1): int(rule_bin[0]),
             (1, 1, 0): int(rule_bin[1]),
             (1, 0, 1): int(rule_bin[2]),
             (1, 0, 0): int(rule_bin[3]),
             (0, 1, 1): int(rule_bin[4]),
             (0, 1, 0): int(rule_bin[5]),
             (0, 0, 1): int(rule_bin[6]),
             (0, 0, 0): int(rule_bin[7])}
    grid = np.zeros((n, n), dtype=int)
    grid[0, n // 2] = 1
    for i in range(1, n):
        for j in range(n):
            left = grid[i-1, (j-1) % n]
            center = grid[i-1, j]
            right = grid[i-1, (j+1) % n]
            grid[i, j] = rules[(left, center, right)]
    return grid

def combine_patterns_with_xor_and_colors(carrier, modulators, n):
    """Combine patterns of carrier and modulators using XOR operation and unique colors."""
    combined_pattern = generate_rule_pattern(carrier, n)
    for idx, modulator in enumerate(modulators, start=2):
        modulator_pattern = generate_rule_pattern(modulator, n)
        xor_result = np.bitwise_xor(combined_pattern, modulator_pattern)
        combined_pattern += idx * xor_result
    return combined_pattern

def main_xor_colored_overlay():
    carrier = int(input("Enter the Carrier rule number (0-255): "))
    n = int(input("Enter the size n (for n x n grid): "))
    modulators = []
    while True:
        modulator = input("Enter a Modulator rule number (0-255) or press Enter to continue: ")
        if modulator == "":
            break
        modulators.append(int(modulator))
    combined_pattern = combine_patterns_with_xor_and_colors(carrier, modulators, n)
    num_colors = 2 + len(modulators)
    cmap = plt.get_cmap('tab10', num_colors)
    plt.figure(figsize=(10, 10))
    plt.imshow(combined_pattern, cmap=cmap, origin='upper')
    plt.axis('off')
    plt.title(f"XOR Colored Overlay of Rule {carrier} with Modulators")
    modulator_str = '_'.join(map(str, modulators))
    plt.savefig(f"rule{carrier}_modulators_{modulator_str}_xor_colored_overlay.png", dpi=300)
    plt.show()

#main_xor_colored_overlay()

import tempfile

def generate_xor_colored_overlay(carrier, n, modulators):
    # Convert the parameters from string to int
    carrier = int(carrier)
    n = int(n)
    modulators = list(map(int, modulators.split(",")))

    # Generate the combined pattern using your existing code
    combined_pattern = combine_patterns_with_xor_and_colors(carrier, modulators, n)
    num_colors = 2 + len(modulators)
    cmap = plt.get_cmap('tab10', num_colors)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(combined_pattern, cmap=cmap, origin='upper')
    plt.axis('off')
    plt.title(f"XOR Colored Overlay of Rule {carrier} with Modulators")
    
    # Save to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_file.name, dpi=300)
    plt.close()  # Close the figure to release the memory
    
    return temp_file.name


import gradio as gr

# Create a gradio interface
iface = gr.Interface(
  fn=generate_xor_colored_overlay,
  inputs=[
    gr.Slider(0, 255, label="Carrier rule number"),
    gr.Slider(10, 5000, label="Size n (for n x n grid)", step=1),
    gr.Textbox(label="Modulator rule numbers (comma-separated)")
  ],
  outputs=gr.Image(),
  title="XOR Colored Overlay Generator",
  description="This is a demo of a gradio interface that generates cellular automaton patterns based on a given rule number and modulators, and then combines them with XOR operation and unique colors.",
)
# Launch the interface
iface.launch()

