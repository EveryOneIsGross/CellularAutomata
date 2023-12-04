import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import tempfile
import string
ALLOWED_CHARS = string.ascii_letters + string.digits + string.punctuation + string.whitespace


def generate_rule_pattern(rule_number, n, initial_condition):
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


def modulate_with_multiple_modulators(carrier, modulators, n, initial_condition=""):
    """Modulate the carrier rule using multiple modulators."""
    pattern = generate_rule_pattern(carrier, n, initial_condition)  # Start with the carrier pattern using the initial_condition
    
    for modulator in modulators:
        modulator_pattern = generate_rule_pattern(modulator, n, initial_condition)
        
        # Apply modulation using XOR operation
        pattern = np.bitwise_xor(pattern, modulator_pattern)  # Update the pattern with the result of the modulation
    
    return pattern


def generate_automaton_image(carrier, n, modulators=""):
    # Convert the parameters from string to int
    carrier = int(carrier)
    n = int(n)
    
    # Convert modulators to a list of ints, filtering out empty strings
    modulators = [int(m) for m in modulators.split(",") if m]
    
    # Determine mode based on whether modulators are provided
    mode = "yes" if modulators else "no"
    
    # Generate the automaton pattern based on user choice
    if mode.lower() == "yes":
        pattern = modulate_with_multiple_modulators(carrier, modulators, n)
    else:
        pattern = generate_rule_pattern(carrier, n)

    # Plot
    plt.figure(figsize=(10, 10))
    plt.imshow(pattern, cmap='binary', origin='upper')
    plt.axis('off')
    modulator_str = '_'.join(map(str, modulators))
    #plt.title(f"Automaton for Rule {carrier} {'with' if mode.lower() == 'yes' else 'without'} Modulation")
    
    # Save to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_file.name, dpi=300)
    plt.savefig(f"rule{carrier}_modulators_{modulator_str}_mono.png", dpi=300)
    plt.close()  # Close the figure to release the memory
    
    return temp_file.name



def modulate_with_multiple_modulators_shift(carrier, modulators, n, initial_condition=""):
    carrier_pattern = generate_rule_pattern(carrier, n, initial_condition)
    for modulator in modulators:
        modulator_pattern = generate_rule_pattern(modulator, n, initial_condition)  # Add initial_condition here
        
        # Apply modulation using the central value of the modulator as the shift
        for i in range(1, n):
            shift_amount = int(modulator_pattern[i, n//2])
            carrier_pattern[i] = np.roll(carrier_pattern[i], shift=shift_amount)
    
    return carrier_pattern

def modulate_with_frequency_modulation(carrier, modulators, n, initial_condition=""):
    carrier_pattern = generate_rule_pattern(carrier, n, initial_condition)
    for modulator in modulators:
        modulator_pattern = generate_rule_pattern(modulator, n, initial_condition)  # Add initial_condition here
        
        # Apply Fourier Transform
        carrier_fft = np.fft.fft2(carrier_pattern)
        modulator_fft = np.fft.fft2(modulator_pattern)
        
        # Modulate the frequency components
        modulated_freq = np.multiply(carrier_fft, modulator_fft)
        
        # Apply inverse Fourier Transform
        modulated_pattern = np.fft.ifft2(modulated_freq)
        
        # Handle NaN and Infinite Values
        modulated_pattern = np.where(np.isfinite(modulated_pattern), modulated_pattern, 0)
        modulated_pattern = np.nan_to_num(modulated_pattern)  # Ensure NaN values are replaced
        
        # Take the real part and round to get binary values
        carrier_pattern = np.round(np.real(modulated_pattern)).astype(int)
    
    return carrier_pattern



def modulate_with_multiplication_modulation(carrier, modulators, n, initial_condition=""):
    carrier_pattern = generate_rule_pattern(carrier, n, initial_condition)
    for modulator in modulators:
        modulator_pattern = generate_rule_pattern(modulator, n, initial_condition)  # Using the initial_condition
        
        # Multiply the states and take the result modulo 2
        carrier_pattern = np.multiply(carrier_pattern, modulator_pattern) % 2
    
    return carrier_pattern





def decode_top_row(pattern, n):
    """Decodes the encoded binary string from the top row of the pattern."""
    top_row = pattern[0, :]
    binary_str = ''.join(map(str, top_row))
    return binary_to_string(binary_str)

def calculate_entropy(pattern):
    unique, counts = np.unique(pattern, return_counts=True)
    probabilities = counts / len(pattern)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def calculate_lyapunov_exponent(original_pattern, modulated_pattern, N):
    differences = np.sum(original_pattern != modulated_pattern, axis=1)
    
    # Add a small value epsilon to prevent log(0)
    epsilon = 1e-10
    lyapunov_vals = np.log(differences + epsilon)
    
    return np.mean(lyapunov_vals)

def filter_input(sentence):
    return ''.join([char for char in sentence if char in ALLOWED_CHARS])

def string_to_binary(s):
    """Convert a string to its binary representation."""
    return ''.join(format(ALLOWED_CHARS.index(c), '07b') for c in s if c in ALLOWED_CHARS)

def binary_to_string(b):
    """Convert a binary representation back to a string."""
    result = []
    for i in range(0, len(b), 7):  # Adjusted for 7 bits representation
        binary_chunk = b[i:i+7]
        if all(char in '01' for char in binary_chunk) and len(binary_chunk) == 7:  # Check if valid binary
            idx = int(binary_chunk, 2)
            result.append(ALLOWED_CHARS[idx] if idx < len(ALLOWED_CHARS) else '?')  # Fetch character or use placeholder
        else:
            result.append('?')  # Placeholder for invalid binary chunks
    return ''.join(result)

# Gradio Interface function
def generate_automaton_image_with_encoding(carrier, n, modulation_type, sentence, modulators=""):
    # Convert the input sentence to a binary representation
    initial_condition = string_to_binary(sentence)
    
    # Adjust the grid size to accommodate the binary length of the sentence
    n = max(n, len(initial_condition))
    
    # Convert other parameters
    carrier = int(carrier)
    modulators = [int(m) for m in modulators.split(",") if m]
  
    # If no modulators are provided, default to no modulation
    if not modulators:
        pattern = generate_rule_pattern(carrier, n, initial_condition)
    else:
        if modulation_type == "XOR Modulation":
            pattern = modulate_with_multiple_modulators(carrier, modulators, n, initial_condition)
        elif modulation_type == "Shift Modulation":
            pattern = modulate_with_multiple_modulators_shift(carrier, modulators, n, initial_condition)
        elif modulation_type == "Frequency Modulation":
            pattern = modulate_with_frequency_modulation(carrier, modulators, n, initial_condition)
        elif modulation_type == "Multiplication Modulation":
            pattern = modulate_with_multiplication_modulation(carrier, modulators, n, initial_condition)


    # Decode the middle column back to a string
    #middle_column = pattern[:, n//2]
    #decoded_string = binary_to_string(''.join(map(str, middle_column)))

    # Decode the top row back to a string
    decoded_string = decode_top_row(pattern, n)

    
    # Prepare data for output
    modulator_str = '_'.join(map(str, modulators))

    # Calculate the entropy and Lyapunov Exponent
    entropy = calculate_entropy(pattern.ravel())
    original_pattern = generate_rule_pattern(carrier, n, initial_condition)
    lyapunov_exponent = calculate_lyapunov_exponent(original_pattern, pattern, n)
    
    with open("output_data.txt", 'a') as file:
        file.write(f"\n---------- NEW ENTRY ----------\n")
        file.write(f"Carrier Rule Number: {carrier}\n")
        file.write(f"Grid Size: {n}x{n}\n")
        file.write(f"Modulation Type: {modulation_type}\n")
        file.write(f"Modulator Rule Numbers: {modulator_str}\n")
        file.write(f"Input Sentence: {sentence}\n")
        file.write(f"Decoded String: {decoded_string}\n")
        file.write(f"Lyapunov Exponent: {lyapunov_exponent}\n")
        file.write(f"Entropy of Generated Pattern: {entropy}\n")

    plt.figure(figsize=(10, 10))
    plt.imshow(pattern, cmap='binary', origin='upper')
    plt.axis('off')
    plt.savefig(f"rule{carrier}_{modulation_type}_{modulator_str}_{n}_mono.png", dpi=300)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_file.name, dpi=300)
    plt.close()

    # Return both the image and the decoded string
    return temp_file.name, f"Decoded String: {decoded_string}"




css = """
:root {
  --background-fill-secondary: white !important;
  --shadow-drop-lg: white !important;
  --block-label-border-width: 0px; */
  --block-label-text-color: white */
  --block-label-margin: 0; */
  --block-label-padding: var(--spacing-sm) var(--spacing-lg); */
  --block-label-radius: calc(var(--radius-lg) - 1px) 0 calc(var(--radius-lg) - 0px) 0; */
  --block-label-right-radius: 0 calc(var(--radius-lg) - 0px) 0 calc(var(--radius-lg) - 0px); */
  --block-label-text-size: var(--text-md); */
  --block-label-text-weight: 0;
}
.hide-label .gradio-block-label {display: none;}
.hide-icon .gradio-image-icon {display: none;}

.gradio-input-section {
  --background-fill-secondary: white;
}

.gradio-content {
    background-color: white;
}

.gradio-input-section, .gradio-output-section {
    background-color: white;
    box-shadow: none;
}

.gradio-input, .gradio-output {
    border: none;
    box-shadow: none;
}

body, label, .gradio-textbox textarea {
    font-family: 'Comic Sans MS', 'Comic Sans';
    font-weight: bold;
}
"""
description_md = """
1. **Carrier Rule Number**: This is the main rule that defines the behavior of the cellular automaton. Enter a number between 0 and 255 to select a specific rule.
2. **Size n x n**: Define the grid size for visualization. The pattern starts from the top-middle cell and evolves downwards.
3. **Modulation Type**: Enhance or modify the carrier pattern using different modulations. Options include XOR, Shift, Frequency, and Multiplication Modulations.
4. **Modulator Rule Numbers**: If you're using modulation, enter the rule numbers (comma-separated) of the modulators. These modulators will interact with the carrier to produce unique patterns.
5. **Input Sentence**: Enter a sentence to encode it into the pattern. The sentence will be converted to binary and placed in the middle of the top row. The pattern will evolve downwards from there.
"""

iface = gr.Interface(
    fn=generate_automaton_image_with_encoding,
    theme=gr.themes.Monochrome(),
    css=css,
    inputs=[
        gr.Slider(0, 255, label="Carrier rule number"),
        gr.Slider(10, 5000, label="Size n (for n x n grid)", step=1),
        gr.Radio(["XOR Modulation", "Shift Modulation", "Frequency Modulation", "Multiplication Modulation"], 
                label="Modulation Type"),
        gr.Textbox(label="Input Sentence"),
        gr.Textbox(label="Modulator rule numbers (comma-separated, leave empty for no modulation)")
    ],
    outputs=[gr.Image(label=" "), gr.Textbox(label="Decoded Message")],
    title="Hyper-Cellular Automaton Visualizer",
    description=description_md
)


# Launch the interface
iface.launch(share=False)