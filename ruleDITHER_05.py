import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from PIL import Image, ImageSequence

def load_and_convert_image(image_path, grid_size):
    """Load an image, resize it, and convert it to grayscale."""
    image = Image.open(image_path)
    image = image.resize(grid_size)  # Resize the image to the user-defined grid size
    image = np.array(image)
    gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    normalized_image = gray_image / np.max(gray_image)
    return normalized_image

def get_initial_state(image):
    """Threshold the image for initial CA state."""
    return np.where(image > 0.5, 1, 0)

def generate_rule_pattern(rule_number, state, original_image, dither_option):
    """Generates a cellular automaton pattern based on the given rule number and dithering option."""
    rule_bin = format(rule_number, '08b')
    rules = {(1, 1, 1): int(rule_bin[0]),
             (1, 1, 0): int(rule_bin[1]),
             (1, 0, 1): int(rule_bin[2]),
             (1, 0, 0): int(rule_bin[3]),
             (0, 1, 1): int(rule_bin[4]),
             (0, 1, 0): int(rule_bin[5]),
             (0, 0, 1): int(rule_bin[6]),
             (0, 0, 0): int(rule_bin[7])}
    
    next_state = np.zeros_like(state)
    n = state.shape[0]
    for i in range(n):
        for j in range(n):
            if dither_option == "whites" and original_image[i, j] <= 0.9:
                next_state[i, j] = state[i, j]
                continue
            elif dither_option == "blacks" and original_image[i, j] > 0.1:
                next_state[i, j] = state[i, j]
                continue
            elif dither_option == "grays" and (original_image[i, j] <= 0.1 or original_image[i, j] >= 0.9):
                next_state[i, j] = state[i, j]
                continue
            elif dither_option == "all":
                # No condition, apply the rule to all pixels
                pass
            else:
                continue  # If none of the above, continue to next pixel without changing

            top = state[(i-1) % n, j]
            left = state[i, (j-1) % n]
            right = state[i, (j+1) % n]
            neighbors = (top, left, right)
            next_state[i, j] = rules[neighbors]
    
    return next_state


def evolve_and_save_by_rule(initial_state, original_image, rule_number, dither_option, num_iterations=10, base_output_dir="output_frames"):
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, timestamp_str)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    state_frames = []
    state = initial_state
    for i in range(num_iterations):
        img = Image.fromarray((state * 255).astype(np.uint8))
        img = img.convert("1").convert("P")
        img_path = os.path.join(output_dir, f"frame_{i}.png")
        img.save(img_path)
        state_frames.append(img)
        state = generate_rule_pattern(rule_number, state, original_image, dither_option)

    # Function to create a GIF with reverse looping
    def create_gif_with_reverse_loop(frames, gif_name):
        if frames:
            reversed_frames = frames[::-1]  # Reverse the entire sequence
            looped_frames = frames + reversed_frames[1:]  # Combine forward and reverse, excluding the duplicated last frame
            gif_path = os.path.join(output_dir, gif_name)
            looped_frames[0].save(gif_path, save_all=True, append_images=looped_frames[1:], duration=100, loop=0)

    # Create GIFs
    create_gif_with_reverse_loop(state_frames, "automata_evolution.gif")
    create_gif_with_reverse_loop(state_frames[1::2], "odd_frames_evolution.gif")
    create_gif_with_reverse_loop(state_frames[0::2], "even_frames_evolution.gif")

    return output_dir






if __name__ == "__main__":
    filename = input("Enter the image filename/path: ")
    n = int(input("Enter the grid size (e.g., 256): "))
    grid_size = (n, n)  # Convert the single value to a tuple
    rule_number = int(input("Enter the rule number (0-255): "))
    dither_option = input("Choose which to dither (grays/blacks/whites/all): ").lower()
    num_generations = int(input("Enter the number of generations: "))

    image = load_and_convert_image(filename, grid_size)
    initial_state = get_initial_state(image)
    output_dir = evolve_and_save_by_rule(initial_state, image, rule_number, dither_option, num_generations)

    print(f"Frames saved in the '{output_dir}' directory.")
