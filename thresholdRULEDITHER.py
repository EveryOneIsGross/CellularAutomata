import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageFilter
import datetime
from PIL import ImageSequence

def load_and_convert_image(image_path, desired_size):
    """Load an image, resize it, and convert it to grayscale."""
    image = Image.open(image_path)
    image = image.resize(desired_size)  # Resize the image to the user-defined grid size
    image = np.array(image)
    gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    normalized_image = gray_image / np.max(gray_image)
    return normalized_image

def compute_otsu_threshold(image):
    """Determine the best threshold using Otsu's method."""
    histogram, bin_edges = np.histogram(image.ravel(), bins=256, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total = histogram.sum()
    current_weight = 0
    total_mean = 0
    between_class_variance = 0
    threshold = 0

    for i, weight in enumerate(histogram):
        current_weight += weight
        total_mean += weight * bin_centers[i]

    running_weight = 0
    running_mean = 0

    for i, weight in enumerate(histogram):
        running_weight += weight
        running_mean += weight * bin_centers[i]

        if current_weight == 0 or running_weight == current_weight:
            continue

        variance = (total_mean * running_weight - running_mean**2) / (running_weight * (current_weight - running_weight))
        
        if variance > between_class_variance:
            between_class_variance = variance
            threshold = bin_centers[i]

    return threshold

def otsu_thresholding(image, bias=0.9):
    """Determine the best threshold using a combination of Otsu's method and mean intensity."""
    otsu_thresh = compute_otsu_threshold(image)
    mean_thresh = np.mean(image)
    return bias * otsu_thresh + (1 - bias) * mean_thresh

def get_initial_state(image, dither_option):
    """Threshold the image for initial CA state based on dither_option."""
    lower_threshold = np.percentile(image, 30)
    upper_threshold = np.percentile(image, 60)

    if dither_option == "whites":
        return np.where(image > upper_threshold, 1, 0)
    elif dither_option == "blacks":
        return np.where(image <= lower_threshold, 1, 0)
    elif dither_option == "grays":
        return np.where((image > lower_threshold) & (image <= upper_threshold), 1, 0)
    else:
        return np.zeros_like(image)


def generate_rule_pattern(rule_number, state, original_image, dither_option):
    """Generates a cellular automaton pattern based on the given rule number and dithering option."""
    rule_bin = format(rule_number, '0256b')
    rules = {}
    for i in range(256):
        bin_rep = format(i, '08b')
        pattern = tuple(map(int, bin_rep))
        rules[pattern] = int(rule_bin[i])

    next_state = np.zeros_like(state)
    dithered_image = original_image.copy()
    n = state.shape[0]

    for i in range(n):
        for j in range(n):
            neighbors = (
                state[(i-1) % n, j], state[(i+1) % n, j], state[i, (j-1) % n], state[i, (j+1) % n],
                state[(i-1) % n, (j-1) % n], state[(i-1) % n, (j+1) % n], state[(i+1) % n, (j-1) % n], state[(i+1) % n, (j+1) % n]
            )
            next_state[i, j] = rules[neighbors]

            # Apply dithering to active pixels
            if next_state[i, j] == 1:
                dithered_image[i, j] = 0 if original_image[i, j] > 0.5 else 1
    
    return next_state, dithered_image

from PIL import ImageSequence

def evolve_and_save_by_rule(initial_state, original_image, rule_number, dither_option, num_iterations=10, iterations_per_frame=1, base_output_dir="output_frames"):
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, timestamp_str)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    state_frames = []
    dithered_frames = []
    
    state = initial_state
    for frame in range(num_iterations):
        img = Image.fromarray((state * 255).astype(np.uint8))
        img = img.convert("1")
        img_path = os.path.join(output_dir, f"state_{frame}.gif")
        img.save(img_path)
        state_frames.append(img_path)

        state, dithered = generate_rule_pattern(rule_number, state, original_image, dither_option)
        dithered_img = Image.fromarray((dithered * 255).astype(np.uint8))
        dithered_img = dithered_img.convert("1")
        dithered_img_path = os.path.join(output_dir, f"dithered_{frame}.gif")
        dithered_img.save(dithered_img_path)
        dithered_frames.append(dithered_img_path)

    # Create GIFs from the saved frames
    with Image.open(state_frames[0]) as img:
        img.save(os.path.join(output_dir, "state_evolution.gif"), save_all=True, append_images=[Image.open(frame) for frame in state_frames[1:]], duration=50, loop=0)
    
    with Image.open(dithered_frames[0]) as img:
        img.save(os.path.join(output_dir, "dithered_evolution.gif"), save_all=True, append_images=[Image.open(frame) for frame in dithered_frames[1:]], duration=50, loop=0)

    return output_dir




if __name__ == "__main__":
    filename = input("Enter the image filename/path: ")
    
    # Modify the grid_size input to accept only one value
    n = int(input("Enter the grid size (e.g., 256): "))
    grid_size = (n, n)  # Convert the single value to a tuple

    rule_number = int(input("Enter the rule number (0-255): "))
    dither_option = input("Choose which to dither (grays/blacks/whites): ").lower()
    num_generations = int(input("Enter the number of generations: "))
    iterations_per_frame = int(input("Enter the number of iterations per frame: "))

    image = load_and_convert_image(filename, grid_size)
    initial_state = get_initial_state(image, dither_option)

    output_dir = evolve_and_save_by_rule(initial_state, image, rule_number, dither_option, num_generations, iterations_per_frame)

    print(f"Frames saved in the '{output_dir}' directory.")

