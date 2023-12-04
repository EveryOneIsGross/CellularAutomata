import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import io
from datetime import datetime
import cv2

# Global variables
n = 512  # Grid size 128
num_generations = 77

kernel_size = 4 # 5
input_threshold = 64 # 64
allow_dying = True #True
lower_threshold = 125 # 125
upper_threshold = 175 # 175
mode = 'constant' # wrap or constant border

init_type = "random"  # Options: "random", "cluster"
cluster_size = 5  # Size of the cluster for 'cluster' initialization

cell_seed = [1]  # Binary seed string for cluster construction


def initialize_grid(n, init_type="random", cluster_size=cluster_size, cell_seed=None):
    grid = np.zeros((n, n), dtype=np.uint8)
    if init_type == "random":
        # Adjust the probability to make the grid sparser
        # For example, 90% chance of 0 and 10% chance of 10
        grid = np.random.choice([0, 10], size=(n, n), p=[0.9, 0.1])
    elif init_type == "cluster" and cell_seed is not None:
        start = n // 2 - cluster_size // 2
        end = start + cluster_size
        seed_len = len(cell_seed)
        seed_index = 0
        for i in range(start, end):
            for j in range(start, end):
                grid[i, j] = 255 * cell_seed[seed_index % seed_len]
                seed_index += 1
    return grid


def apply_gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_laplacian(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)

def save_frame(grid, generation, output_folder, base_filename):
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='binary')
    ax.axis('off')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=72)
    buf.seek(0)
    frame = imageio.imread(buf)
    frame_filename = f"{output_folder}/{base_filename}_gen_{generation+1}.png"
    imageio.imwrite(frame_filename, frame)
    buf.close()
    plt.close(fig)
    return frame

def ca_evolution_step(image, generation, init_kernel_size=kernel_size, blur_iterations=3, lower_threshold=lower_threshold, upper_threshold=upper_threshold, allow_dying=allow_dying):
    blurred_image = image.astype(np.uint8)  # Convert to uint8
    for _ in range(blur_iterations):
        kernel_size = init_kernel_size + 1 * generation # increase kernel size by 1 each generation
        kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1  # Ensure kernel size is odd
        blurred_image = apply_gaussian_blur(blurred_image, kernel_size) # apply gaussian blur

    laplacian = apply_laplacian(blurred_image) # apply laplacian
    wrapped_laplacian = np.pad(laplacian, pad_width=1, mode=mode)

    new_state = np.zeros_like(image)
    for i in range(1, wrapped_laplacian.shape[0] - 1):
        for j in range(1, wrapped_laplacian.shape[1] - 1):
            neighborhood = wrapped_laplacian[i-1:i+2, j-1:j+2]
            neighborhood_average = np.mean(neighborhood)

            if allow_dying:
                if lower_threshold < neighborhood_average < upper_threshold:
                    new_state[i-1, j-1] = 255
                else:
                    new_state[i-1, j-1] = 0
            else:
                new_state[i-1, j-1] = 255 if neighborhood_average >= lower_threshold else 0

    return new_state


def run_ca(n, num_generations, init_type="random"):
    ca_state = initialize_grid(n, init_type, cluster_size, cell_seed)
    all_frames = []
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = "ca_grid"
    output_folder = f"output_ca/{base_filename}_{current_time}_frames"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for generation in range(num_generations):
        ca_state = ca_evolution_step(ca_state, generation)
        frame = save_frame(ca_state, generation, output_folder, base_filename)
        all_frames.append(frame)
    return all_frames, output_folder



def save_gif(frames, output_folder, filename):
    filepath = os.path.join(output_folder, filename)
    imageio.mimsave(filepath, frames, format='GIF', duration=0.5)

if __name__ == "__main__":
    all_frames, output_folder = run_ca(n, num_generations, init_type)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save GIF with all frames
    save_gif(all_frames, output_folder, f'generations_all_{current_time}.gif')

    # Save GIF with odd frames
    odd_frames = all_frames[1::2]  # Start from index 1 and take every 2nd element
    save_gif(odd_frames, output_folder, f'generations_odd_{current_time}.gif')

    # Save GIF with even frames
    even_frames = all_frames[0::2]  # Start from index 0 and take every 2nd element
    save_gif(even_frames, output_folder, f'generations_even_{current_time}.gif')
