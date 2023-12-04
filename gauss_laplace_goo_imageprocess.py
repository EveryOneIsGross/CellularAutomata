import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import io
from datetime import datetime

#global variables
kernel_size = 5 #5
CA_threshold = 128 #128

num_generations = 256 #128

import_threshold = 128 #128
image_path = 'originally based on Rule 7.png'



def initialize_binary_grid(image_path, threshold=import_threshold):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_grid = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_grid

def apply_gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_laplacian(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)

def ca_evolution_step(image, generation, init_kernel_size=kernel_size, threshold=CA_threshold):
    kernel_size = init_kernel_size + 2 * generation
    blurred = apply_gaussian_blur(image, kernel_size)
    laplacian = apply_laplacian(blurred)
    _, new_state = cv2.threshold(laplacian, threshold, 255, cv2.THRESH_BINARY)
    return new_state

def save_frame(ca_state, generation, output_folder, base_filename):
    fig, ax = plt.subplots()
    ax.imshow(ca_state, cmap='binary')
    #ax.set_title(f"Generation {generation + 1}")
    ax.axis('off')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    frame = imageio.imread(buf)

    frame_filename = f"{output_folder}/{base_filename}_gen_{generation+1}.png"
    imageio.imwrite(frame_filename, frame)

    buf.close()
    plt.close(fig)
    return frame

def run_ca(image_path, num_generations):
    ca_state = initialize_binary_grid(image_path)
    all_frames, even_frames, odd_frames = [], [], []

    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # Create a date-stamped output folder
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"output_immortal/{base_filename}_{current_time}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for generation in range(num_generations):
        ca_state = ca_evolution_step(ca_state, generation)
        frame = save_frame(ca_state, generation, output_folder, base_filename)
        all_frames.append(frame)
        if generation % 2 == 0:
            even_frames.append(frame)
        else:
            odd_frames.append(frame)

    return all_frames, even_frames, odd_frames, output_folder

def save_gif(frames, output_folder, filename):
    filepath = os.path.join(output_folder, filename)
    imageio.mimsave(filepath, frames, format='GIF', duration=0.5)

if __name__ == "__main__":
    image_path = image_path  # Replace with your image path
    num_generations = num_generations

    all_frames, even_frames, odd_frames, output_folder = run_ca(image_path, num_generations)

    # Save GIFs with unique filenames
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_gif(all_frames, output_folder, f'all_generations_{current_time}.gif')
    save_gif(even_frames, output_folder, f'even_generations_{current_time}.gif')
    save_gif(odd_frames, output_folder, f'odd_generations_{current_time}.gif')