import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import io

# global variables
kernel_size = 5
input_threshold = 128
num_generations = 32
allow_dying = True
lower_threshold = 128
upper_threshold = 225
mode = 'wrap' # other modes are 'constant', 'reflect', 'replicate', 'wrap'

image_path = 'Rule 30 by Rule 73.png'

def initialize_binary_grid(image_path, threshold=input_threshold):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_grid = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_grid

def apply_gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_laplacian(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)

def ca_evolution_step(image, generation, init_kernel_size=kernel_size, lower_threshold=lower_threshold, upper_threshold=upper_threshold, allow_dying=allow_dying):
    kernel_size = init_kernel_size + 2 * generation
    blurred = apply_gaussian_blur(image, kernel_size)
    laplacian = apply_laplacian(blurred)

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


def save_frame(ca_state, generation, output_folder, base_filename):
    fig, ax = plt.subplots()
    # available cmaps are here: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
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
    output_folder = f"output_mortal/{base_filename}_{current_time}_frames"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for generation in range(num_generations):
        ca_state = ca_evolution_step(ca_state, generation, allow_dying=True)
        frame = save_frame(ca_state, generation, output_folder, base_filename)
        all_frames.append(frame)
        if generation % 2 == 0:
            even_frames.append(frame)
        else:
            odd_frames.append(frame)

    return all_frames, even_frames, odd_frames, output_folder

def save_gif(frames, filename):
    imageio.mimsave(filename, frames, format='GIF', duration=0.5)

from datetime import datetime

# Other functions remain the same

if __name__ == "__main__":
    image_path = image_path  # Replace with your image path
    num_generations = num_generations

    # Extract the base name of the input image
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # Get the current datetime
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_frames, even_frames, odd_frames, output_folder = run_ca(image_path, num_generations)


    # Save GIFs with unique filenames
    save_gif(all_frames, f'{base_filename}_all_generations_{current_time}.gif')
    save_gif(even_frames, f'{base_filename}_even_generations_{current_time}.gif')
    save_gif(odd_frames, f'{base_filename}_odd_generations_{current_time}.gif')
