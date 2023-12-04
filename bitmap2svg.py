import cv2
import numpy as np
import svgwrite
import datetime

# Global variables
compression_ratio = 0.1
inverse_image = True
use_circles_for_points = False  # Toggle between circles or lines for points

def convert_to_binary(image_path, threshold=128, compression_ratio=1.0):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if inverse_image:
        image = cv2.bitwise_not(image)
    
    if compression_ratio != 1.0:
        new_size = (int(image.shape[1] * compression_ratio), int(image.shape[0] * compression_ratio))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def save_svg(binary_image, output_path, point_size=1):
    height, width = binary_image.shape
    dwg = svgwrite.Drawing(output_path, size=(width, height), profile='tiny')
    
    for i in range(height):
        for j in range(width):
            if binary_image[i, j] == 255:
                if use_circles_for_points:
                    dwg.add(dwg.circle(center=(j, i), r=point_size, fill='black'))
                else:
                    start_end = (j, i)
                    dwg.add(dwg.line(start=start_end, end=start_end, stroke='black', stroke_width=point_size))
    
    dwg.save()

def process_image_to_svg(input_image_path, output_svg_path, threshold=128, point_size=1, compression_ratio=1.0):
    binary_image = convert_to_binary(input_image_path, threshold, compression_ratio)
    save_svg(binary_image, output_svg_path, point_size)

if __name__ == "__main__":
    input_image_path = 'output_ca\ca_grid_20231201_133756_frames\ca_grid_gen_43.png'  # Replace with your image path
    output_svg_path = 'output_image' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.svg'  # Replace with your output path
    process_image_to_svg(input_image_path, output_svg_path, threshold=128, point_size=1, compression_ratio=compression_ratio)
