import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import datetime

def cellular_automaton_rule(rule_number):
    # Convert rule number to binary and pad with zeros
    rule_string = format(rule_number, '08b')
    return {format(i, '03b'): rule_string[7 - i] for i in range(8)}


def generate_ca_pattern(height, width, rule):
    pattern = np.zeros((height, width), dtype=np.uint8)
    pattern[0, width // 2] = 255  # Starting with one cell in the center
    
    for y in range(1, height):
        for x in range(1, width - 1):
            # Get the current cell and its two neighbors
            neighbors = pattern[y-1, x-1:x+2]
            neighbors_str = ''.join(map(str, neighbors // 255))
            
            # Determine the next state of the cell based on the rule
            index = 7 - int(neighbors_str, 2)
            pattern[y, x] = 255 if rule[format(index, '03b')] == '1' else 0

    return pattern

def generate_brians_brain_pattern(height, width):
    pattern = np.zeros((height, width), dtype=np.uint8)
    
    # Initialize some random cells to be "on"
    percentage_on = 0.1
    total_cells = height * width
    num_on = int(total_cells * percentage_on)
    on_indices_y, on_indices_x = np.divmod(np.random.choice(total_cells, num_on, replace=False), width)
    pattern[on_indices_y, on_indices_x] = 255
    
    for y in range(1, height):
        for x in range(1, width - 1):
            on_neighbors = 0
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if i == 0 and j == 0:  # Skip the center cell
                        continue
                    if 0 <= y + i < height and 0 <= x + j < width and pattern[y + i, x + j] == 255:
                        on_neighbors += 1

            if pattern[y - 1, x] == 255:  # if the cell is "on"
                pattern[y, x] = 127  # make it "dying"
            elif pattern[y - 1, x] == 127:  # if the cell is "dying"
                pattern[y, x] = 0  # make it "off"
            elif pattern[y - 1, x] == 0 and on_neighbors == 2:  # if the cell is "off" and has exactly 2 "on" neighbors
                pattern[y, x] = 255  # make it "on"

    return pattern



def generate_autostereogram(depth_map, pattern):
    height, width = depth_map.shape
    pattern_height, pattern_width = pattern.shape
    
    # Create a blank canvas for the stereogram
    stereogram = np.zeros((height, width), dtype=np.uint8)
    
    # Tile the pattern across the entire width of the stereogram
    for y in range(height):
        for x in range(width):
            stereogram[y, x] = pattern[y % pattern_height, x % pattern_width]
    
    max_shift = pattern_width // 1
    for y in range(height):
        for x in range(width):
            sensitivity = 0.5  # Adjust this value for more or less sensitivity
            shift = int(((depth_map[y, x] / 255) ** sensitivity) * max_shift)
            source_x = x - shift
            if source_x >= 0:
                stereogram[y, x] = stereogram[y, source_x]
    
    return stereogram



def generate_depth_map_from_image(image):
    # Assuming the input is now an image array instead of a file path
    # Convert the image to grayscale
    grayscale_image = 255 - np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8) # Invert the grayscale image
    #grayscale_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)


    return grayscale_image  # Use grayscale image directly as depth map


import imageio.v2 as imageio

def main_updated():
    rule_number = int(input("Enter a CA Rule number (0-255): "))
    rule_column_width = int(input("Enter the desired column width: "))
    gif_path = input("Enter the path to your animated GIF: ")
    pattern_choice = input("Choose a pattern generation method (1: Brian's Brain, 2: Classic Cellular Automaton): ")

    reader = imageio.get_reader(gif_path)
    frames = []
    for i, frame in enumerate(reader):
        depth_map = generate_depth_map_from_image(frame)
        rule = cellular_automaton_rule(rule_number)

        if pattern_choice == "1":
            pattern = generate_brians_brain_pattern(depth_map.shape[0], depth_map.shape[1] // 10)
        else:
            pattern = generate_ca_pattern(depth_map.shape[0], rule_column_width, rule)

        autostereogram = generate_autostereogram(depth_map, pattern)
        frames.append(autostereogram)

    # Handling the case where 'fps' metadata is not available
    try:
        fps = reader.get_meta_data()['fps']
    except KeyError:
        fps = 10  # Default FPS, you can change this as needed
    # Create a timestamp for file naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Correctly concatenate the output file path
    output_gif_path = 'magic_eye' + str(rule_number) + str(rule_column_width) + timestamp + '.gif'

    imageio.mimsave(output_gif_path, frames, format='GIF', fps=fps, loop=0)


    print("Saved animated GIF as", output_gif_path)

if __name__ == "__main__":
    main_updated()
