import pygame
import numpy as np
import pyaudio
from queue import Queue

# Constants
GRID_SIZE = (8, 8, 7)  # 3D grid size
CELL_SIZE = 64  # Size of each cell in pixels
FRAME_RATE = 2  # Frames per second

# Game of Life rules
BIRTH_RULE = [3]  # Number of neighbors for a dead cell to become alive
SURVIVAL_RULE = [2, 1]  # Number of neighbors for an alive cell to stay alive

# Initialize Pygame
pygame.init()

# Create a font for rendering text
font = pygame.font.SysFont("Comic Sans MS", 24, bold=True)

# Window settings
WINDOW_WIDTH = GRID_SIZE[0] * CELL_SIZE
WINDOW_HEIGHT = GRID_SIZE[1] * CELL_SIZE
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("3D Game of Life")

# Initialize 3D grid with random states (0: dead, 1: alive)
grid = np.random.choice([0, 1], size=GRID_SIZE)

def get_neighbors_3d(x, y, z, grid):
    """Calculate the number of alive neighboring cells for a given cell in the 3D grid."""
    neighbors = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < GRID_SIZE[0] and 0 <= ny < GRID_SIZE[1] and 0 <= nz < GRID_SIZE[2]:
                    # Decrease the survival rate based on the layer
                    survival_rate = 1 / (nz + 1)
                    if np.random.random() < survival_rate:
                        neighbors += grid[nx][ny][nz]
    return neighbors

def game_of_life_3d_step(grid):
    """Update the grid state based on the Game of Life rules."""
    new_grid = np.copy(grid)
    for x in range(GRID_SIZE[0]):
        for y in range(GRID_SIZE[1]):
            for z in range(GRID_SIZE[2]):
                neighbors = get_neighbors_3d(x, y, z, grid)
                if grid[x][y][z] == 1:
                    if neighbors not in SURVIVAL_RULE:
                        new_grid[x][y][z] = 0  # Cell dies
                else:
                    if neighbors in BIRTH_RULE:
                        new_grid[x][y][z] = 1  # Cell becomes alive
    return new_grid

# Initialize PyAudio
pa = pyaudio.PyAudio()
sound_queue = Queue()

def generate_sine_wave(freq, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(freq * t * 2 * np.pi)
    return tone

def apply_fm_synthesis(carrier_freq, modulator_freq, modulation_index, duration):
    t = np.linspace(0, duration, int(44100 * duration), False)
    modulator = np.sin(modulator_freq * t * 2 * np.pi)
    carrier = np.sin((carrier_freq + modulator * modulation_index) * t * 2 * np.pi)
    return carrier

def char_to_freq(char, x, y, z, grid_size):
    base_freq = 440  # Base frequency for A4 note

    # Modulate frequency based on grid position
    x_mod = (x / grid_size[0]) * 100  # Example modulation based on x
    y_mod = (y / grid_size[1]) * 100  # Example modulation based on y
    z_mod = (z / grid_size[2]) * 100  # Example modulation based on z

    return base_freq + (ord(char) - ord('a')) * 5 + x_mod + y_mod + z_mod

def audio_callback(in_data, frame_count, time_info, status):
    current_time = pygame.time.get_ticks()
    data = np.zeros(frame_count)  # Initialize buffer with silence
    
    while not sound_queue.empty():
        sound_wave, start_time = sound_queue.queue[0]
        if current_time >= start_time:
            # Ensure sound_wave is the correct length
            # Play only a portion of the sound_wave based on frame_count
            sound_wave_portion = sound_wave[:frame_count]
            data = np.maximum(data, sound_wave_portion)  # Mix the sound with the current buffer
            sound_queue.get()  # Remove the played sound from the queue
        else:
            break  # Exit the loop if the next sound is not ready to be played

    return (data.astype(np.float32).tobytes(), pyaudio.paContinue)


stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=44100, output=True, stream_callback=audio_callback)
stream.start_stream()

# User-input string and reset flag
user_input = ""
reset_game = False

# Initialize the current layer
current_layer = GRID_SIZE[2] // 2  # Start at the middle layer

# Main loop
running = True
clock = pygame.time.Clock()
input_index = 0  # Index to keep track of which character in the input to display
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                # Reset the game when 'ESC' is pressed
                grid = np.random.choice([0, 1], size=GRID_SIZE)
                user_input = ""
                input_index = 0
            elif event.key == pygame.K_RETURN:
                # Clear the user input when 'RETURN' is pressed
                user_input = ""
                input_index = 0
            elif event.key == pygame.K_BACKSPACE:
                # Remove the last character from user input
                user_input = user_input[:-1]
                input_index = 0
            elif event.key == pygame.K_LEFTBRACKET:
                # Move to the layer above
                current_layer = max(0, current_layer - 1)
            elif event.key == pygame.K_RIGHTBRACKET:
                # Move to the layer below
                current_layer = min(GRID_SIZE[2] - 1, current_layer + 1)
            elif event.key == pygame.K_PAGEUP:
                # Increase frame rate
                FRAME_RATE += 1
            elif event.key == pygame.K_PAGEDOWN:
                # Decrease frame rate, ensure it stays above 1
                FRAME_RATE = max(1, FRAME_RATE - 1)
            elif event.unicode.isprintable():
                # Append only printable characters to user input
                user_input += event.unicode



    new_grid = game_of_life_3d_step(grid)

    # Clear the sound queue at the start of each frame
    sound_queue.queue.clear()

    # Sound generation logic only activates if user_input is not empty
    if user_input:  # Check if user_input is not empty
        for x in range(GRID_SIZE[0]):
            for y in range(GRID_SIZE[1]):
                # Only consider the current layer
                current_state = grid[x][y][current_layer]
                new_state = new_grid[x][y][current_layer]

                # Generate sound only if the cell in the current layer changes from dead to alive
                if current_state == 0 and new_state == 1:
                    # Sound generation logic (as previously defined)
                    char = user_input[input_index % len(user_input)] if user_input else 'a'
                    # Corrected: Pass all required arguments to char_to_freq
                    freq = char_to_freq(char, x, y, current_layer, GRID_SIZE)

                    mod_freq = 10 * (current_layer + freq)
                    mod_index = 5
                    # duration is number of active cells in the current layer
                    #duration = np.sum(new_grid[:, :, current_layer])
                    # duration is low processor intensive count of active cells in the current layer
                    duration = 0.5 + current_layer
                    sound_wave = apply_fm_synthesis(freq, mod_freq, mod_index, duration)
                    start_time = pygame.time.get_ticks()
                    sound_queue.put((sound_wave, start_time))

    grid = new_grid

    # Clear the window
    window.fill((255, 255, 255))

    # Render the 3D grid with characters from user input for the current layer
    for x in range(GRID_SIZE[0]):
        for y in range(GRID_SIZE[1]):
            cell_rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if grid[x][y][current_layer] == 1:
                pygame.draw.rect(window, (255, 255, 255), cell_rect)
                color_value = (255 * current_layer) // (GRID_SIZE[2] - 1)
                text_color = (color_value, color_value, color_value)
                character = user_input[input_index % len(user_input)] if user_input else " "
                text = font.render(character, True, text_color)
                text_rect = text.get_rect()
                text_rect.center = (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2)
                window.blit(text, text_rect)
                input_index += 1

    # Update the display
    pygame.display.flip()

    # Limit the frame rate
    clock.tick(FRAME_RATE)

# Cleanup
stream.stop_stream()
stream.close()
pa.terminate()
pygame.quit()
