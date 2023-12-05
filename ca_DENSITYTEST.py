# Necessary imports
import numpy as np
import matplotlib.pyplot as plt

# Define the cellular automaton rule function
def cellular_automaton_rule(rule_number):
    # Convert rule number to binary and pad with zeros
    rule_string = format(rule_number, '08b')
    return {format(i, '03b'): rule_string[7 - i] for i in range(8)}

def generate_ca_pattern(height, width, rule_number):
    rule = cellular_automaton_rule(rule_number)
    pattern = np.zeros((height, width), dtype=np.uint8)
    pattern[0, width // 2] = 1  # Starting with one cell in the center
    
    for y in range(1, height):
        for x in range(width):
            # Get the current cell and its two neighbors, with toroidal boundaries
            left = x - 1 if x - 1 >= 0 else width - 1
            right = x + 1 if x + 1 < width else 0
            neighbors = [pattern[y-1, left], pattern[y-1, x], pattern[y-1, right]]
            
            neighbors_str = ''.join(map(str, neighbors))
            
            # Determine the next state of the cell based on the rule
            pattern[y, x] = int(rule[neighbors_str])
    
    return pattern

# Define the function to get slices and measure density of active cells in each slice
def get_slices(generations, slice_count):
    num_generations, size = generations.shape
    slice_indices = np.linspace(0, size-1, slice_count, dtype=int)
    slice_densities = []

    for idx in slice_indices:
        slice_density = np.sum(generations[:, idx]) / num_generations
        slice_densities.append(slice_density)
    
    return slice_indices, slice_densities

# Define the function to compute density of active cells over generations
def density_over_generations(generations):
    return np.sum(generations, axis=1) / generations.shape[1]

# Define the function for visualizing the classic representation of the cellular automaton
def classic_representation(generations, num_generations=100):
    plt.figure(figsize=(10, 5))
    plt.imshow(generations[:num_generations], cmap='binary', aspect='auto')
    plt.title(f"Classic Representation of Rule 30 for {num_generations} Generations")
    plt.xlabel("Position")
    plt.ylabel("Generation")
    plt.show()

def generate_xor_combined_pattern(height, width, rule1, rule2):
    # Generate patterns for both rules
    pattern1 = generate_ca_pattern(height, width, rule1)
    pattern2 = generate_ca_pattern(height, width, rule2)
    
    # Combine the two patterns using XOR operation
    combined_pattern = np.logical_xor(pattern1, pattern2).astype(np.uint8)
    
    return combined_pattern




# Main function to generate and visualize the patterns
def main():
    # Generate Rule 30 for 1000 generations with a single active cell in the middle
    generations = generate_ca_pattern(1000, 201, 30)

    # Display the classic representation for the first 100 generations
    classic_representation(generations, 100)
    
    # Plot the density of active cells over generations
    densities = density_over_generations(generations)
    plt.plot(densities)
    plt.title("Density of Active Cells Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Density")
    plt.show()

    # Plot the generated pattern of Rule 30 for all 1000 generations
    plt.figure(figsize=(10, 5))
    plt.imshow(generations, cmap='binary', aspect='auto')
    plt.title("Rule 30 for 1000 Generations")
    plt.xlabel("Position")
    plt.ylabel("Generation")

    # Draw vectors representing slices on the Rule 30 pattern
    slice_count = 20
    slice_indices, _ = get_slices(generations, slice_count)
    for idx in slice_indices:
        plt.axvline(x=idx, color='red', linestyle='--')
    plt.show()

    # Get slices and their densities
    _, slice_densities = get_slices(generations, slice_count)

    # Plot the densities
    width = generations.shape[1] / slice_count
    plt.bar(slice_indices, slice_densities, width=width)
    plt.title("Density of Active Cells in Slices")
    plt.xlabel("Position")
    plt.ylabel("Density")
    plt.show()

        # Generate the XOR combined pattern for Rule 30 and Rule 73
    combined_pattern = generate_xor_combined_pattern(1000, 201, 30, 110)

    # Visualize the combined pattern
    plt.figure(figsize=(10, 5))
    plt.imshow(combined_pattern, cmap='binary', aspect='auto')
    plt.title("Combined Pattern of Rule 30 and Rule 73 using XOR")
    plt.xlabel("Position")
    plt.ylabel("Generation")
    plt.show()




# Execute the main function
#main()

# Updated main function to allow user input for generations and modulating rule
def main_user_input():
    # Prompt user for number of generations and modulating rule
    num_generations = int(input("Enter the number of generations (e.g., 1000): "))
    modulating_rule = int(input("Enter the modulating rule number (e.g., 73): "))
    
    # Generate Rule 30 pattern
    rule_30_pattern = generate_ca_pattern(num_generations, 201, 30)
    
    # Display the classic representation for the first 100 generations
    classic_representation(rule_30_pattern, 100)
    
    # Plot the density of active cells over generations
    densities = density_over_generations(rule_30_pattern)
    plt.plot(densities)
    plt.title("Density of Active Cells Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Density")
    plt.show()

    # Plot the generated pattern of Rule 30 for all generations
    plt.figure(figsize=(10, 5))
    plt.imshow(rule_30_pattern, cmap='binary', aspect='auto')
    plt.title(f"Rule 30 for {num_generations} Generations")
    plt.xlabel("Position")
    plt.ylabel("Generation")

    # Draw vectors representing slices on the Rule 30 pattern
    slice_count = 20
    slice_indices, _ = get_slices(rule_30_pattern, slice_count)
    for idx in slice_indices:
        plt.axvline(x=idx, color='red', linestyle='--')
    plt.show()

    # Get slices and their densities
    _, slice_densities = get_slices(rule_30_pattern, slice_count)

    # Plot the densities
    slice_width = rule_30_pattern.shape[1] / slice_count
    plt.bar(slice_indices, slice_densities, width=slice_width)
    plt.title("Density of Active Cells in Slices")
    plt.xlabel("Position")
    plt.ylabel("Density")
    plt.show()

    # Generate the XOR combined pattern using Rule 30 and the user-specified modulating rule
    combined_pattern = generate_xor_combined_pattern(num_generations, 201, 30, modulating_rule)

    # Visualize the combined pattern
    plt.figure(figsize=(10, 5))
    plt.imshow(combined_pattern, cmap='binary', aspect='auto')
    plt.title(f"Combined Pattern of Rule 30 and Rule {modulating_rule} using XOR")
    plt.xlabel("Position")
    plt.ylabel("Generation")
    plt.show()

    # Slice the combined pattern and plot the densities
    _, slice_densities = get_slices(combined_pattern, slice_count)
    plt.bar(slice_indices, slice_densities, width=slice_width)
    plt.title("Density of Active Cells in Slices")
    plt.xlabel("Position")
    plt.ylabel("Density")
    plt.show()

    # Plot the density of active cells over generations for the combined pattern
    densities = density_over_generations(combined_pattern)
    plt.plot(densities)
    plt.title("Density of Active Cells Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Density")
    plt.show()

    # Get the density of active cells over generations for Rule 30 and combined patterns
    rule_30_densities = density_over_generations(rule_30_pattern)

    combined_densities = density_over_generations(combined_pattern)
    
    # Plot the densities for comparison
    plt.figure(figsize=(10, 5))
    plt.plot(rule_30_densities, label="Rule 30", linestyle='-', color='blue')
    plt.plot(combined_densities, label=f"Combined (Rule 30 XOR Rule {modulating_rule})", linestyle='--', color='red')
    plt.title(f"Density Comparison Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    


    



# For demonstration purposes in this environment, I will run the main function with fixed values.
# In a local environment, you can run the function as-is to get user input.
main_user_input()
