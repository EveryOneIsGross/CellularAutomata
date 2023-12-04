import numpy as np

def generate_center_column(rule_number, n):
    """Generate the center column of a cellular automaton for the given rule number."""
    rule_bin = format(rule_number, '08b')
    rules = {
        (1, 1, 1): int(rule_bin[0]),
        (1, 1, 0): int(rule_bin[1]),
        (1, 0, 1): int(rule_bin[2]),
        (1, 0, 0): int(rule_bin[3]),
        (0, 1, 1): int(rule_bin[4]),
        (0, 1, 0): int(rule_bin[5]),
        (0, 0, 1): int(rule_bin[6]),
        (0, 0, 0): int(rule_bin[7])
    }

    grid = np.zeros((n, n), dtype=int)
    grid[0, n // 2] = 1

    for i in range(1, n):
        for j in range(n):
            left = grid[i-1, (j-1) % n]
            center = grid[i-1, j]
            right = grid[i-1, (j+1) % n]
            grid[i, j] = rules[(left, center, right)]

    return grid[:, n // 2]

def save_to_file(filename, content):
    with open(filename, 'a') as file:
        file.write(content + "\n")

def predict_from_prefix_simple(modulated_col, prefix_length=4):
    """Predict the column using a simple replication strategy."""
    n = len(modulated_col)
    predicted_col = np.zeros(n, dtype=int)
    prefix = modulated_col[:prefix_length]
    
    for i in range(n):
        predicted_col[i] = prefix[i % prefix_length]
    
    return predicted_col

def simplified_display_prediction():
    carrier = int(input("Enter the Carrier rule number (0-255): "))
    n = int(input("Enter the number of generations: "))

    modulator_rules = []
    while True:
        modulator = input("Enter a Modulator rule number (0-255) or press Enter to proceed: ")
        if modulator:
            modulator_rules.append(int(modulator))
        else:
            break

    filename = "rule30_results.txt"

    carrier_col = generate_center_column(carrier, n)

    modulated_col = carrier_col.copy()
    for modulator in modulator_rules:
        modulator_col = generate_center_column(modulator, n)
        modulated_col = modulated_col ^ modulator_col

    # Attempt to predict the original column by reversing the XOR modulation
    predicted_col_from_modulator = modulated_col.copy()
    for modulator in reversed(modulator_rules):
        modulator_col = generate_center_column(modulator, n)
        predicted_col_from_modulator = predicted_col_from_modulator ^ modulator_col

    predicted_col_simple_replication = predict_from_prefix_simple(modulated_col)

    # Save the results to file
    save_to_file(filename, "\n---\n\nRaw Center Column (Carrier Rule):")
    save_to_file(filename, ''.join(map(str, carrier_col)))
    
    save_to_file(filename, "\nModulated Center Column:")
    save_to_file(filename, ''.join(map(str, modulated_col)))

    save_to_file(filename, "\nPredicted Center Column from Modulator:")
    save_to_file(filename, ''.join(map(str, predicted_col_from_modulator)))

    save_to_file(filename, "\nPredicted Center Column based on first 10 characters:")
    save_to_file(filename, ''.join(map(str, predicted_col_simple_replication)))

    # Check accuracy
    accuracy_from_modulator = np.mean(carrier_col == predicted_col_from_modulator) * 100
    accuracy_simple_replication = np.mean(carrier_col == predicted_col_simple_replication) * 100
    save_to_file(filename, f"\nPrediction Accuracy from Modulator: {accuracy_from_modulator:.2f}%")
    save_to_file(filename, f"\nPrediction Accuracy from Simple Replication: {accuracy_simple_replication:.2f}%")

    # Also display results on the console
    print("\nRaw Center Column (Carrier Rule):")
    print(''.join(map(str, carrier_col)))

    print("\nModulated Center Column:")
    print(''.join(map(str, modulated_col)))

    print("\nPredicted Center Column from Modulator:")
    print(''.join(map(str, predicted_col_from_modulator)))

    print("\nPredicted Center Column based on first 10 characters:")
    print(''.join(map(str, predicted_col_simple_replication)))

    print(f"\nPrediction Accuracy from Modulator: {accuracy_from_modulator:.2f}%")
    print(f"\nPrediction Accuracy from Simple Replication: {accuracy_simple_replication:.2f}%")

    return carrier_col, modulated_col, predicted_col_from_modulator, predicted_col_simple_replication

# Running the function
if __name__ == "__main__":
    simplified_display_prediction()
