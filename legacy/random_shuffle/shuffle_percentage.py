def load_data(file_path, is_output=False):
    with open(file_path, "r") as f:
        data = [float(line.strip()) for line in f]
    return data

def calculate_shuffle_percentage(original_data, shuffled_data):
    total_elements = len(original_data)
    changed_positions = sum(1 for original, shuffled in zip(original_data, shuffled_data) if original != shuffled)
    shuffle_percentage = (changed_positions / total_elements) * 100
    return shuffle_percentage

# Load the original data from input.txt
original_data = load_data("input.txt")

# Load the shuffled data from output.txt
shuffled_data = load_data("cleaned_output.txt", is_output=True)

# Check the first few elements to ensure correct loading
# print("Original Data (First 10 elements):", original_data[:10])
# print("Shuffled Data (First 10 elements):", shuffled_data[:10])

# Calculate the percentage of shuffle
shuffle_percentage = calculate_shuffle_percentage(original_data, shuffled_data)
print(f"Shuffle Percentage: {shuffle_percentage:.2f}%")

