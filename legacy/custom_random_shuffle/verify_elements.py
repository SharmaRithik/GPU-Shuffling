def load_data(file_path):
    with open(file_path, "r") as f:
        data = [float(line.strip()) for line in f]
    return data

def verify_elements(input_file, output_file):
    # Load data from both files
    input_data = load_data(input_file)
    output_data = load_data(output_file)
    
    # Convert the output data to a set for faster lookup
    output_set = set(output_data)
    
    # Check if all elements in input data are present in output data
    missing_elements = [item for item in input_data if item not in output_set]
    
    if not missing_elements:
        print("All elements from input.txt are present in cleaned_output.txt.")
    else:
        print(f"Missing elements: {missing_elements}")

# File paths
#input_file = "input.txt"
input_file = "a.txt"
output_file = "output.txt"

# Verify the elements
verify_elements(input_file, output_file)

