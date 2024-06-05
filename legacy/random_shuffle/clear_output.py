import ast

def clean_data(file_path, output_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    cleaned_data = []
    for line in lines:
        # Parse the dictionary string and extract the value
        data = ast.literal_eval(line.strip())
        cleaned_data.append(data['item'])

    # Write the cleaned data to the output file
    with open(output_path, "w") as f:
        for item in cleaned_data:
            f.write(f"{item}\n")

# File paths
input_file = "output.txt"
output_file = "cleaned_output.txt"

# Clean the data
clean_data(input_file, output_file)

print(f"Cleaned data written to {output_file}")

