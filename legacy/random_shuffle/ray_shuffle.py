import ray

# Initialize Ray
ray.init()

# Load data from input.txt
def load_data(file_path):
    with open(file_path, "r") as f:
        data = [float(line.strip()) for line in f]
    return data

# Step 1: Load the dataset from input.txt
data = load_data("input.txt")

# Step 2: Create Ray Dataset
from ray.data import from_items
dataset = from_items(data)

# Step 3: Perform Random Shuffle
shuffled_dataset = dataset.random_shuffle()

# Step 4: Convert the Shuffled Data to a List
shuffled_data = shuffled_dataset.take_all()

# Step 5: Save the Shuffled Data to output.txt
with open("output.txt", "w") as f:
    for item in shuffled_data:
        f.write(f"{item}\n")

# Shutdown Ray
ray.shutdown()

