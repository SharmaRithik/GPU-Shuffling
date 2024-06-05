import ray
import time

# Initialize Ray
ray.init(address="auto")

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

# Step 3: Perform Random Shuffle and time the operation
start_time = time.time()
shuffled_dataset = dataset.random_shuffle()
# Force computation
shuffled_data = shuffled_dataset.take_all()
end_time = time.time()

# Calculate the time taken for shuffling
shuffle_time = end_time - start_time
print(f"Time taken for shuffling: {shuffle_time} seconds")

# Step 4: Save the Shuffled Data to output.txt
with open("output.txt", "w") as f:
    for item in shuffled_data:
        f.write(f"{item}\n")

# Shutdown Ray
ray.shutdown()

