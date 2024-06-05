import numpy as np

# Generate 1 million random numbers
data = np.random.rand(1_000_000)

# Save the data to input.txt
with open("input.txt", "w") as f:
    for item in data:
        f.write(f"{item}\n")

