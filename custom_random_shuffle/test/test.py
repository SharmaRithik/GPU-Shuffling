import numpy as np
from numba import cuda

# A simple kernel that adds two arrays
@cuda.jit
def add_kernel(a, b, c):
    idx = cuda.grid(1)
    if idx < a.size:
        c[idx] = a[idx] + b[idx]

# Initialize arrays
n = 100000
a = np.random.random(n).astype(np.float32)
b = np.random.random(n).astype(np.float32)
c = np.zeros(n, dtype=np.float32)

# Allocate device memory
d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.to_device(c)

# Define the number of threads and blocks
threads_per_block = 128
blocks_per_grid = (a.size + (threads_per_block - 1)) // threads_per_block

# Run the kernel
add_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)

# Copy the result back to the host
d_c.copy_to_host(c)

# Verify the result
print(f"First 5 results: {c[:5]}")

