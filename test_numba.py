from numba import cuda
import numpy as np

@cuda.jit
def add_kernel(a, b, c):
    idx = cuda.grid(1)
    if idx < a.size:
        c[idx] = a[idx] + b[idx]

N = 1000
a = np.arange(N, dtype=np.float32)
b = np.arange(N, dtype=np.float32)
c = np.zeros(N, dtype=np.float32)

d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
d_c = cuda.to_device(c)

threads_per_block = 256
blocks_per_grid = (a.size + (threads_per_block - 1)) // threads_per_block
add_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)

# Copy the result back to the host
d_c.copy_to_host(c)
print(c[:10])  # Print first 10 elements to verify

