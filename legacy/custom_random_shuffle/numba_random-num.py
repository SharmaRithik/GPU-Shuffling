import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import time

# Define the CUDA kernel
@cuda.jit
def gpu_random_numbers(rng_states, result):
    idx = cuda.grid(1)
    if idx < result.size:
        result[idx] = xoroshiro128p_uniform_float32(rng_states, idx)

def main():
    n = 1000000  # Number of random numbers to generate

    # Create an array to store the result
    result = np.zeros(n, dtype=np.float32)

    # Allocate GPU memory for the result array
    d_result = cuda.to_device(result)

    # Create random number generator states
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    rng_states = create_xoroshiro128p_states(threads_per_block * blocks_per_grid, seed=42)

    # Measure the time taken to generate random numbers on the GPU
    start_time = time.time()
    gpu_random_numbers[blocks_per_grid, threads_per_block](rng_states, d_result)
    cuda.synchronize()
    end_time = time.time()

    # Copy the result back to the host
    d_result.copy_to_host(result)

    print(f"Time taken: {end_time - start_time:.6f} seconds")
    print("First 10 random numbers:", result[:10])

if __name__ == "__main__":
    main()

