import ray
import time
import numpy as np
from numba import cuda
import math

# Initialize Ray
ray.init()

# GPU shuffling kernel
@cuda.jit
def gpu_shuffle_kernel(array, n, rng_states):
    idx = cuda.grid(1)
    if idx < n:
        j = int(rng_states[idx] * n)
        if j < n:
            array[idx], array[j] = array[j], array[idx]

@ray.remote(num_gpus=1)
def gpu_random_shuffle(array):
    n = array.size
    array_device = cuda.to_device(array)
    
    # Generate random states
    rng_states = np.random.random(n).astype(np.float32)
    rng_states_device = cuda.to_device(rng_states)

    threads_per_block = 256
    blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block
    
    # Call the kernel
    gpu_shuffle_kernel[blocks_per_grid, threads_per_block](array_device, n, rng_states_device)
    
    # Copy the shuffled array back to host
    shuffled_array = array_device.copy_to_host()
    return shuffled_array

def main():
    # Create a large dataset
    array_size = 10**7
    array = np.arange(array_size)
    
    # Measure Ray random_shuffle execution time
    ds = ray.data.from_numpy(array)
    start_time = time.time()
    shuffled_ds = ds.random_shuffle()
    shuffled_array_ray = shuffled_ds.to_pandas().to_numpy().flatten()
    ray_execution_time = time.time() - start_time
    print(f"Ray random_shuffle Execution Time: {ray_execution_time} seconds")
    
    # Measure custom GPU shuffle execution time
    start_time = time.time()
    shuffled_array_gpu = ray.get(gpu_random_shuffle.remote(array))
    gpu_execution_time = time.time() - start_time
    print(f"Custom GPU shuffle Execution Time: {gpu_execution_time} seconds")

if __name__ == "__main__":
    main()
    
    # Shutdown Ray
    ray.shutdown()

