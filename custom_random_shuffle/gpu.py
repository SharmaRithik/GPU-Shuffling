import time
import sys
from numba import cuda
import numpy as np
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

@cuda.jit
def shuffle_kernel(arr, shuffled_arr, rng_states):
    idx = cuda.grid(1)
    if idx < len(arr):
        rand_idx = int(xoroshiro128p_uniform_float32(rng_states, idx) * len(arr))
        temp = arr[idx]
        shuffled_arr[idx] = arr[rand_idx]
        shuffled_arr[rand_idx] = temp

def gpu_shuffle(arr):
    n = len(arr)
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    rng_states = create_xoroshiro128p_states(threads_per_block * blocks_per_grid, seed=42)
    d_arr = cuda.to_device(arr)
    d_shuffled_arr = cuda.device_array_like(d_arr)
    
    shuffle_kernel[blocks_per_grid, threads_per_block](d_arr, d_shuffled_arr, rng_states)
    cuda.synchronize()
    
    shuffled_arr = d_shuffled_arr.copy_to_host()
    return shuffled_arr

def read_input(file_path):
    with open(file_path, 'r') as file:
        data = file.read().split()
        return list(map(int, data))

def write_output(file_path, data):
    with open(file_path, 'w') as file:
        for item in data:
            file.write(f"{item}\n")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 gpu.py <input_file>")
        return

    input_file = sys.argv[1]

    output_file = 'output.txt'

    data = read_input(input_file)
    data = np.array(data, dtype=np.int32)

    start_time = time.time()
    
    shuffled_data = gpu_shuffle(data)

    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"Merge shuffle took {elapsed_time:.6f} seconds")

    write_output(output_file, shuffled_data)

if __name__ == "__main__":
    main()

