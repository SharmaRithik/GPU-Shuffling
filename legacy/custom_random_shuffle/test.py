import random
import time
import sys
from numba import cuda
import numpy as np
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

def merge_shuffle_cpu(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        merge_shuffle_cpu(left_half)
        merge_shuffle_cpu(right_half)

        i = j = k = 0
        while i < len(left_half) and j < len(right_half):
            if random.random() < 0.5:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

@cuda.jit
def merge_shuffle_gpu_kernel(arr, left_half, right_half, mid, rng_states):
    idx = cuda.grid(1)
    if idx < len(arr):
        rand_val = xoroshiro128p_uniform_float32(rng_states, idx)
        if idx < mid:
            if rand_val < 0.5:
                arr[idx] = left_half[idx]
            else:
                arr[idx] = right_half[idx]
        else:
            if rand_val < 0.5:
                arr[idx] = left_half[idx - mid]
            else:
                arr[idx] = right_half[idx - mid]

def merge_shuffle_gpu(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        merge_shuffle_gpu(left_half)
        merge_shuffle_gpu(right_half)

        d_arr = cuda.to_device(arr)
        d_left_half = cuda.to_device(left_half)
        d_right_half = cuda.to_device(right_half)

        threads_per_block = 256
        blocks_per_grid = (len(arr) + threads_per_block - 1) // threads_per_block

        rng_states = create_xoroshiro128p_states(threads_per_block * blocks_per_grid, seed=42)

        merge_shuffle_gpu_kernel[blocks_per_grid, threads_per_block](d_arr, d_left_half, d_right_half, mid, rng_states)
        cuda.synchronize()

        d_arr.copy_to_host(arr)

def read_input(file_path):
    with open(file_path, 'r') as file:
        data = file.read().split()
        return list(map(float, data))

def write_output(file_path, data):
    with open(file_path, 'w') as file:
        file.write(' '.join(map(str, data)))

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 test.py <input_file> <cpu_percentage> <gpu_percentage>")
        return

    input_file = sys.argv[1]
    cpu_percentage = int(sys.argv[2])
    gpu_percentage = int(sys.argv[3])

    if cpu_percentage + gpu_percentage != 100:
        print("CPU and GPU percentages must sum up to 100")
        return

    output_file = 'output.txt'

    data = read_input(input_file)
    
    cpu_split = int(len(data) * cpu_percentage / 100)
    cpu_part = data[:cpu_split]
    gpu_part = data[cpu_split:]

    start_time = time.time()

    if cpu_percentage > 0:
        merge_shuffle_cpu(cpu_part)

    if gpu_percentage > 0:
        gpu_part = np.array(gpu_part, dtype=np.float32)
        merge_shuffle_gpu(gpu_part)

    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"Merge shuffle took {elapsed_time:.6f} seconds")

    shuffled_data = cpu_part + list(gpu_part)
    
    write_output(output_file, shuffled_data)

if __name__ == "__main__":
    main()
