import random
import time
import concurrent.futures
import numpy as np
from numba import cuda, float32
import numba.cuda.random as cuda_random

# CPU functions
def fisher_yates_shuffle(arr):
    n = len(arr)
    for i in range(n - 1, 0, -1):
        j = random.randint(0, i)
        arr[i], arr[j] = arr[j], arr[i]

def merge(arr, start, mid, end):
    left = arr[start:mid]
    right = arr[mid:end]
    i = j = 0
    k = start

    while i < len(left) and j < len(right):
        if random.randint(0, 1) == 0:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1

    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1

    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1

    for i in range(start, end):
        swap_index = random.randint(start, i)
        arr[i], arr[swap_index] = arr[swap_index], arr[i]

def merge_shuffle(arr, k):
    n = len(arr)
    if n <= k:
        fisher_yates_shuffle(arr)
        return

    block_size = (n + 2 * k - 1) // (2 * k)
    blocks = [arr[i:i + block_size] for i in range(0, n, block_size)]

    for block in blocks:
        fisher_yates_shuffle(block)

    while len(blocks) > 1:
        new_blocks = []
        for i in range(0, len(blocks) - 1, 2):
            new_block = blocks[i] + blocks[i + 1]
            merge(new_block, 0, len(blocks[i]), len(new_block))
            new_blocks.append(new_block)
        if len(blocks) % 2 == 1:
            new_blocks.append(blocks[-1])
        blocks = new_blocks

    for i in range(len(arr)):
        arr[i] = blocks[0][i]

# GPU functions
@cuda.jit
def fisher_yates_shuffle_gpu(arr, rng_states):
    tid = cuda.grid(1)
    n = arr.size
    if tid < n:
        for i in range(n - 1, 0, -1):
            j = int(cuda_random.xoroshiro128p_uniform_float32(rng_states, tid) * (i + 1))
            arr[i], arr[j] = arr[j], arr[i]

@cuda.jit
def merge_gpu(arr, left, right, rng_states):
    tid = cuda.grid(1)
    n = arr.size
    if tid < n:
        l_idx = 0
        r_idx = 0
        while l_idx < left.size and r_idx < right.size:
            if cuda_random.xoroshiro128p_uniform_float32(rng_states, tid) < 0.5:
                arr[tid] = left[l_idx]
                l_idx += 1
            else:
                arr[tid] = right[r_idx]
                r_idx += 1
            tid += cuda.gridDim.x * cuda.blockDim.x

def gpu_merge_shuffle(arr, k):
    print("Starting gpu_merge_shuffle")
    arr = np.array(arr, dtype=np.float32)
    arr_gpu = cuda.to_device(arr)
    n = arr_gpu.size
    rng_states = cuda_random.create_xoroshiro128p_states(n, seed=1)
    threads_per_block = 256
    blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block

    if n <= k:
        print("Running fisher_yates_shuffle_gpu")
        fisher_yates_shuffle_gpu[blocks_per_grid, threads_per_block](arr_gpu, rng_states)
        cuda.synchronize()
        print("fisher_yates_shuffle_gpu completed")
        arr = arr_gpu.copy_to_host()
        return arr

    block_size = (n + 2 * k - 1) // (2 * k)
    blocks = [arr_gpu[i:i + block_size] for i in range(0, n, block_size)]

    for block in blocks:
        print("Running fisher_yates_shuffle_gpu on block")
        fisher_yates_shuffle_gpu[blocks_per_grid, threads_per_block](block, rng_states)
        cuda.synchronize()
        print("fisher_yates_shuffle_gpu block completed")

    while len(blocks) > 1:
        new_blocks = []
        for i in range(0, len(blocks) - 1, 2):
            left_block = blocks[i].copy_to_host()
            right_block = blocks[i + 1].copy_to_host()
            new_block_host = np.concatenate((left_block, right_block))
            new_block_gpu = cuda.to_device(new_block_host)
            print("Running merge_gpu on new block")
            merge_gpu[blocks_per_grid, threads_per_block](new_block_gpu, cuda.to_device(left_block), cuda.to_device(right_block), rng_states)
            cuda.synchronize()
            print("merge_gpu new block completed")
            new_blocks.append(new_block_gpu)
        if len(blocks) % 2 == 1:
            new_blocks.append(blocks[-1])
        blocks = new_blocks

    cuda.synchronize()
    arr = blocks[0].copy_to_host()
    print("gpu_merge_shuffle completed")
    return arr

def read_input(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def write_output(filename, arr):
    with open(filename, 'w') as file:
        for item in arr:
            file.write(f"{item}\n")

def check_elements(original, shuffled):
    return set(original) == set(shuffled)

def calculate_shuffle_percentage(original, shuffled):
    match_count = sum(1 for o, s in zip(original, shuffled) if o == s)
    total_count = len(original)
    return (total_count - match_count) / total_count * 100

def shuffle_task(elements, k):
    merge_shuffle(elements, k)
    return elements

def gpu_shuffle_task(elements, k):
    return gpu_merge_shuffle(elements, k)

def benchmark(elements, percentages, k):
    results = []
    for percentage in percentages:
        cpu_elements_count = int(len(elements) * (percentage / 100))
        gpu_elements_count = len(elements) - cpu_elements_count

        cpu_elements = elements[:cpu_elements_count]
        gpu_elements = elements[cpu_elements_count:]

        start_time_cpu = time.time()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            cpu_future = executor.submit(shuffle_task, cpu_elements, k)
            cpu_elements = cpu_future.result()
        end_time_cpu = time.time()

        start_time_gpu = time.time()
        gpu_elements = gpu_shuffle_task(gpu_elements, k)
        end_time_gpu = time.time()

        cpu_time = end_time_cpu - start_time_cpu
        gpu_time = end_time_gpu - start_time_gpu

        results.append((percentage, cpu_time, gpu_time, cpu_elements_count, gpu_elements_count))

    return results

if __name__ == "__main__":
    input_filename = "input.txt"
    output_filename = "output.txt"

    elements = read_input(input_filename)
    original_elements = elements.copy()

    k = 32  # Adjust this value to find the optimal performance
    percentages = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

    benchmark_results = benchmark(elements, percentages, k)

    for percentage, cpu_time, gpu_time, cpu_elements_count, gpu_elements_count in benchmark_results:
        print(f"CPU/GPU Distribution: {percentage}% / {100-percentage}%, CPU Time Taken: {cpu_time:.4f} seconds, GPU Time Taken: {gpu_time:.4f} seconds, CPU Elements Shuffled: {cpu_elements_count}, GPU Elements Shuffled: {gpu_elements_count}")

    # Optionally, shuffle the entire array and save to output file
    merge_shuffle(elements, k)
    if check_elements(original_elements, elements):
        write_output(output_filename, elements)
        shuffle_percentage = calculate_shuffle_percentage(original_elements, elements)
        print(f"Shuffle Percentage: {shuffle_percentage:.2f}%")
    else:
        print("Error: Elements missing after shuffling!")

