import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from numba import cuda
import math

def read_file_to_list(file_path):
    with open(file_path, 'r') as file:
        return [float(line.strip()) for line in file]  # Read floats directly

def task_partition(elements, percentage):
    if not 0 <= percentage <= 100:
        raise ValueError("Percentage must be between 0 and 100")
    
    partition_index = int(len(elements) * (percentage / 100))
    return elements[:partition_index], elements[partition_index:]

def shuffle_chunk(chunk):
    random.shuffle(chunk)
    return chunk

def shuffle_on_cpu(elements, num_cores):
    chunk_size = len(elements) // num_cores
    chunks = [elements[i * chunk_size: (i + 1) * chunk_size] for i in range(num_cores)]
    if len(elements) % num_cores != 0:  # Handle remaining elements in the last chunk
        chunks[-1].extend(elements[num_cores * chunk_size:])
    
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        shuffled_chunks = list(executor.map(shuffle_chunk, chunks))
    
    shuffled_elements = [element for chunk in shuffled_chunks for element in chunk]
    random.shuffle(shuffled_elements)
    
    return shuffled_elements

@cuda.jit
def gpu_shuffle_kernel(arr, seed):
    i = cuda.grid(1)
    n = arr.shape[0]
    if i < n:
        rnd = seed + i
        j = i + rnd % (n - i)
        if j < n:
            arr[i], arr[j] = arr[j], arr[i]

def shuffle_on_gpu(elements):
    np_elements = np.array(elements, dtype=np.float32)  # Use float32 for floating-point numbers
    
    d_elements = cuda.to_device(np_elements)
    
    threads_per_block = 256
    blocks_per_grid = (len(elements) + threads_per_block - 1) // threads_per_block
    
    seed = random.randint(0, len(elements))
    
    gpu_shuffle_kernel[blocks_per_grid, threads_per_block](d_elements, seed)
    cuda.synchronize()
    
    shuffled_elements = d_elements.copy_to_host()
    
    return shuffled_elements.tolist()

if __name__ == "__main__":
    file_path = 'input.txt'
    elements = read_file_to_list(file_path)
    print(f"Read {len(elements)} elements from the file.")
    
    percentage = 30
    partition_1, partition_2 = task_partition(elements, percentage)
    
    print(f"Partition 1: {len(partition_1)} elements")
    print(f"Partition 2: {len(partition_2)} elements")
    
    num_cores = 4
    shuffled_elements_cpu = shuffle_on_cpu(elements, num_cores)
    print(f"Shuffled elements on CPU: {len(shuffled_elements_cpu)}")
    
    shuffled_elements_gpu = shuffle_on_gpu(elements)
    print(f"Shuffled elements on GPU: {len(shuffled_elements_gpu)}")

