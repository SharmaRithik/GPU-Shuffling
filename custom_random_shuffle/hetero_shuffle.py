import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from numba import cuda

def read_file_to_list(file_path):
    elements = []
    with open(file_path, 'r') as file:
        for line in file:
            elements.append(line.strip())
    return elements

def task_partition(elements, percentage):
    if not 0 <= percentage <= 100:
        raise ValueError("Percentage must be between 0 and 100")
    
    total_elements = len(elements)
    partition_index = int(total_elements * (percentage / 100))
    
    partition_1 = elements[:partition_index]
    partition_2 = elements[partition_index:]
    
    return partition_1, partition_2

def shuffle_chunk(chunk):
    random.shuffle(chunk)
    return chunk

def shuffle_On_Cpu(elements, num_cores):
    chunk_size = len(elements) // num_cores
    chunks = [elements[i * chunk_size: (i + 1) * chunk_size] for i in range(num_cores)]
    
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        shuffled_chunks = list(executor.map(shuffle_chunk, chunks))
    
    # Merge the shuffled chunks
    shuffled_elements = []
    for chunk in shuffled_chunks:
        shuffled_elements.extend(chunk)
    
    # Perform a final shuffle on the merged list
    random.shuffle(shuffled_elements)
    
    return shuffled_elements

@cuda.jit
def gpu_shuffle_kernel(arr, seed):
    i = cuda.grid(1)
    n = arr.shape[0]
    if i < n:
        j = i + seed % (n - i)
        if j < n:
            arr[i], arr[j] = arr[j], arr[i]

def shuffle_On_Gpu(elements):
    # Convert list to numpy array for GPU processing
    np_elements = np.array(elements)
    
    # Allocate device memory and copy data
    d_elements = cuda.to_device(np_elements)
    
    # Define block and grid sizes
    threads_per_block = 256
    blocks_per_grid = (len(elements) + (threads_per_block - 1)) // threads_per_block
    
    # Seed for random number generation
    seed = random.randint(0, len(elements))
    
    # Launch kernel to shuffle elements on GPU
    gpu_shuffle_kernel[blocks_per_grid, threads_per_block](d_elements, seed)
    
    # No need to copy back to host explicitly if using unified memory
    shuffled_elements = d_elements.copy_to_host()
    
    return shuffled_elements.tolist()

if __name__ == "__main__":
    file_path = 'input.txt'
    elements = read_file_to_list(file_path)
    print(f"Read {len(elements)} elements from the file.")
    
    # Example partition usage
    percentage = 30  # Example partition percentage
    partition_1, partition_2 = task_partition(elements, percentage)
    
    print(f"Partition 1: {len(partition_1)} elements")
    print(f"Partition 2: {len(partition_2)} elements")
    
    # Example shuffle usage on CPU
    num_cores = 4  # Example number of cores
    shuffled_elements_cpu = shuffle_On_Cpu(elements, num_cores)
    print(f"Shuffled elements on CPU: {len(shuffled_elements_cpu)}")
    
    # Example shuffle usage on GPU
    shuffled_elements_gpu = shuffle_On_Gpu(elements)
    print(f"Shuffled elements on GPU: {len(shuffled_elements_gpu)}")

