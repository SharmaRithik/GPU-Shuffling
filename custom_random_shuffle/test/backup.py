import numpy as np
from numba import cuda, float32
import ray
import time
import random

# GPU-based shuffling kernel
@cuda.jit
def shuffle_kernel(arr, indices, out):
    idx = cuda.grid(1)
    if idx < len(arr):
        out[idx] = arr[indices[idx]]

def gpu_advanced_shuffle(arr):
    n = len(arr)
    indices = np.arange(n).astype(np.int32)
    np.random.shuffle(indices)
    
    d_arr = cuda.to_device(arr)
    d_indices = cuda.to_device(indices)
    d_out = cuda.device_array_like(d_arr)

    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    shuffle_kernel[blocks_per_grid, threads_per_block](d_arr, d_indices, d_out)

    return d_out.copy_to_host()

# Ray-based shuffling function for CPU
@ray.remote
def shuffle_partition_cpu(partition):
    partition_list = list(partition.take_all())
    np.random.shuffle(partition_list)
    return partition_list

# Ray-based GPU shuffling function using the new advanced shuffle
@ray.remote(num_gpus=1)
def shuffle_partition_gpu(partition):
    partition_list = list(partition.take_all())
    # Handle nested dictionary structures
    if isinstance(partition_list[0], dict):
        partition_list = [item for sublist in partition_list for item in sublist.values()]
    partition_array = np.array(partition_list).astype(np.float32)
    shuffled_list = gpu_advanced_shuffle(partition_array)
    return shuffled_list.tolist()

def ray_random_shuffle(dataset, num_partitions=10, use_gpu=False):
    partitions = dataset.split(num_partitions)
    if use_gpu:
        shuffle_tasks = [shuffle_partition_gpu.remote(p) for p in partitions]
    else:
        shuffle_tasks = [shuffle_partition_cpu.remote(p) for p in partitions]
    shuffled_partitions = ray.get(shuffle_tasks)
    shuffled_dataset = sum(shuffled_partitions, [])
    return shuffled_dataset

# Benchmarking functions
def benchmark_ray_random_shuffle(data, num_partitions=10, use_gpu=False):
    start_time = time.time()
    dataset = ray.data.from_items(data)
    shuffled_dataset = ray_random_shuffle(dataset, num_partitions, use_gpu)
    end_time = time.time()
    return end_time - start_time

def benchmark_custom_gpu_shuffle(data):
    start_time = time.time()
    shuffled_data = gpu_advanced_shuffle(np.array(data).astype(np.float32))
    end_time = time.time()
    return end_time - start_time

# Initialize Ray with GPU resources
ray.init(ignore_reinit_error=True, num_gpus=1)

# Create a large dataset for benchmarking
data_size = 10**6  # Adjust size as needed
data = list(np.random.rand(data_size).astype(np.float32))

# Benchmark Ray random shuffle with CPU
ray_shuffle_time_cpu = benchmark_ray_random_shuffle(data, use_gpu=False)
print(f"Ray random_shuffle (CPU) time: {ray_shuffle_time_cpu:.4f} seconds")

# Benchmark Ray random shuffle with GPU using the new shuffle
ray_shuffle_time_gpu = benchmark_ray_random_shuffle(data, use_gpu=True)
print(f"Ray random_shuffle (GPU) time: {ray_shuffle_time_gpu:.4f} seconds")

# Benchmark the custom GPU shuffling directly
custom_gpu_shuffle_time = benchmark_custom_gpu_shuffle(data)
print(f"Custom GPU shuffle time: {custom_gpu_shuffle_time:.4f} seconds")

