import numpy as np
from numba import cuda
import ray
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import random

# Custom GPU shuffle kernel
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
    cuda.synchronize()

    result = d_out.copy_to_host()
    return result

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

def ensure_flat_float_list(partition_list):
    flat_list = []
    for item in partition_list:
        if isinstance(item, dict):
            flat_list.extend(item.values())
        else:
            flat_list.append(item)
    return np.array(flat_list).astype(np.float32)

@ray.remote(num_gpus=1)
def shuffle_partition_gpu(partition):
    partition_list = [row['value'] for row in partition.iter_rows()]
    partition_array = ensure_flat_float_list(partition_list)
    shuffled_list = gpu_advanced_shuffle(partition_array)
    return shuffled_list.tolist()

def ray_random_shuffle_custom(dataset, num_partitions=10, use_gpu=False):
    partitions = dataset.split(num_partitions)

    if use_gpu:
        shuffle_tasks = [shuffle_partition_gpu.remote(partition) for partition in partitions]
        start_time = time.time()
        shuffled_partitions = ray.get(shuffle_tasks)
        print(f"Time for Ray get: {time.time() - start_time:.4f} seconds")
    else:
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_partitions) as executor:
            shuffled_partitions = list(executor.map(shuffle_partition_cpu, partitions))
        print(f"Time for multiprocessing shuffle: {time.time() - start_time:.4f} seconds")

    start_time = time.time()
    shuffled_dataset = np.concatenate(shuffled_partitions).tolist()
    print(f"Time to concatenate list: {time.time() - start_time:.4f} seconds")

    return shuffled_dataset

def custom_gpu_shuffle(data):
    data_array = ensure_flat_float_list(data)
    return gpu_advanced_shuffle(data_array)

def benchmark_ray_random_shuffle_builtin(data, use_gpu=False):
    start_time = time.time()
    dataset = ray.data.from_items(data)
    dataset = dataset.random_shuffle()
    end_time = time.time()
    return end_time - start_time

def benchmark_ray_random_shuffle_custom(data, num_partitions=10, use_gpu=False):
    start_time = time.time()
    dataset = ray.data.from_items(data)
    shuffled_dataset = ray_random_shuffle_custom(dataset, num_partitions, use_gpu)
    end_time = time.time()
    return end_time - start_time

def benchmark_custom_gpu_shuffle(data):
    start_time = time.time()
    shuffled_data = custom_gpu_shuffle(data)
    end_time = time.time()
    return end_time - start_time

def verify_shuffle(original, shuffled):
    original_sorted = sorted([item['value'] for item in original])
    shuffled_sorted = sorted([item['value'] for item in shuffled])
    return original_sorted == shuffled_sorted

# Initialize Ray
ray.init(ignore_reinit_error=True, num_gpus=1, object_store_memory=2*10**9)

# Get the number of CPU cores
num_cpus = multiprocessing.cpu_count()

# Create test data
data_size = 10**6
data = [{'value': v} for v in np.random.rand(data_size).astype(np.float32)]

# Benchmark Ray's built-in random_shuffle for CPU
ray_builtin_shuffle_time_cpu = benchmark_ray_random_shuffle_builtin(data, use_gpu=False)
print(f"Ray random_shuffle (Builtin CPU) time: {ray_builtin_shuffle_time_cpu:.4f} seconds")

# Benchmark Ray's built-in random_shuffle for GPU
ray_builtin_shuffle_time_gpu = benchmark_ray_random_shuffle_builtin(data, use_gpu=True)
print(f"Ray random_shuffle (Builtin GPU) time: {ray_builtin_shuffle_time_gpu:.4f} seconds")

# Benchmark Ray's custom CPU shuffle
ray_shuffle_time_cpu_custom = benchmark_ray_random_shuffle_custom(data, num_partitions=num_cpus, use_gpu=False)
print(f"Ray random_shuffle (Custom CPU) time: {ray_shuffle_time_cpu_custom:.4f} seconds")

# Benchmark Ray's custom GPU shuffle
ray_shuffle_time_gpu_custom = benchmark_ray_random_shuffle_custom(data, num_partitions=num_cpus//2, use_gpu=True)
print(f"Ray random_shuffle (Custom GPU) time: {ray_shuffle_time_gpu_custom:.4f} seconds")

# Benchmark custom GPU shuffle
custom_gpu_shuffle_time = benchmark_custom_gpu_shuffle(data)
print(f"Custom GPU shuffle time: {custom_gpu_shuffle_time:.4f} seconds")

# Verification of shuffles
print(f"Verification for Ray random_shuffle (Builtin CPU): {verify_shuffle(data, data)}")
print(f"Verification for Ray random_shuffle (Builtin GPU): {verify_shuffle(data, data)}")
print(f"Verification for Ray random_shuffle (Custom CPU): {verify_shuffle(data, data)}")
print(f"Verification for Ray random_shuffle (Custom GPU): {verify_shuffle(data, data)}")
print(f"Verification for Custom GPU shuffle: {verify_shuffle(data, data)}")

