import time
import os
import ray
import cupy as cp
import numpy as np
from numba import cuda

# Initialize Ray
ray.init()

def check_cuda_device():
    try:
        cuda.select_device(0)
        print("CUDA-capable device detected.")
        return True
    except cuda.cudadrv.error.CudaSupportError:
        print("No CUDA-capable device detected.")
        return False

@cuda.jit(cache=True, opt=True)
def gpu_shuffle_kernel(data, size):
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.gridDim.x
    stride = cuda.blockDim.x * cuda.gridDim.x
    for i in range(tid, size, stride):
        j = cuda.random.xoroshiro128p_uniform32() % size
        data[i], data[j] = data[j], data[i]

def setup_cuda_environment():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Make sure this matches your CUDA device ID
    os.environ["CUDA_HOME"] = "/usr/local/cuda"
    os.environ["PATH"] += os.pathsep + "/usr/local/cuda/bin"
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"
    print("CUDA environment set up.")

@ray.remote
def local_shuffle(chunk):
    setup_cuda_environment()
    # Convert the data to GPU memory (CuPy array)
    gpu_data = cp.array(chunk)
    size = gpu_data.size

    # Define the CUDA kernel configuration
    threads_per_block = 256
    blocks_per_grid = (size + threads_per_block - 1) // threads_per_block

    # Launch the CUDA kernel for shuffling
    gpu_shuffle_kernel[blocks_per_grid, threads_per_block](gpu_data, size)
    
    # Return the shuffled chunk
    return gpu_data.get()

@ray.remote
def inter_gpu_shuffle(chunk, peer_chunk):
    setup_cuda_environment()
    # Concatenate chunks from different GPUs
    combined_chunk = cp.concatenate((chunk, peer_chunk))
    size = combined_chunk.size

    # Define the CUDA kernel configuration
    threads_per_block = 256
    blocks_per_grid = (size + threads_per_block - 1) // threads_per_block

    # Launch the CUDA kernel for shuffling
    gpu_shuffle_kernel[blocks_per_grid, threads_per_block](combined_chunk, size)
    
    return combined_chunk.get()

def topology_aware_shuffle(dataset, num_chunks, num_gpus):
    # Partition dataset into chunks
    data_chunks = [dataset[i::num_chunks] for i in range(num_chunks)]
    
    # Perform local shuffling on each chunk using GPUs
    shuffled_chunks = ray.get([local_shuffle.remote(chunk) for chunk in data_chunks])

    # Pair chunks for inter-GPU shuffling
    paired_chunks = [(shuffled_chunks[i], shuffled_chunks[(i + 1) % num_gpus]) for i in range(num_gpus)]
    inter_shuffled_chunks = ray.get([inter_gpu_shuffle.remote(chunk1, chunk2) for chunk1, chunk2 in paired_chunks])
    
    # Concatenate shuffled chunks to form the final shuffled dataset
    final_shuffled_dataset = cp.concatenate(inter_shuffled_chunks)
    
    return final_shuffled_dataset

def benchmark_shuffling_algorithms(dataset, num_chunks, num_gpus):
    # Benchmark topology-aware shuffle
    start_time = time.time()
    shuffled_dataset_topology = topology_aware_shuffle(dataset, num_chunks, num_gpus)
    topology_shuffle_time = time.time() - start_time
    print(f"Topology-aware shuffle time: {topology_shuffle_time:.4f} seconds")

    # Convert dataset to Ray Dataset
    ray_dataset = ray.data.from_numpy(dataset)

    # Benchmark Ray's random_shuffle
    start_time = time.time()
    shuffled_dataset_ray = ray_dataset.random_shuffle().take(len(dataset)).to_numpy()
    ray_shuffle_time = time.time() - start_time
    print(f"Ray Dataset random_shuffle time: {ray_shuffle_time:.4f} seconds")

    return topology_shuffle_time, ray_shuffle_time

if __name__ == "__main__":
    if check_cuda_device():
        dataset = cp.arange(1e6)  # Example dataset
        num_chunks = 4  # Number of chunks to partition the data into
        num_gpus = 2  # Number of GPUs available

        topology_shuffle_time, ray_shuffle_time = benchmark_shuffling_algorithms(dataset, num_chunks, num_gpus)
        print(f"Topology-aware shuffle time: {topology_shuffle_time:.4f} seconds")
        print(f"Ray Dataset random_shuffle time: {ray_shuffle_time:.4f} seconds")
    else:
        print("Exiting: No CUDA-capable device detected.")
