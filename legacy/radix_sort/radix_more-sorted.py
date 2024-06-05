import numpy as np
import ray
import math
import time
import os
from numba import cuda

@cuda.jit
def gpu_permutation(arr, indices):
    """
    Perform GPU permutation on the array.
    Args:
        arr (numpy.ndarray): The array to be shuffled.
        indices (numpy.ndarray): The indices used for shuffling.
    Returns:
        None
    """
    idx = cuda.grid(1)
    if idx < arr.size:
        arr[idx], arr[indices[idx]] = arr[indices[idx]], arr[idx]

def shuffle_array(arr):
    """
    Shuffle the array using GPU permutation.
    Args:
        arr (numpy.ndarray): The array to be shuffled.
    Returns:
        numpy.ndarray: The shuffled array.
    """
    if cuda.is_available():
        n = arr.size
        threads_per_block = 256
        blocks_per_grid = math.ceil(n / threads_per_block)
        if blocks_per_grid == 0:
            blocks_per_grid = 1
        indices = np.arange(n)
        np.random.shuffle(indices)
        d_arr = cuda.to_device(arr)
        d_indices = cuda.to_device(indices)
        gpu_permutation[blocks_per_grid, threads_per_block](d_arr, d_indices)
        shuffled_arr = d_arr.copy_to_host()
        return shuffled_arr
    else:
        raise RuntimeError("CUDA is not available on this device.")

@ray.remote(num_gpus=1)
def round_robin_shuffle(arr):
    """
    Perform round-robin shuffling on the array using multiple GPUs.
    Args:
        arr (numpy.ndarray): The array to be shuffled.
    Returns:
        numpy.ndarray: The shuffled array.
    """
    n = arr.size
    cuda.detect()
    num_gpus = len(cuda.gpus)
    chunk_size = math.ceil(n / num_gpus)
    chunks = [arr[i*chunk_size:(i+1)*chunk_size] for i in range(num_gpus)]
    shuffled_chunks = [shuffle_array(chunk) for chunk in chunks]
    shuffled_arr = np.concatenate(shuffled_chunks)
    return shuffled_arr

@ray.remote(num_gpus=1)
def load_balancing_shuffle(arr):
    """
    Perform load balancing shuffling on the array using multiple GPUs.
    Args:
        arr (numpy.ndarray): The array to be shuffled.
    Returns:
        numpy.ndarray: The shuffled array.
    """
    n = arr.size
    cuda.detect()
    num_gpus = len(cuda.gpus)
    loads = np.zeros(num_gpus, dtype=np.int64)
    gpu_assignments = np.zeros(n, dtype=np.int64)
    for i in range(n):
        gpu_idx = np.argmin(loads)
        loads[gpu_idx] += 1
        gpu_assignments[i] = gpu_idx
    shuffled_arr = arr.copy()
    for gpu_idx in range(num_gpus):
        indices = np.where(gpu_assignments == gpu_idx)[0]
        chunk = arr[indices]
        shuffled_chunk = shuffle_array(chunk)
        shuffled_arr[indices] = shuffled_chunk
    return shuffled_arr

@ray.remote(num_gpus=1)
def data_aware_shuffle(arr):
    """
    Perform data-aware shuffling on the array using multiple GPUs.
    Args:
        arr (numpy.ndarray): The array to be shuffled.
    Returns:
        numpy.ndarray: The shuffled array.
    """
    n = arr.size
    cuda.detect()
    num_gpus = len(cuda.gpus)
    chunk_size = math.ceil(n / num_gpus)
    chunks = [arr[i*chunk_size:(i+1)*chunk_size] for i in range(num_gpus)]
    shuffled_chunks = [shuffle_array(chunk) for chunk in chunks]
    data_sizes = [chunk.nbytes for chunk in chunks]
    gpu_assignments = np.argsort(data_sizes)
    shuffled_arr = np.concatenate([shuffled_chunks[i] for i in gpu_assignments])
    return shuffled_arr

@ray.remote(num_gpus=1)
def topology_aware_shuffle(arr):
    """
    Perform topology-aware shuffling on the array using multiple GPUs.
    Args:
        arr (numpy.ndarray): The array to be shuffled.
    Returns:
        numpy.ndarray: The shuffled array.
    """
    n = arr.size
    cuda.detect()
    num_gpus = len(cuda.gpus)
    chunk_size = math.ceil(n / num_gpus)
    chunks = [arr[i*chunk_size:(i+1)*chunk_size] for i in range(num_gpus)]
    shuffled_chunks = [shuffle_array(chunk) for chunk in chunks]
    gpu_distances = np.zeros((num_gpus, num_gpus))
    for i in range(num_gpus):
        for j in range(num_gpus):
            gpu_distances[i, j] = abs(i - j)
    gpu_assignments = np.argsort(gpu_distances, axis=1)
    shuffled_arr = np.concatenate([shuffled_chunks[i] for i in gpu_assignments[0]])
    return shuffled_arr

@ray.remote(num_gpus=1)
def adaptive_shuffle(arr):
    """
    Perform adaptive shuffling on the array using multiple GPUs.
    Args:
        arr (numpy.ndarray): The array to be shuffled.
    Returns:
        numpy.ndarray: The shuffled array.
    """
    n = arr.size
    cuda.detect()
    num_gpus = len(cuda.gpus)
    chunk_size = math.ceil(n / num_gpus)
    chunks = [arr[i*chunk_size:(i+1)*chunk_size] for i in range(num_gpus)]
    shuffled_chunks = [shuffle_array(chunk) for chunk in chunks]
    gpu_loads = np.array([chunk.size for chunk in chunks])
    while True:
        max_load_idx = np.argmax(gpu_loads)
        min_load_idx = np.argmin(gpu_loads)
        if gpu_loads[max_load_idx] - gpu_loads[min_load_idx] <= chunk_size:
            break
        amount = min(gpu_loads[max_load_idx] - gpu_loads[min_load_idx], chunk_size)
        indices = np.random.choice(np.where(gpu_loads == gpu_loads[max_load_idx])[0], size=amount, replace=False)
        gpu_loads[max_load_idx] -= amount
        gpu_loads[min_load_idx] += amount
        shuffled_chunks[min_load_idx] = np.concatenate((shuffled_chunks[min_load_idx], shuffled_chunks[max_load_idx][indices]))
        shuffled_chunks[max_load_idx] = np.delete(shuffled_chunks[max_load_idx], indices)
    shuffled_arr = np.concatenate(shuffled_chunks)
    return shuffled_arr

@ray.remote(num_gpus=1)
def predictive_shuffle(arr):
    """
    Perform predictive shuffling on the array using multiple GPUs.
    Args:
        arr (numpy.ndarray): The array to be shuffled.
    Returns:
        numpy.ndarray: The shuffled array.
    """
    n = arr.size
    cuda.detect()
    num_gpus = len(cuda.gpus)
    chunk_size = math.ceil(n / num_gpus)
    chunks = [arr[i*chunk_size:(i+1)*chunk_size] for i in range(num_gpus)]
    shuffled_chunks = [shuffle_array(chunk) for chunk in chunks]
    gpu_loads = np.array([chunk.size for chunk in chunks])
    gpu_speeds = np.ones(num_gpus)
    for i in range(num_gpus):
        start_time = time.time()
        shuffle_array(chunks[i])
        end_time = time.time()
        gpu_speeds[i] = chunks[i].size / (end_time - start_time)
    predicted_times = gpu_loads / gpu_speeds
    gpu_assignments = np.argsort(predicted_times)
    shuffled_arr = np.concatenate([shuffled_chunks[i] for i in gpu_assignments])
    return shuffled_arr

def read_elements_from_file(filename):
    """
    Read elements from a file and convert them into a numpy array.
    Args:
        filename (str): The path to the file containing the elements.
    Returns:
        numpy.ndarray: The array containing the elements read from the file.
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    abs_filename = os.path.join(script_dir, "data", filename)
    with open(abs_filename, 'r') as file:
        data = [int(line.strip()) for line in file]
    return np.array(data, dtype=np.float32)

def write_elements_to_file(filename, data):
    """
    Write elements to a file.
    Args:
        filename (str): The path to the file where elements will be written.
        data (numpy.ndarray): The array containing the elements to be written.
    Returns:
        None
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    abs_filename = os.path.join(script_dir, "result", filename)
    with open(abs_filename, 'w') as file:
        for element in data:
            file.write(f"{int(element)}\n")

def test_shuffled_output(original_data, shuffled_data):
    """
    Test the correctness of the shuffled output.
    Args:
        original_data (numpy.ndarray): The original input array.
        shuffled_data (numpy.ndarray): The shuffled output array.
    Returns:
        bool: True if the shuffled output is correct, False otherwise.
    """
    if len(original_data) != len(shuffled_data):
        print("Length mismatch between original and shuffled data.")
        return False

    if set(original_data) != set(shuffled_data):
        print("Elements in the shuffled array do not match the original array.")
        return False

    if np.array_equal(original_data, shuffled_data):
        print("Shuffled array is identical to the original array.")
        return False

    if np.array_equal(shuffled_data, np.sort(shuffled_data)):
        print("Shuffled array is sorted.")
        return False

    print("Shuffled output passed the tests.")
    return True

if __name__ == "__main__":
    ray.init(address="auto")

    input_file = 'element_100000.txt'
    output_file = 'shuffled_elements.txt'

    data = read_elements_from_file(input_file)

    algorithms = [
        ("Round-Robin Shuffle", round_robin_shuffle),
        ("Load Balancing Shuffle", load_balancing_shuffle),
        ("Data-Aware Shuffle", data_aware_shuffle),
        ("Topology-Aware Shuffle", topology_aware_shuffle),
        ("Adaptive Shuffle", adaptive_shuffle),
        ("Predictive Shuffle", predictive_shuffle)
    ]

    for algorithm_name, shuffle_function in algorithms:
        start_time = time.time()
        shuffled_data = ray.get(shuffle_function.remote(data))
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution Time for {algorithm_name}: {execution_time:.4f} seconds")

        if test_shuffled_output(data, shuffled_data):
            write_elements_to_file(f"{algorithm_name.lower().replace(' ', '_')}_output.txt", shuffled_data)
        else:
            print(f"Error: Shuffled output for {algorithm_name} is incorrect.")

        write_elements_to_file(f"{algorithm_name.lower().replace(' ', '_')}_output.txt", shuffled_data)

    ray.shutdown()
