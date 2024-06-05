import numpy as np
import time
import ray
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import signal

@cuda.jit
def gpu_shuffle_kernel(arr, rng_states):
    """
    GPU kernel function for shuffling an array using random numbers generated from
    xoroshiro128+ RNG states.
    
    Parameters:
    arr (cuda device array): Array to be shuffled.
    rng_states (cuda device array): RNG states for generating random numbers.
    """
    idx = cuda.grid(1)
    if idx < arr.shape[0]:
        xoroshiro128p_uniform_float32(rng_states, idx)  # Advance RNG state
        for i in range(arr.shape[0] - 1, 0, -1):
            j = int(xoroshiro128p_uniform_float32(rng_states, idx) * (i + 1))
            if idx == 0:
                arr[i], arr[j] = arr[j], arr[i]

def gpu_shuffle(arr, threads_per_block):
    """
    Shuffle an array using a GPU-based algorithm.
    
    Parameters:
    arr (numpy array): Array to be shuffled.
    threads_per_block (int): Number of threads per block for the GPU kernel.
    
    Returns:
    tuple: Shuffled array and time taken to shuffle in seconds.
    """
    n = arr.shape[0]
    d_arr = cuda.to_device(arr)
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    rng_states = create_xoroshiro128p_states(threads_per_block * blocks_per_grid, seed=np.random.randint(0, 100000))

    start_time = time.perf_counter()
    gpu_shuffle_kernel[blocks_per_grid, threads_per_block](d_arr, rng_states)
    cuda.synchronize()
    end_time = time.perf_counter()

    shuffled_arr = d_arr.copy_to_host()
    return shuffled_arr, end_time - start_time

def verify_shuffling(original_data, shuffled_data, method="GPU"):
    """
    Verify that the shuffling was performed correctly by comparing sets of original
    and shuffled data.
    
    Parameters:
    original_data (numpy array): Original array before shuffling.
    shuffled_data (numpy array): Array after shuffling.
    method (str): Method used for shuffling, for display purposes.
    
    Returns:
    bool: True if verification passed, False otherwise.
    """
    if isinstance(shuffled_data[0], dict):
        shuffled_data = [v for d in shuffled_data for v in d.values()]
    original_set = set(original_data)
    shuffled_set = set(shuffled_data)
    is_correct = original_set == shuffled_set and len(original_data) == len(shuffled_data)
    print(f"Verification ({method}): {'Passed' if is_correct else 'Failed'}")
    return is_correct

def ray_random_shuffle(dataset, original_data):
    """
    Shuffle an array using Ray's random_shuffle() method.
    
    Parameters:
    dataset (ray.data.Dataset): Dataset to be shuffled.
    original_data (numpy array): Original array before shuffling.
    
    Returns:
    float: Time taken to shuffle in seconds.
    """
    start_time = time.perf_counter()
    shuffled_dataset = dataset.random_shuffle().materialize()
    elapsed_time = time.perf_counter() - start_time
    shuffled_data = list(shuffled_dataset.take_all())
    verify_shuffling(original_data, shuffled_data, "Ray random_shuffle")
    return elapsed_time

def human_readable_size(num_elements, element_size_bytes):
    """
    Convert a size in bytes to a human-readable string in MB or GB.
    
    Parameters:
    num_elements (int): Number of elements in the array.
    element_size_bytes (int): Size of each element in bytes.
    
    Returns:
    str: Human-readable size string.
    """
    size_bytes = num_elements * element_size_bytes
    size_mb = size_bytes / (1024 ** 2)
    size_gb = size_bytes / (1024 ** 3)
    if size_gb >= 1:
        return f"{size_gb:.2f} GB"
    else:
        return f"{size_mb:.2f} MB"

def handler(signum, frame):
    """
    Signal handler for timeout.
    
    Parameters:
    signum (int): Signal number.
    frame (frame object): Current stack frame.
    """
    raise TimeoutError("The function took too long")

def main():
    """
    Main function to test GPU and Ray shuffling algorithms with different data sizes.
    """
    ray.init()  # Initialize Ray
    sizes = [10000000, 100, 1000, 10000, 100000, 1000000]  # Different sizes to test
    threads_per_block = 1024  # Set the number of threads per block
    element_size_bytes = np.dtype(np.int64).itemsize  # Size of each element in bytes (assuming int64)
    results = []

    signal.signal(signal.SIGALRM, handler)  # Set the signal handler for timeout

    for size in sizes:
        print("\nTesting size:", size)
        original_data = np.arange(size)

        # GPU Shuffle
        try:
            signal.alarm(60)  # Set the alarm for 60 seconds
            shuffled_data, gpu_time = gpu_shuffle(original_data.copy(), threads_per_block)
            verify_shuffling(original_data, shuffled_data, "GPU")
        except TimeoutError:
            print("GPU shuffle timed out")
            gpu_time = float('inf')
        finally:
            signal.alarm(0)  # Disable the alarm

        # Ray Shuffle
        try:
            signal.alarm(60)  # Set the alarm for 60 seconds
            dataset = ray.data.from_numpy(original_data.copy())
            ray_time = ray_random_shuffle(dataset, original_data)
        except TimeoutError:
            print("Ray shuffle timed out")
            ray_time = float('inf')
        finally:
            signal.alarm(0)  # Disable the alarm

        data_size_human_readable = human_readable_size(size, element_size_bytes)
        results.append((size, data_size_human_readable, gpu_time, ray_time))

    print("\nResults:")
    print("{:<10} {:<15} {:<15} {:<15}".format("Size", "Data Size", "GPU Time (s)", "Ray Time (s)"))
    for size, data_size_human_readable, gpu_time, ray_time in results:
        print("{:<10} {:<15} {:<15.6f} {:<15.6f}".format(size, data_size_human_readable, gpu_time, ray_time))

    ray.shutdown()  # Shut down Ray

if __name__ == "__main__":
    main()

