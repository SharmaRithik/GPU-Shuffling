import ray
import time
import os
import numpy as np

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

    # Only bother checking shuffle percentage if under certain length
    if len(original_data) < 1000000:
        num_shuffled = 0
        for i in range(len(original_data)):
            if original_data[i] != shuffled_data[i]:
                num_shuffled += 1
        print(f"% of items shuffled: {num_shuffled / len(original_data) * 100}%")

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
    shuffled_data = list(shuffled_dataset.take_all())
    elapsed_time = time.perf_counter() - start_time
    return elapsed_time, shuffled_data

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

def main():
    """
    Main function to test GPU and Ray shuffling algorithms with different data sizes.
    """
    #sizes = [100000, 500000, 1000000, 10000000]
    sizes = [100000, 500000, 1000000, 10000000]
    element_size_bytes = np.dtype(np.int64).itemsize  # Size of each element in bytes (assuming int64)
    results = []

    sizes = [sizes[0]] + sizes[:]

    for s in range(len(sizes)):
        print("\nTesting size:", sizes[s])
        original_data = np.arange(sizes[s])

        # GPU Shuffle
        os.environ['RAY_DATA_GPU_SHUFFLE'] = '1'
        ray.init()
        dataset = ray.data.from_numpy(original_data.copy())
        gpu_time, shuffled_data = ray_random_shuffle(dataset, original_data)
        verify_shuffling(original_data, shuffled_data, "GPU random_shuffle")
        ray.shutdown()

        # Ray Shuffle
        os.environ['RAY_DATA_GPU_SHUFFLE'] = '0'
        ray.init()
        dataset = ray.data.from_numpy(original_data.copy())
        ray_time, shuffled_data = ray_random_shuffle(dataset, original_data)
        verify_shuffling(original_data, shuffled_data, "Ray random_shuffle")
        ray.shutdown()

        data_size_human_readable = human_readable_size(sizes[s], element_size_bytes)
        
        if s > 0: # ignoring first run, used to force compile of numba kernel
            results.append((sizes[s], data_size_human_readable, gpu_time, ray_time))

    print("\nResults:")
    print("{:<10} {:<15} {:<15} {:<15}".format("Size", "Data Size", "GPU Time (s)", "Ray Time (s)"))
    for size, data_size_human_readable, gpu_time, ray_time in results:
        print("{:<10} {:<15} {:<15.6f} {:<15.6f}".format(size, data_size_human_readable, gpu_time, ray_time))

if __name__ == "__main__":
    main()

