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
        temp = arr[idx]
        target_idx = indices[idx]
        arr[idx] = arr[target_idx]
        arr[target_idx] = temp

@ray.remote(num_gpus=1)
def shuffle_array(arr):
    """
    Perform array shuffling using Ray with GPU support.

    Args:
        arr (numpy.ndarray): The array to be shuffled.

    Returns:
        numpy.ndarray: The shuffled array.
    """
    if cuda.is_available():
        n = arr.size
        threads_per_block = 256
        blocks_per_grid = math.ceil(n / threads_per_block)

        # Check if blocks_per_grid is zero
        if blocks_per_grid == 0:
            blocks_per_grid = 1  # Set at least one block

        indices = np.arange(n)
        np.random.shuffle(indices)

        d_arr = cuda.to_device(arr)
        d_indices = cuda.to_device(indices)

        gpu_permutation[blocks_per_grid, threads_per_block](d_arr, d_indices)

        shuffled_arr = d_arr.copy_to_host()
        return shuffled_arr
    else:
        raise RuntimeError("CUDA is not available on this device.")

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

if __name__ == "__main__":
    ray.init(address="auto")

    input_file = 'element_100000.txt'
    output_file = 'shuffled_elements.txt'

    start_time = time.time()

    data = read_elements_from_file(input_file)

    read_time = time.time()
    #print(f"Time to read data: {read_time - start_time:.4f} seconds")

    shuffled_data = ray.get(shuffle_array.remote(data))

    shuffle_time = time.time()
    #print(f"Time to shuffle data: {shuffle_time - read_time:.4f} seconds")

    write_elements_to_file(output_file, shuffled_data)

    write_time = time.time()
    #print(f"Time to write data: {write_time - shuffle_time:.4f} seconds")
    print(f"Total Execution Time for Radix Shuffle Sort: {write_time - start_time:.4f} seconds")

    ray.shutdown()

